import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import transformers
import wandb
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, HfArgumentParser, TrainingArguments

from utils.metrics import calculate_metric
from utils.tasks import get_task
from utils.utils import logger, count_time, encode_prompt, Prediction, write_metrics_to_file


@dataclass
class OurArguments(TrainingArguments):
    wandb_project: str = "test"
    task_name: str = "DROP"
    overwrite_output_dir: bool = True
    output_dir: str = './trained_models/test'
    # Number of examples
    num_train: int = 0
    num_dev: int = None
    num_eval: int = None
    num_train_sets: int = None
    train_set_seed: int = None
    seed: int = 42
    result_file: str = None
    weight_decay: float = 0.0

    # Model loading
    model_name: str = "meta-llama/Llama-2-7b-hf"  # HuggingFace model name
    load_float16: bool = False  # load model parameters as float16
    load_bfloat16: bool = True  # load model parameters as bfloat16
    load_int8: bool = False  # load model parameters as int8
    max_length: int = 2048  # max length the model can take

    # Training
    trainer: str = "none"
    only_train_option: bool = True  # whether to only train the option part of the input
    train_as_classification: bool = False  # take the log likelihood of all options and train as classification

    # parameter setup for PEFT methods
    tuning_type: str = 'ft'
    # QuanTA
    quanta_d: int = 4
    quanta_per_dim_features: List[int] = (16, 8, 8, 4)
    # quanta_per_dim_features2: Tuple[int] = (16, 8, 4, 2)
    quanta_per_dim_features2: Tuple[int] = None
    quanta_sum_mode: bool = False
    quanta_initialize_mode: str = 'sum_opposite_freeze_one'
    quanta_dropout: float = 0.0
    target_modules: List[str] = None

    # Generation
    sampling: bool = False  # whether to use sampling
    temperature: float = 1.0  # temperature for generation
    num_beams: int = 1  # number of beams for generation
    top_k: int = None  # top-k for generation
    top_p: float = 0.95  # top-p for generation
    max_new_tokens: int = 50  # max number of new tokens to generate
    eos_token: str = "\n"  # end of sentence token

    # Saving
    save_model: bool = False  # whether to save the model
    no_eval: bool = False  # whether to skip evaluation
    tag: str = ""  # saving tag

    # Linear probing
    linear_probing: bool = False  # whether to do linear probing
    lp_early_stopping: bool = False  # whether to do early stopping in linear probing

    # Display
    verbose: bool = False  # verbose output

    # Non-diff objective
    non_diff: bool = False  # use non-differentiable objective (only support F1 for SQuAD for now)

    # Auto saving when interrupted
    save_on_interrupt: bool = False  # save model when interrupted (useful for long training)
    checkpoint_dir: str = None  # directory to load the checkpoint


def parse_args():
    parser = HfArgumentParser(OurArguments)
    args = parser.parse_args_into_dataclasses()[0]
    return args


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters()) / 1000 / 1000
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad) / 1000 / 1000
    wandb.log({"Total(M)": total_num, "Trainable(M)": trainable_num})
    return {'Total(M)': total_num, 'Total Trainable(M)': trainable_num}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)


class Framework:

    def __init__(self, args, task):
        self.args = args
        self.task = task
        self.model, self.tokenizer = self.load_model()

    def load_model(self):
        with count_time("Loading model with FP%d" % (16 if self.args.load_float16 else 32)):
            config = AutoConfig.from_pretrained(self.args.model_name)
            torch_dtype = torch.float32
            if self.args.load_float16:
                torch_dtype = torch.float16
            elif self.args.load_bfloat16:
                torch_dtype = torch.bfloat16
            model = AutoModelForCausalLM.from_pretrained(self.args.model_name, config=config, device_map='auto',
                                                         torch_dtype=torch_dtype, load_in_8bit=self.args.load_int8, )
            model.eval()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, use_fast=False)

        # HF tokenizer bug fix
        if "llama" in self.args.model_name:
            # LLaMA padding token
            tokenizer.pad_token_id = 0  # technically <unk>

        def find_first_checkpoint(checkpoint_dir):
            all_items = os.listdir(checkpoint_dir)
            checkpoints = [item for item in all_items if
                           "checkpoint" in item and os.path.isdir(os.path.join(checkpoint_dir, item))]
            checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
            if checkpoints:
                return checkpoints[0]
            else:
                return None

        actual_checkpoint_dir = os.path.join(self.args.checkpoint_dir, find_first_checkpoint(self.args.checkpoint_dir))

        if self.args.tuning_type == 'quanta':
            from quanta import QuanTAConfig, get_peft_model
            peft_config = QuanTAConfig(d=self.args.quanta_d, per_dim_features=self.args.quanta_per_dim_features,
                                       per_dim_features2=self.args.quanta_per_dim_features2, merge_weights=True,
                                       target_modules=self.args.target_modules, sum_mode=self.args.quanta_sum_mode,
                                       initialize_mode=self.args.quanta_initialize_mode, bias="none",
                                       task_type="CAUSAL_LM", quanta_dropout=self.args.quanta_dropout, )
            model = get_peft_model(model, peft_config)
            model.load_state_dict(torch.load(actual_checkpoint_dir + '/pytorch_model.bin'), strict=False)

        if self.args.tuning_type == 'ft':
            try:
                state_dicts = [torch.load(actual_checkpoint_dir + '/pytorch_model-00001-of-00003.bin'),
                               torch.load(actual_checkpoint_dir + '/pytorch_model-00002-of-00003.bin'),
                               torch.load(actual_checkpoint_dir + '/pytorch_model-00003-of-00003.bin')]
                state_dict = {}
                for sd in state_dicts:
                    for k, v in sd.items():
                        state_dict[k] = v
                model.load_state_dict(state_dict)
            except Exception as e:
                print(f'error {e}, loading from {actual_checkpoint_dir} failed, running base model instead')
        # print the name and shape of trainable parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.shape} parameters")
        logger.info("Total Parameter Count: {}M".format(model.num_parameters() / 1000 / 1000))
        logger.info("Total and trainable params: {}".format(str(get_parameter_number(model))))
        return model, tokenizer

    def forward(self, input_ids, option_len=None, generation=False):
        input_ids = torch.tensor([input_ids]).to(self.model.device)

        if generation:
            args = self.args
            outputs = self.model.generate(input_ids=input_ids, do_sample=args.sampling, temperature=args.temperature,
                                          num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k,
                                          max_new_tokens=min(args.max_new_tokens, args.max_length - input_ids.size(1)),
                                          num_return_sequences=1, eos_token_id=[
                    self.tokenizer.encode(args.eos_token, add_special_tokens=False)[-1], self.tokenizer.eos_token_id], )
            output_text = self.tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True).strip()
            return output_text
        else:
            with torch.inference_mode():
                self.model.eval()
                logits = self.model(input_ids=input_ids).logits
            labels = input_ids[0, 1:]
            logits = logits[0, :-1]
            log_probs = F.log_softmax(logits, dim=-1)

            selected_log_probs = log_probs[torch.arange(len(labels)).to(labels.device), labels]
            selected_log_probs = selected_log_probs.cpu().detach()
            return selected_log_probs[-option_len:]

    def one_step_pred(self, train_samples, eval_sample, verbose=False):
        verbose = verbose or self.args.verbose
        if verbose:
            logger.info("========= Example =========")
            logger.info(f"Candidate: {eval_sample.candidates}")
            logger.info(f"Correct candidate: {eval_sample.correct_candidate}")

        encoded_candidates, option_lens = encode_prompt(self.task, self.task.get_template(), train_samples, eval_sample,
                                                        self.tokenizer, max_length=self.args.max_length,
                                                        generation=self.task.generation,
                                                        max_new_tokens=self.args.max_new_tokens)

        outputs = []
        if self.task.generation:
            # For generation tasks, return the autoregressively-generated text
            output_text = self.forward(encoded_candidates[0], generation=True)
            if verbose:
                logger.info("=== Prompt ===")
                logger.info(self.tokenizer.decode(encoded_candidates[0]))
                logger.info(f"Output: {output_text}")
            return Prediction(correct_candidate=eval_sample.correct_candidate, predicted_candidate=output_text)
        else:
            # For classification/multiple-choice, calculate the probabilities of all candidates
            for candidate_id, encoded_candidate in enumerate(encoded_candidates):
                selected_log_probs = self.forward(encoded_candidate, option_len=option_lens[candidate_id])
                if verbose:
                    if candidate_id == 0:
                        logger.info("=== Candidate %d ===" % candidate_id)
                        logger.info(self.tokenizer.decode(encoded_candidate))
                    else:
                        logger.info("=== Candidate %d (without context)===" % candidate_id)
                        logger.info(self.tokenizer.decode(encoded_candidate).split(self.task.train_sep)[-1])
                    logger.info(f"Log probabilities of the option tokens: {selected_log_probs}")

                outputs.append({"log_probs": selected_log_probs, "sfc_log_probs": None})

            # (Default) length-normalized log probabilities
            # log p(candidate | input) = log p_lm(candidate | input) / |candidate #tokens|
            scores = [x['log_probs'].mean().item() for x in outputs]

            if verbose:
                logger.info(f"Prediction scores: {scores}")

            if isinstance(eval_sample.correct_candidate, list):
                # For some datasets there are multiple correct answers
                correct_candidate_id = [eval_sample.candidates.index(c) for c in eval_sample.correct_candidate]
            else:
                correct_candidate_id = eval_sample.candidates.index(eval_sample.correct_candidate)

            return Prediction(correct_candidate=correct_candidate_id, predicted_candidate=int(np.argmax(scores)))

    def evaluate(self, train_samples, eval_samples, one_train_set_per_eval_sample=False):
        """
        Evaluate function. If one_train_set_per_eval_sample is True, then each eval sample has its own training (demonstration) set.
        """
        self.model.eval()  # may not be necessary
        with torch.no_grad():
            if one_train_set_per_eval_sample:
                logger.info(f"There are {len(eval_samples)} validation samples and one train set per eval sample")
            else:
                logger.info(
                    f"There are {len(train_samples)} training samples and {len(eval_samples)} validation samples")
            # Prediction loop
            predictions = []
            for eval_id, eval_sample in enumerate(tqdm(eval_samples)):
                predictions.append(
                    self.one_step_pred(train_samples[eval_id] if one_train_set_per_eval_sample else train_samples,
                                       eval_sample, verbose=(eval_id < 10)))

            # Calculate metrics 
            metric_name = getattr(self.task, "metric_name", "accuracy")
            metrics = {metric_name: calculate_metric(predictions, metric_name)}
        self.model.train()  # may not be necessary
        return metrics


def result_file_tag(args):
    save_model_name = args.model_name.split("/")[-1]
    sample_eval_tag = "-sampleeval%d" % args.num_eval if args.num_eval is not None else ""
    sample_train_tag = "-ntrain%d" % args.num_train if args.num_train > 0 else ""
    sample_dev_tag = "-ndev%d" % args.num_dev if args.num_dev is not None else ""
    customized_tag = f"-{args.tag}" if len(args.tag) > 0 else ""
    return f"{args.task_name}-{save_model_name}" + sample_eval_tag + sample_train_tag + sample_dev_tag + customized_tag


def main():
    args = parse_args()
    args.save_safetensors = False
    set_seed(args.seed)
    task = get_task(args.task_name)
    train_sets, dev_sets, test_sets = task.sample_train_sets(num_train=args.num_train, num_dev=args.num_dev,
                                                             num_eval=args.num_eval, num_train_sets=args.num_train_sets,
                                                             seed=args.train_set_seed)
    wandb_run_name = args.tag.replace('/', '-').replace(' ', '_')
    wandb.init(project=args.wandb_project, name=wandb_run_name)
    framework = Framework(args, task)
    if args.train_set_seed is not None or args.num_train_sets is not None:

        train_set_seed = args.train_set_seed
        train_samples = train_sets[0] if len(train_sets) > 0 else []
        dev_samples = dev_sets[0] if len(dev_sets) > 0 else []
        test_samples = test_sets[0]

        metrics = framework.evaluate([], test_samples)  # No in-context learning if there is training
        if dev_samples is not None and len(dev_samples) > 0:
            dev_metrics = framework.evaluate([], dev_samples)
            for m in dev_metrics:
                metrics["dev_" + m] = dev_metrics[m]

        if not args.no_eval:
            logger.info("===== Train set %d =====" % train_set_seed)
            logger.info(metrics)
            if args.local_rank <= 0:
                write_metrics_to_file(metrics, "result/" + result_file_tag(
                    args) + ".json" if args.result_file is None else args.result_file)
                wandb.log(metrics)


if __name__ == "__main__":
    main()
