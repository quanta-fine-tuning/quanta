import math
import os
import random
import time
from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import transformers
import wandb
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import TrainerCallback, Trainer, AutoConfig, AutoTokenizer, AutoModelForCausalLM, HfArgumentParser, \
    TrainingArguments, DataCollatorForTokenClassification, TrainerState, TrainerControl
from transformers.trainer_utils import speed_metrics

from utils.metrics import calculate_metric
from utils.tasks import get_task
from utils.utils import logger, count_time, encode_prompt, Prediction, forward_wrap_with_option_len, \
    write_metrics_to_file, NondiffCollator, SIGUSR1Callback, DataCollatorWithPaddingAndNesting


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
    only_train_option: bool = True
    train_as_classification: bool = False

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
    save_model: bool = False
    no_eval: bool = False
    tag: str = ""  # saving tag

    # Linear probing
    linear_probing: bool = False
    lp_early_stopping: bool = False

    # Display
    verbose: bool = False

    # Non-diff objective
    non_diff: bool = False

    # Auto saving when interrupted
    save_on_interrupt: bool = False


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

        if self.args.tuning_type == 'quanta':
            from quanta import QuanTAConfig, get_peft_model
            peft_config = QuanTAConfig(d=self.args.quanta_d, per_dim_features=self.args.quanta_per_dim_features,
                                       per_dim_features2=self.args.quanta_per_dim_features2, merge_weights=True,
                                       target_modules=self.args.target_modules, sum_mode=self.args.quanta_sum_mode,
                                       initialize_mode=self.args.quanta_initialize_mode, bias="none",
                                       task_type="CAUSAL_LM", quanta_dropout=self.args.quanta_dropout, )
            model = get_peft_model(model, peft_config)
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
                                       eval_sample, verbose=(eval_id < 3)))

            # Calculate metrics 
            metric_name = getattr(self.task, "metric_name", "accuracy")
            metrics = {metric_name: calculate_metric(predictions, metric_name)}
        self.model.train()  # may not be necessary
        return metrics

    def train(self, train_samples, eval_samples):
        """
        Training function
        """
        # Set tokenizer to left padding (so that all the options are right aligned)
        self.tokenizer.padding_side = "left"

        class HFDataset(Dataset):

            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        def _convert(samples):
            """
            Convert samples to HF-compatible dataset
            """
            data = []
            for sample in samples:
                encoded_candidates, option_lens = encode_prompt(self.task, self.task.get_template(), [], sample,
                                                                self.tokenizer, max_length=self.args.max_length,
                                                                generation=self.task.generation,
                                                                generation_with_gold=True,
                                                                max_new_tokens=self.args.max_new_tokens)
                if self.task.generation:
                    correct_candidate_id = 0
                elif isinstance(sample.correct_candidate, list):
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate[0])
                else:
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate)

                if self.args.non_diff:
                    encoded_candidates[correct_candidate_id] = encoded_candidates[correct_candidate_id][
                                                               :-option_lens[correct_candidate_id]]

                if self.args.train_as_classification:
                    data.append([{"input_ids": encoded_candidates[_i], "labels": correct_candidate_id,
                                  "option_len": option_lens[_i], "num_options": len(sample.candidates)} for _i in
                                 range(len(encoded_candidates))])
                elif self.args.only_train_option:
                    if self.args.non_diff:
                        data.append({"input_ids": encoded_candidates[correct_candidate_id],
                                     "labels": encoded_candidates[correct_candidate_id],
                                     "option_len": option_lens[correct_candidate_id], "gold": sample.correct_candidate})
                    else:
                        data.append({"input_ids": encoded_candidates[correct_candidate_id],
                                     "labels": encoded_candidates[correct_candidate_id],
                                     "option_len": option_lens[correct_candidate_id]})
                else:
                    data.append({"input_ids": encoded_candidates[correct_candidate_id],
                                 "labels": encoded_candidates[correct_candidate_id]})
                if getattr(self.task, "metric_name", "accuracy") == "math":
                    data[-1]["input_ids"] = data[-1]["input_ids"]
            return data

        with count_time("Tokenizing training samples"):
            train_dataset = HFDataset(_convert(train_samples))
            eval_dataset = HFDataset(_convert(eval_samples))

        if self.args.only_train_option and not self.args.non_diff:
            self.model.original_forward = self.model.forward
            self.model.forward = forward_wrap_with_option_len.__get__(self.model, type(self.model))

        if self.args.non_diff:
            collator = NondiffCollator
        else:
            collator = DataCollatorForTokenClassification

        class ModelInfoCallback(TrainerCallback):
            def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
                logger.info(
                    f"New model checkpoint saved: {state.global_step}, best model checkpoint: {state.best_model_checkpoint}")

            def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
                if args.load_best_model_at_end:
                    logger.info(f"Best model loaded at end of training: {state.best_model_checkpoint}")

        class MyTrainer(Trainer):

            def evaluate(self_trainer, eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
                         ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "eval", ) -> Dict[
                str, float]:
                ### NOTE: self is different from self_trainer!!! ###
                self.model.eval()  # may not be necessary
                with torch.no_grad():
                    # handle multipe eval datasets
                    eval_dataset = eval_dataset if eval_dataset is not None else self_trainer.eval_dataset
                    if isinstance(eval_dataset, dict):
                        metrics = {}
                        for eval_dataset_name, _eval_dataset in eval_dataset.items():
                            dataset_metrics = self_trainer.evaluate(eval_dataset=_eval_dataset, ignore_keys=ignore_keys,
                                                                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}", )
                            metrics.update(dataset_metrics)
                        return metrics

                    # memory metrics - must set up as early as possible
                    self_trainer._memory_tracker.start()

                    eval_dataloader = self_trainer.get_eval_dataloader(eval_dataset)

                    start_time = time.time()

                    eval_loop = self_trainer.prediction_loop if self_trainer.args.use_legacy_prediction_loop else self_trainer.evaluation_loop
                    output = eval_loop(eval_dataloader, description="Evaluation",
                                       # No point gathering the predictions if there are no metrics, otherwise we defer to
                                       # self_trainer.args.prediction_loss_only
                                       prediction_loss_only=True if self_trainer.compute_metrics is None else None,
                                       ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix, )

                    total_batch_size = self_trainer.args.eval_batch_size * self_trainer.args.world_size
                    if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
                        start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
                    output.metrics.update(speed_metrics(metric_key_prefix, start_time, num_samples=output.num_samples,
                                                        num_steps=math.ceil(output.num_samples / total_batch_size), ))

                    dev_metrics = self.evaluate([], eval_samples)

                    # Update result dictionary with custom metrics
                    for m in dev_metrics:
                        output.metrics[metric_key_prefix + '_' + m] = dev_metrics[m]

                    self_trainer.log(output.metrics)

                    self_trainer.control = self_trainer.callback_handler.on_evaluate(self_trainer.args,
                                                                                     self_trainer.state,
                                                                                     self_trainer.control,
                                                                                     output.metrics)

                    self_trainer._memory_tracker.stop_and_update_metrics(output.metrics)

                self.model.train()  # may not be necessary
                return output.metrics

        self.args.metric_for_best_model = 'eval_' + getattr(self.task, "metric_name", "accuracy")
        self.args.greater_is_better = True

        trainer = MyTrainer(model=self.model, args=self.args, train_dataset=train_dataset, eval_dataset=eval_dataset,
                            tokenizer=self.tokenizer, data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer,
                                                                                                      pad_to_multiple_of=8) if self.args.train_as_classification else collator(
                self.tokenizer, pad_to_multiple_of=8), )
        trainer.add_callback(ModelInfoCallback())

        if self.args.save_on_interrupt:
            trainer.add_callback(SIGUSR1Callback())

        # Resume training from a last checkpoint
        last_checkpoint = None
        from transformers.trainer_utils import get_last_checkpoint
        if os.path.isdir(self.args.output_dir) and not self.args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(self.args.output_dir)
        if last_checkpoint is not None and self.args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                        "the `--output_dir` or add `--overwrite_output_dir` to train from scratch.")
        if self.args.resume_from_checkpoint is not None:
            last_checkpoint = self.args.resume_from_checkpoint

        trainer.train(resume_from_checkpoint=last_checkpoint)

        # Explicitly save the model
        if self.args.save_model:
            logger.warn("Save model..")
            trainer.save_model()

        # FSDP compatibility
        self.model = trainer.model

        # Reset the forward function for evaluation
        if self.args.only_train_option and not self.args.non_diff:
            if type(self.model) == FSDP:
                logger.info("This is an FSDP model now. Be careful when assigning back the original forward function")
                self.model._fsdp_wrapped_module.forward = self.model._fsdp_wrapped_module.original_forward
            else:
                self.model.forward = self.model.original_forward


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
    # training with huggingface trainer
    framework = Framework(args, task)
    if args.train_set_seed is not None or args.num_train_sets is not None:

        train_set_seed = args.train_set_seed
        train_samples = train_sets[0]
        dev_samples = dev_sets[0]
        test_samples = test_sets[0]

        # Training
        framework.train(train_samples, dev_samples)

        # if not args.no_eval:
        metrics = framework.evaluate([], test_samples)  # No in-context learning if there is training
        if dev_samples is not None:
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
