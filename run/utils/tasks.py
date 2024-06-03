import json
import logging
import re
import sys
from dataclasses import dataclass
from typing import List, Union

import numpy as np
from datasets import load_dataset

from .templates import *
from .utils import temp_seed

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_task(task_name):
    aa = task_name.split("__")
    if len(aa) == 2:
        task_group, subtask = aa
    else:
        task_group = aa[0]
        subtask = None
    class_ = getattr(sys.modules[__name__], f"{task_group}Dataset")
    instance = class_(subtask)
    return instance


@dataclass
class Sample:
    id: int = None
    data: dict = None
    correct_candidate: Union[str, List[str]] = None
    candidates: List[str] = None


class Dataset:
    mixed_set = False
    train_sep = "\n\n"
    generation = False  # whether this is a generation task

    def __init__(self, subtask=None, **kwargs) -> None:
        self.subtask = subtask

    def get_task_name(self):
        return self.subtask

    def load_dataset(self):
        raise NotImplementedError

    def get_template(self, template_version=0):
        templates = {0: Template}
        return templates[template_version]

    def build_sample(self, example):
        return

    def sample_train_sets(self, num_train=32, num_dev=None, num_eval=None, num_train_sets=None, seed=None):
        if seed is not None:
            # one train/demo set using the designated seed
            seeds = [seed]
        elif num_train_sets is not None:
            # num_train_sets train/demo sets
            seeds = list(range(num_train_sets))
        else:
            # one train/demo set per evaluation sample
            assert num_dev is None  # not supported
            len_valid_samples = len(self.samples["valid"]) if num_eval is None else num_eval
            with temp_seed(0):
                seeds = np.random.randint(0, 10000, len_valid_samples)

        train_samples = []
        dev_samples = []
        test_samples = []
        for i, set_seed in enumerate(seeds):
            if self.mixed_set:
                raise NotImplementedError
                train_samples.append(self.sample_subset(data_split="valid", seed=set_seed, num=num_train, exclude=i))
            else:
                if num_dev is not None and num_dev > 0:
                    if 'test' not in self.samples:
                        train_sample = self.sample_subset(data_split="train", seed=set_seed,
                                                          num=num_train + num_dev)  # dev set is included at the end of train set
                        if num_train + num_dev > len(self.samples["train"]):
                            logger.warn(
                                f"{num_train + num_dev=} > available training examples {len(self.samples['train'])}, will prioritize the dev set")  # num_train = int(len(self.samples["train"]) * num_train / (num_train + num_dev))  # num_dev = len(self.samples["train"]) - num_train
                        train_samples.append(train_sample[:-num_dev])
                        dev_samples.append(train_sample[-num_dev:])
                    else:
                        train_samples.append(self.sample_subset(data_split="train", seed=set_seed, num=num_train))
                        dev_samples.append(self.sample_subset(data_split="valid", seed=set_seed + 1, num=num_dev))
                else:
                    if num_train is not None and num_train > 0:
                        train_samples.append(self.sample_subset(data_split="train", seed=set_seed, num=num_train))
                    else:
                        train_samples.append([])
                if num_eval is not None and num_eval > 0:
                    if 'test' not in self.samples:
                        test_samples.append(self.sample_subset(data_split="valid", seed=set_seed + 2, num=num_eval))
                    else:
                        test_samples.append(self.sample_subset(data_split="test", seed=set_seed + 2, num=num_eval))
                else:
                    test_samples.append([])

                # if num_dev is not None:
                #     logger.info(f"Sample train set {len(train_samples[-1])}/{len(self.samples['train'])}")
                #     logger.info(f"... including dev set {num_dev} samples")
                if num_train is not None and num_train > 0:
                    logger.info(f"Sample train set {len(train_samples[-1])}/{len(self.samples['train'])}")
                if num_dev is not None and num_dev > 0:
                    if 'test' not in self.samples:
                        logger.info(
                            f"Sample dev set {len(dev_samples[-1])} from train set {len(self.samples['train'])} because no test set")
                    else:
                        logger.info(
                            f"Sample dev set {len(dev_samples[-1])}/{len(self.samples['valid'])} from valid set")
                if num_eval is not None and num_eval > 0:
                    if 'test' not in self.samples:
                        logger.info(
                            f"Sample eval set {len(test_samples[-1])}/{len(self.samples['valid'])} from valid set because no test set")
                    else:
                        logger.info(
                            f"Sample eval set {len(test_samples[-1])}/{len(self.samples['test'])} from test set")
        return train_samples, dev_samples, test_samples

    def sample_subset(self, data_split="train", seed=0, num=100, exclude=None):
        with temp_seed(seed):
            samples = self.samples[data_split]
            lens = len(samples)
            index = np.random.permutation(lens).tolist()[:num if exclude is None else num + 1]
            if exclude is not None and exclude in index:
                index.remove(exclude)
            else:
                index = index[:num]
            return [samples[i] for i in index]

    @property
    def valid_samples(self):
        return self.samples["valid"]


class DROPDataset(Dataset):
    metric_name = "f1"
    generation = True

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset()

    def load_dataset(self):
        dataset = load_dataset("drop")
        train_examples = dataset["train"]
        valid_examples = dataset["validation"]

        train_samples = [self.build_sample(example, idx) for idx, example in enumerate(train_examples)]
        valid_samples = [self.build_sample(example, idx) for idx, example in enumerate(valid_examples)]
        self.samples = {"train": train_samples, "valid": valid_samples}

    # for generative tasks, candidates are []
    def build_sample(self, example, idx):
        answers = example['answers_spans']['spans']
        assert len(answers) > 0
        return Sample(id=idx, data={"context": example['passage'], "question": example['question'], "answers": answers},
                      candidates=None, correct_candidate=answers)

    def get_template(self, template_version=0):
        return {0: DROPTemplate}[template_version]()


class MATH10KDataset(Dataset):
    metric_name = "math"
    generation = True

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset()

    def load_dataset(self):
        # dataset = load_dataset("math10k", data_files={"train": "ft-training_set/math_10k.json"})
        # train_examples = dataset["train"]
        with open("ft-training_set/math_10k.json") as f:
            train_examples = json.load(f)

        train_samples = [self.build_sample(example, idx) for idx, example in enumerate(train_examples)]
        self.samples = {"train": train_samples}

    def build_sample(self, example, idx):
        return Sample(id=idx, data={"question": self.clean_text(example['instruction']),
                                    "output": self.clean_text(example['output']) + '\n',
                                    "answer": example['answer'].strip()}, candidates=None,
                      correct_candidate=example['answer'].strip())

    def clean_text(self, text):
        text = re.sub(r'\n+', ' ', re.sub(r'(?<![.!?;:\s])\n+', '. ', text))

        return text.strip()

    def get_template(self, template_version=0):
        return {0: MATH10KTemplate}[template_version]()


class GSM8KTestDataset(Dataset):
    metric_name = "math"
    generation = True

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset()

    def load_dataset(self):
        with open('dataset/gsm8k/test.json') as f:
            test_examples = json.load(f)

        test_samples = [self.build_sample(example, idx) for idx, example in enumerate(test_examples)]
        self.samples = {"test": test_samples}

    def build_sample(self, example, idx):
        return Sample(id=idx, data={"question": self.clean_text(example['instruction']),
                                    "output": self.clean_text(example['output']) + '\n',
                                    "answer": example['answer'].strip()}, candidates=None,
                      correct_candidate=example['answer'].strip())

    def clean_text(self, text):
        text = re.sub(r'\n+', ' ', re.sub(r'(?<![.!?;:\s])\n+', '. ', text))

        return text.strip()

    def get_template(self, template_version=0):
        return {0: MATH10KTemplate}[template_version]()


class AQuATestDataset(Dataset):
    metric_name = "math"
    generation = True

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset()

    def load_dataset(self):
        with open('dataset/AQuA/test.json') as f:
            test_examples = json.load(f)

        test_samples = [self.build_sample(example, idx) for idx, example in enumerate(test_examples)]
        self.samples = {"test": test_samples}

    def build_sample(self, example, idx):
        return Sample(id=idx, data={"question": self.clean_text(example['instruction']),
                                    "output": self.clean_text(example['output']) + '\n',
                                    "answer": example['answer'].strip()}, candidates=None,
                      correct_candidate=example['answer'].strip())

    def clean_text(self, text):
        text = re.sub(r'\n+', ' ', re.sub(r'(?<![.!?;:\s])\n+', '. ', text))

        return text.strip()

    def get_template(self, template_version=0):
        return {0: MATH10KTemplate}[template_version]()


class MAWPSTestDataset(Dataset):
    metric_name = "math"
    generation = True

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset()

    def load_dataset(self):
        with open('dataset/mawps/test.json') as f:
            test_examples = json.load(f)

        test_samples = [self.build_sample(example, idx) for idx, example in enumerate(test_examples)]
        self.samples = {"test": test_samples}

    def build_sample(self, example, idx):
        return Sample(id=idx, data={"question": self.clean_text(example['instruction']),
                                    "output": self.clean_text(example['output']) + '\n',
                                    "answer": example['answer'].strip()}, candidates=None,
                      correct_candidate=example['answer'].strip())

    def clean_text(self, text):
        text = re.sub(r'\n+', ' ', re.sub(r'(?<![.!?;:\s])\n+', '. ', text))

        return text.strip()

    def get_template(self, template_version=0):
        return {0: MATH10KTemplate}[template_version]()


class SVAMPTestDataset(Dataset):
    metric_name = "math"
    generation = True

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset()

    def load_dataset(self):
        with open('dataset/SVAMP/test.json') as f:
            test_examples = json.load(f)

        test_samples = [self.build_sample(example, idx) for idx, example in enumerate(test_examples)]
        self.samples = {"test": test_samples}

    def build_sample(self, example, idx):
        return Sample(id=idx, data={"question": self.clean_text(example['instruction']),
                                    "output": self.clean_text(example['output']) + '\n',
                                    "answer": example['answer'].strip()}, candidates=None,
                      correct_candidate=example['answer'].strip())

    def clean_text(self, text):
        text = re.sub(r'\n+', ' ', re.sub(r'(?<![.!?;:\s])\n+', '. ', text))

        return text.strip()

    def get_template(self, template_version=0):
        return {0: MATH10KTemplate}[template_version]()


class COMMONSENSE170KDataset(Dataset):
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset()

    def load_dataset(self):
        with open("ft-training_set/commonsense_170k.json") as f:
            train_examples = json.load(f)

        train_samples = [self.build_sample(example, idx) for idx, example in enumerate(train_examples)]
        self.samples = {"train": train_samples}

    def build_sample(self, example, idx):
        candidates = self.extract_answer_format(example['instruction'].lower())
        correct_candidate = example['answer'].lower()
        assert correct_candidate in candidates, f'{correct_candidate=} not in {candidates=}, with {example=}'
        return Sample(id=idx, data={"question": self.clean_text(example['instruction']), "answer": example['answer']},
                      candidates=candidates, correct_candidate=correct_candidate)

    def clean_text(self, text):
        text = re.sub(r'\n+', ' ', re.sub(r'(?<![.!?;:\s])\n+', '. ', text))

        return text.strip()

    def extract_answer_format(self, text):
        # Search for the 'Answer format:' substring and find its starting index
        start_index = text.find("answer format:")

        if start_index != -1:
            # Find the colon that follows "Answer format" to locate the start of the actual format
            colon_index = text.find(":", start_index)

            if colon_index != -1:
                # Extract the substring from after the colon to the end of the text or newline
                end_index = text.find("\n", colon_index)
                if end_index == -1:  # If there's no newline, take the rest of the text
                    end_index = len(text)

                # Get the format string, strip whitespace, and split by '/'
                format_part = text[colon_index + 1:end_index].strip()
                answer_format = format_part.split('/')
                return [option.strip() for option in answer_format]  # Clean up any extra whitespace around options

        # Return an empty list if no answer format substring is found
        return []

    def get_template(self, template_version=0):
        return {0: COMMONSENSE170KTemplate}[template_version]()


class BoolQTestDataset(Dataset):
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset()

    def load_dataset(self):
        with open("dataset/boolq/test.json") as f:
            test_examples = json.load(f)

        test_samples = [self.build_sample(example, idx) for idx, example in enumerate(test_examples)]
        self.samples = {"test": test_samples}

    def build_sample(self, example, idx):
        candidates = self.extract_answer_format(example['instruction'].lower())
        correct_candidate = example['answer'].lower()
        assert correct_candidate in candidates, f'{correct_candidate=} not in {candidates=}, with {example=}'
        return Sample(id=idx, data={"question": self.clean_text(example['instruction']), "answer": example['answer']},
                      candidates=candidates, correct_candidate=correct_candidate)

    def clean_text(self, text):
        text = re.sub(r'\n+', ' ', re.sub(r'(?<![.!?;:\s])\n+', '. ', text))

        return text.strip()

    def extract_answer_format(self, text):
        # Search for the 'Answer format:' substring and find its starting index
        start_index = text.find("answer format:")

        if start_index != -1:
            # Find the colon that follows "Answer format" to locate the start of the actual format
            colon_index = text.find(":", start_index)

            if colon_index != -1:
                # Extract the substring from after the colon to the end of the text or newline
                end_index = text.find("\n", colon_index)
                if end_index == -1:  # If there's no newline, take the rest of the text
                    end_index = len(text)

                # Get the format string, strip whitespace, and split by '/'
                format_part = text[colon_index + 1:end_index].strip()
                answer_format = format_part.split('/')
                return [option.strip() for option in answer_format]  # Clean up any extra whitespace around options

        # Return an empty list if no answer format substring is found
        return []

    def get_template(self, template_version=0):
        return {0: COMMONSENSE170KTemplate}[template_version]()


class PIQATestDataset(Dataset):
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset()

    def load_dataset(self):
        with open("dataset/piqa/test.json") as f:
            test_examples = json.load(f)

        test_samples = [self.build_sample(example, idx) for idx, example in enumerate(test_examples)]
        self.samples = {"test": test_samples}

    def build_sample(self, example, idx):
        candidates = self.extract_answer_format(example['instruction'].lower())
        correct_candidate = example['answer'].lower()
        assert correct_candidate in candidates, f'{correct_candidate=} not in {candidates=}, with {example=}'
        return Sample(id=idx, data={"question": self.clean_text(example['instruction']), "answer": example['answer']},
                      candidates=candidates, correct_candidate=correct_candidate)

    def clean_text(self, text):
        text = re.sub(r'\n+', ' ', re.sub(r'(?<![.!?;:\s])\n+', '. ', text))

        return text.strip()

    def extract_answer_format(self, text):
        # Search for the 'Answer format:' substring and find its starting index
        start_index = text.find("answer format:")

        if start_index != -1:
            # Find the colon that follows "Answer format" to locate the start of the actual format
            colon_index = text.find(":", start_index)

            if colon_index != -1:
                # Extract the substring from after the colon to the end of the text or newline
                end_index = text.find("\n", colon_index)
                if end_index == -1:  # If there's no newline, take the rest of the text
                    end_index = len(text)

                # Get the format string, strip whitespace, and split by '/'
                format_part = text[colon_index + 1:end_index].strip()
                answer_format = format_part.split('/')
                return [option.strip() for option in answer_format]  # Clean up any extra whitespace around options

        # Return an empty list if no answer format substring is found
        return []

    def get_template(self, template_version=0):
        return {0: COMMONSENSE170KTemplate}[template_version]()


class SIQATestDataset(Dataset):
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset()

    def load_dataset(self):
        with open("dataset/social_i_qa/test.json") as f:
            test_examples = json.load(f)

        test_samples = [self.build_sample(example, idx) for idx, example in enumerate(test_examples)]
        self.samples = {"test": test_samples}

    def build_sample(self, example, idx):
        candidates = self.extract_answer_format(example['instruction'].lower())
        correct_candidate = example['answer'].lower()
        assert correct_candidate in candidates, f'{correct_candidate=} not in {candidates=}, with {example=}'
        return Sample(id=idx, data={"question": self.clean_text(example['instruction']), "answer": example['answer']},
                      candidates=candidates, correct_candidate=correct_candidate)

    def clean_text(self, text):
        text = re.sub(r'\n+', ' ', re.sub(r'(?<![.!?;:\s])\n+', '. ', text))

        return text.strip()

    def extract_answer_format(self, text):
        # Search for the 'Answer format:' substring and find its starting index
        start_index = text.find("answer format:")

        if start_index != -1:
            # Find the colon that follows "Answer format" to locate the start of the actual format
            colon_index = text.find(":", start_index)

            if colon_index != -1:
                # Extract the substring from after the colon to the end of the text or newline
                end_index = text.find("\n", colon_index)
                if end_index == -1:  # If there's no newline, take the rest of the text
                    end_index = len(text)

                # Get the format string, strip whitespace, and split by '/'
                format_part = text[colon_index + 1:end_index].strip()
                answer_format = format_part.split('/')
                return [option.strip() for option in answer_format]  # Clean up any extra whitespace around options

        # Return an empty list if no answer format substring is found
        return []

    def get_template(self, template_version=0):
        return {0: COMMONSENSE170KTemplate}[template_version]()


class HellaSwagTestDataset(Dataset):
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset()

    def load_dataset(self):
        with open("dataset/hellaswag/test.json") as f:
            test_examples = json.load(f)

        test_samples = [self.build_sample(example, idx) for idx, example in enumerate(test_examples)]
        self.samples = {"test": test_samples}

    def build_sample(self, example, idx):
        candidates = self.extract_answer_format(example['instruction'].lower())
        correct_candidate = example['answer'].lower()
        assert correct_candidate in candidates, f'{correct_candidate=} not in {candidates=}, with {example=}'
        return Sample(id=idx, data={"question": self.clean_text(example['instruction']), "answer": example['answer']},
                      candidates=candidates, correct_candidate=correct_candidate)

    def clean_text(self, text):
        text = re.sub(r'\n+', ' ', re.sub(r'(?<![.!?;:\s])\n+', '. ', text))

        return text.strip()

    def extract_answer_format(self, text):
        # Search for the 'Answer format:' substring and find its starting index
        start_index = text.find("answer format:")

        if start_index != -1:
            # Find the colon that follows "Answer format" to locate the start of the actual format
            colon_index = text.find(":", start_index)

            if colon_index != -1:
                # Extract the substring from after the colon to the end of the text or newline
                end_index = text.find("\n", colon_index)
                if end_index == -1:  # If there's no newline, take the rest of the text
                    end_index = len(text)

                # Get the format string, strip whitespace, and split by '/'
                format_part = text[colon_index + 1:end_index].strip()
                answer_format = format_part.split('/')
                return [option.strip() for option in answer_format]  # Clean up any extra whitespace around options

        # Return an empty list if no answer format substring is found
        return []

    def get_template(self, template_version=0):
        return {0: COMMONSENSE170KTemplate}[template_version]()


class WinoGrandeTestDataset(Dataset):
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset()

    def load_dataset(self):
        with open("dataset/winogrande/test.json") as f:
            test_examples = json.load(f)

        test_samples = [self.build_sample(example, idx) for idx, example in enumerate(test_examples)]
        self.samples = {"test": test_samples}

    def build_sample(self, example, idx):
        candidates = self.extract_answer_format(example['instruction'].lower())
        correct_candidate = example['answer'].lower()
        assert correct_candidate in candidates, f'{correct_candidate=} not in {candidates=}, with {example=}'
        return Sample(id=idx, data={"question": self.clean_text(example['instruction']), "answer": example['answer']},
                      candidates=candidates, correct_candidate=correct_candidate)

    def clean_text(self, text):
        text = re.sub(r'\n+', ' ', re.sub(r'(?<![.!?;:\s])\n+', '. ', text))

        return text.strip()

    def extract_answer_format(self, text):
        # Search for the 'Answer format:' substring and find its starting index
        start_index = text.find("answer format:")

        if start_index != -1:
            # Find the colon that follows "Answer format" to locate the start of the actual format
            colon_index = text.find(":", start_index)

            if colon_index != -1:
                # Extract the substring from after the colon to the end of the text or newline
                end_index = text.find("\n", colon_index)
                if end_index == -1:  # If there's no newline, take the rest of the text
                    end_index = len(text)

                # Get the format string, strip whitespace, and split by '/'
                format_part = text[colon_index + 1:end_index].strip()
                answer_format = format_part.split('/')
                return [option.strip() for option in answer_format]  # Clean up any extra whitespace around options

        # Return an empty list if no answer format substring is found
        return []

    def get_template(self, template_version=0):
        return {0: COMMONSENSE170KTemplate}[template_version]()


class ARCETestDataset(Dataset):
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset()

    def load_dataset(self):
        with open("dataset/ARC-Easy/test.json") as f:
            test_examples = json.load(f)

        test_samples = [self.build_sample(example, idx) for idx, example in enumerate(test_examples)]
        self.samples = {"test": test_samples}

    def build_sample(self, example, idx):
        candidates = self.extract_answer_format(example['instruction'].lower())
        correct_candidate = example['answer'].lower()
        assert correct_candidate in candidates, f'{correct_candidate=} not in {candidates=}, with {example=}'
        return Sample(id=idx, data={"question": self.clean_text(example['instruction']), "answer": example['answer']},
                      candidates=candidates, correct_candidate=correct_candidate)

    def clean_text(self, text):
        text = re.sub(r'\n+', ' ', re.sub(r'(?<![.!?;:\s])\n+', '. ', text))

        return text.strip()

    def extract_answer_format(self, text):
        # Search for the 'Answer format:' substring and find its starting index
        start_index = text.find("answer format:")

        if start_index != -1:
            # Find the colon that follows "Answer format" to locate the start of the actual format
            colon_index = text.find(":", start_index)

            if colon_index != -1:
                # Extract the substring from after the colon to the end of the text or newline
                end_index = text.find("\n", colon_index)
                if end_index == -1:  # If there's no newline, take the rest of the text
                    end_index = len(text)

                # Get the format string, strip whitespace, and split by '/'
                format_part = text[colon_index + 1:end_index].strip()
                answer_format = format_part.split('/')
                return [option.strip() for option in answer_format]  # Clean up any extra whitespace around options

        # Return an empty list if no answer format substring is found
        return []

    def get_template(self, template_version=0):
        return {0: COMMONSENSE170KTemplate}[template_version]()


class ARCCTestDataset(Dataset):
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset()

    def load_dataset(self):
        with open("dataset/ARC-Challenge/test.json") as f:
            test_examples = json.load(f)

        test_samples = [self.build_sample(example, idx) for idx, example in enumerate(test_examples)]
        self.samples = {"test": test_samples}

    def build_sample(self, example, idx):
        candidates = self.extract_answer_format(example['instruction'].lower())
        correct_candidate = example['answer'].lower()
        assert correct_candidate in candidates, f'{correct_candidate=} not in {candidates=}, with {example=}'
        return Sample(id=idx, data={"question": self.clean_text(example['instruction']), "answer": example['answer']},
                      candidates=candidates, correct_candidate=correct_candidate)

    def clean_text(self, text):
        text = re.sub(r'\n+', ' ', re.sub(r'(?<![.!?;:\s])\n+', '. ', text))

        return text.strip()

    def extract_answer_format(self, text):
        # Search for the 'Answer format:' substring and find its starting index
        start_index = text.find("answer format:")

        if start_index != -1:
            # Find the colon that follows "Answer format" to locate the start of the actual format
            colon_index = text.find(":", start_index)

            if colon_index != -1:
                # Extract the substring from after the colon to the end of the text or newline
                end_index = text.find("\n", colon_index)
                if end_index == -1:  # If there's no newline, take the rest of the text
                    end_index = len(text)

                # Get the format string, strip whitespace, and split by '/'
                format_part = text[colon_index + 1:end_index].strip()
                answer_format = format_part.split('/')
                return [option.strip() for option in answer_format]  # Clean up any extra whitespace around options

        # Return an empty list if no answer format substring is found
        return []

    def get_template(self, template_version=0):
        return {0: COMMONSENSE170KTemplate}[template_version]()


class OBQATestDataset(Dataset):
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset()

    def load_dataset(self):
        with open("dataset/openbookqa/test.json") as f:
            test_examples = json.load(f)

        test_samples = [self.build_sample(example, idx) for idx, example in enumerate(test_examples)]
        self.samples = {"test": test_samples}

    def build_sample(self, example, idx):
        candidates = self.extract_answer_format(example['instruction'].lower())
        correct_candidate = example['answer'].lower()
        assert correct_candidate in candidates, f'{correct_candidate=} not in {candidates=}, with {example=}'
        return Sample(id=idx, data={"question": self.clean_text(example['instruction']), "answer": example['answer']},
                      candidates=candidates, correct_candidate=correct_candidate)

    def clean_text(self, text):
        text = re.sub(r'\n+', ' ', re.sub(r'(?<![.!?;:\s])\n+', '. ', text))

        return text.strip()

    def extract_answer_format(self, text):
        # Search for the 'Answer format:' substring and find its starting index
        start_index = text.find("answer format:")

        if start_index != -1:
            # Find the colon that follows "Answer format" to locate the start of the actual format
            colon_index = text.find(":", start_index)

            if colon_index != -1:
                # Extract the substring from after the colon to the end of the text or newline
                end_index = text.find("\n", colon_index)
                if end_index == -1:  # If there's no newline, take the rest of the text
                    end_index = len(text)

                # Get the format string, strip whitespace, and split by '/'
                format_part = text[colon_index + 1:end_index].strip()
                answer_format = format_part.split('/')
                return [option.strip() for option in answer_format]  # Clean up any extra whitespace around options

        # Return an empty list if no answer format substring is found
        return []

    def get_template(self, template_version=0):
        return {0: COMMONSENSE170KTemplate}[template_version]()
