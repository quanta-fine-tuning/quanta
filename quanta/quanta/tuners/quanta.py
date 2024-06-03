import importlib
import itertools
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from math import prod
from typing import List, Optional, Union

import opt_einsum as oe
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import PeftConfig, PeftType


def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


if is_bnb_available():
    import bitsandbytes as bnb


@dataclass
class QuanTAConfig(PeftConfig):
    d: int = field(default=1, metadata={"help": "quanta number of dimensions"})
    quanta_dropout: float = field(default=0.0, metadata={"help": "quanta dropout"})
    merge_weights: bool = field(default=False,
                                metadata={"help": "Merge weights of the original model and the Lora model"})
    fan_in_fan_out: bool = field(default=False,
                                 metadata={
                                     "help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"}, )
    per_dim_features: Optional[List[int]] = field(default=None, metadata={
        "help": "List of the number of features per dimension. If not provided, the features are equally divided."}, )
    per_dim_features2: Optional[List[int]] = field(default=None, metadata={
        "help": "List of the number of features per dimension for the output. If not provided, the features are set to per_dim_features."}, )
    sum_mode: bool = field(default=False, metadata={"help": "Set this to True if the quanta is in sum mode"})
    initialize_mode: str = field(default="sum_opposite_freeze_one",
                                 metadata={
                                     "help": "Initialization mode for the quanta weights. Can be 'sum_opposite_freeze_one'"})
    bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    target_modules: Optional[Union[List[str], str]] = field(default=None,
                                                            metadata={
                                                                "help": "List of module names or regex expression of the module names to replace with Lora."
                                                                        "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "}, )  # not sure if this is needed legacy support for now
    enable_lora: Optional[List[bool]] = field(default=None, metadata={
        "help": "Used with `lora.MergedLinear`."})  # not sure if this is needed legacy support for now
    tensor_rank: int = 5  # not sure if this is needed legacy support for now

    def __post_init__(self):
        self.peft_type = PeftType.QUANTA


class QuanTAModel(torch.nn.Module):

    def __init__(self, config, model):
        super().__init__()
        self.peft_config = config
        self.model = model
        self._find_and_replace()
        mark_lora_layernorm_cls_trainable(self.model, self.peft_config.task_type, self.peft_config.tensor_rank,
                                          self.peft_config.bias)
        self.forward = self.model.forward

    def _find_and_replace(self):
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit and not is_bnb_available():
            raise ImportError("To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
                              "You can install it with `pip install bitsandbytes`.")
        is_target_modules_in_base_model = False
        is_hf_device_map_available = hasattr(self.model, "hf_device_map")
        kwargs = {"d": self.peft_config.d, "per_dim_features": self.peft_config.per_dim_features,
                  "per_dim_features2": self.peft_config.per_dim_features2,
                  "quanta_dropout": self.peft_config.quanta_dropout,
                  "fan_in_fan_out": self.peft_config.fan_in_fan_out,
                  "merge_weights": (self.peft_config.merge_weights or self.peft_config.inference_mode),
                  "sum_mode": self.peft_config.sum_mode, "initialize_mode": self.peft_config.initialize_mode, }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in self.peft_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                bias = target.bias is not None
                if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                    raise NotImplementedError
                elif isinstance(target, torch.nn.Linear) and self.peft_config.enable_lora is None:
                    new_module = Linear(target.in_features, target.out_features, bias=bias, **kwargs)
                elif self.peft_config.enable_lora is not None:
                    raise NotImplementedError
                self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(f"Target modules {self.peft_config.target_modules} not found in the base model. "
                             f"Please check the target modules and try again.")

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name or "quanta_" in name:
                module.to(old_module.weight.device)
            if 'bias' in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @property
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False):
        config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self.peft_config).items()}
        if inference:
            config["inference_mode"] = True
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, LoraLayer) or isinstance(module, QuanTALayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)


# had to adapt it for `lora_only` to work
def mark_lora_layernorm_cls_trainable(model: nn.Module, task_type, tensor_rank, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n and "quanta_" not in n:
            p.requires_grad = False
    if bias == "none":
        pass
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if (isinstance(m, LoraLayer) or isinstance(m, QuanTALayer)) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError  # mark layer-norm trainable
    for n, p in model.named_parameters():
        if "Norm" in n:
            p.requires_grad = True
    # mark cls trainable and tensorized
    if task_type == 'SEQ_CLS':
        raise NotImplementedError


class LoraLayer:
    def __init__(self, r: int, lora_alpha: int, lora_dropout: float, merge_weights: bool, ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False


class MergeBuffer(nn.Module):

    def __init__(self, default=False):
        super(MergeBuffer, self).__init__()
        self.register_buffer('merged', torch.tensor(default))  # to keep track if the trainable weights are merged

    def __bool__(self):
        return self.merged.item()

    def set(self, value):
        self.merged.fill_(value)


class QuanTALayer:
    def __init__(self, d: int, quanta_dropout: float, merge_weights: bool, sum_mode: bool = False, ):
        self.d = d
        self.sum_mode = sum_mode
        if quanta_dropout > 0.:
            self.quanta_dropout = nn.Dropout(p=quanta_dropout)
        else:
            self.quanta_dropout = lambda x: x
        self.merged = MergeBuffer(default=False)  # so that this will be tracked when saving the model and loading it
        self.frozen_merged = MergeBuffer(default=False)  # the frozen weights are separately tracked
        self.merge_weights = merge_weights


class BufferDict(nn.Module):
    def __init__(self, init_dict=None):
        super(BufferDict, self).__init__()
        self.buffer_names = []
        if init_dict is not None:
            for name, tensor in init_dict.items():
                self.add_buffer(name, tensor)

    def add_buffer(self, name, tensor):
        self.register_buffer(name, tensor)
        if name not in self.buffer_names:
            self.buffer_names.append(name)

    def __getitem__(self, name):
        return getattr(self, name)

    def __setitem__(self, name, tensor):
        self.add_buffer(name, tensor)

    def items(self):
        for name in self.buffer_names:
            yield name, getattr(self, name)

    def keys(self):
        return self.buffer_names

    def values(self):
        return [getattr(self, name) for name in self.buffer_names]


class Linear(nn.Linear, QuanTALayer):
    # QuanTA implemented in a dense layer
    def __init__(self, in_features: int, out_features: int, d: int = 1, quanta_dropout: float = 0.,
                 fan_in_fan_out: bool = False,
                 # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
                 per_dim_features: list = None,
                 # usually used as the input features, but will check if need to be swapped with per_dim_features2, or if it should be the same as per_dim_features2
                 per_dim_features2: list = None,
                 # usually used as the output features, but will check if need to be swapped with per_dim_features, or if it should be the same as per_dim_features
                 merge_weights: bool = False, sum_mode: bool = False, initialize_mode: str = 'scale_by_softplus_zero',
                 **kwargs):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)

        self.fan_in_fan_out = fan_in_fan_out
        self.initialize_mode = initialize_mode
        # Actual trainable parameters
        if per_dim_features is not None:
            d = len(per_dim_features)
        QuanTALayer.__init__(self, d=d, quanta_dropout=quanta_dropout, sum_mode=sum_mode, merge_weights=merge_weights)

        if d > 1:
            self.max_features = max(in_features, out_features)
            if per_dim_features is not None:
                self.per_dim_features = per_dim_features
            else:
                self.per_dim_features = [math.ceil(self.max_features ** (1 / d))] * d
            if per_dim_features2 is not None:
                raise NotImplementedError('per_dim_features2 is not implemented yet')
            else:
                self.per_dim_features2 = self.per_dim_features

            self.total_features = prod(self.per_dim_features)
            self.total_features2 = prod(self.per_dim_features2)

            if self.total_features != in_features:
                warnings.warn(
                    f'per_dim_features={self.per_dim_features} does not match in_features={in_features}, this should work but may result in downgraded performance or additional cost. Please make sure this is intended.')
            if self.total_features2 != out_features:
                warnings.warn(
                    f'per_dim_features2={self.per_dim_features2} does not match out_features={out_features}, this should work but may result in downgraded performance or additional cost. Please make sure this is intended.')

            quanta_weights = {}
            for (dim1, dim2) in itertools.combinations(range(-1, -d - 1, -1), 2):
                quanta_weights[f'{dim1} {dim2}'] = nn.Parameter(
                    self.weight.new_zeros(self.per_dim_features2[dim2], self.per_dim_features2[dim1],
                                          self.per_dim_features[dim2], self.per_dim_features[
                                              dim1]))  # reverse the order because dim1 is closer to the end
            self.quanta_weights = nn.ParameterDict(quanta_weights)
            if initialize_mode == 'sum_opposite_freeze_one':
                quanta_weights2 = {}
                for (dim1, dim2) in itertools.combinations(range(-1, -d - 1, -1), 2):
                    quanta_weights2[f'{dim1} {dim2}'] = self.weight.new_zeros(self.per_dim_features2[dim2],
                                                                              self.per_dim_features2[dim1],
                                                                              self.per_dim_features[dim2],
                                                                              self.per_dim_features[dim1])
                self.quanta_weights2 = BufferDict(quanta_weights2)

            else:
                assert False, f'initialize_mode={initialize_mode} not implemented'

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

        self.gen_einsum_expr_train()
        self.gen_einsum_expr_eval()

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'quanta_weights'):
            for k, v in self.quanta_weights.items():
                nn.init.kaiming_uniform_(v.view(v.shape[0] * v.shape[1], v.shape[2] * v.shape[3]), a=math.sqrt(5),
                                         nonlinearity='linear')  # initialize as if it is a matrix
            if self.initialize_mode == 'last_layer_zero':
                nn.init.zeros_(self.quanta_weights[f'{-self.d + 1} {-self.d}'])
            if self.initialize_mode == 'add_local_layer_zero' or self.initialize_mode == 'add_local_layer_zero_sum':
                for k, v in self.local_weights.items():
                    nn.init.zeros_(v)
            if self.initialize_mode == 'sum_opposite':
                for k, v in self.quanta_weights2.items():
                    v.data[:] = self.quanta_weights[k].data
            if self.initialize_mode == 'sum_opposite_freeze_one':
                for k, v in self.quanta_weights2.items():
                    v[:] = self.quanta_weights[k].data

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged and self.initialize_mode in ['sum_opposite_freeze_one']:
                # Make sure that the weights are not merged
                if self.d > 0:
                    full_quanta_weights = F.pad(self.einsum_expr_eval(
                        *[self.quanta_weights[f'{dim1} {dim2}'].to(self.weight.dtype) for (dim1, dim2) in
                          itertools.combinations(range(-1, -self.d - 1, -1), 2)]).reshape(self.total_features2,
                                                                                          self.total_features),
                                                (0, self.in_features - self.total_features, 0,
                                                 self.out_features - self.total_features2),
                                                'constant', 0.)
                    self.weight.data -= T(full_quanta_weights)
                self.merged.set(False)
        else:
            if self.merge_weights and not self.merged and self.initialize_mode in ['sum_opposite_freeze_one']:
                # Merge the weights and mark it
                if self.d > 0:
                    full_quanta_weights = F.pad(self.einsum_expr_eval(
                        *[self.quanta_weights[f'{dim1} {dim2}'].to(self.weight.dtype) for (dim1, dim2) in
                          itertools.combinations(range(-1, -self.d - 1, -1), 2)]).reshape(self.total_features2,
                                                                                          self.total_features),
                                                (0, self.in_features - self.total_features, 0,
                                                 self.out_features - self.total_features2),
                                                'constant', 0.)
                    self.weight.data += T(full_quanta_weights)
                    if not self.frozen_merged:
                        self.merge_frozen_weights()
                self.merged.set(True)

    def merge_frozen_weights(self):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.frozen_merged:
            warnings.warn('The frozen weights are already merged, ignoring the request to merge the frozen weights')
        else:
            full_quanta_weights = -F.pad(self.einsum_expr_eval(
                *[self.quanta_weights2[f'{dim1} {dim2}'].to(self.weight.dtype) for (dim1, dim2) in
                  itertools.combinations(range(-1, -self.d - 1, -1), 2)]).reshape(self.total_features2,
                                                                                  self.total_features),
                                         (0, self.in_features - self.total_features, 0,
                                          self.out_features - self.total_features2), 'constant',
                                         0.)
            self.weight.data += T(full_quanta_weights)
            self.frozen_merged.set(True)

    def unmerge_frozen_weights(self):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if not self.frozen_merged:
            warnings.warn('The frozen weights are already unmerged, ignoring the request to unmerge the frozen weights')
        else:
            full_quanta_weights = -F.pad(self.einsum_expr_eval(
                *[self.quanta_weights2[f'{dim1} {dim2}'].to(self.weight.dtype) for (dim1, dim2) in
                  itertools.combinations(range(-1, -self.d - 1, -1), 2)]).reshape(self.total_features2,
                                                                                  self.total_features),
                                         (0, self.in_features - self.total_features, 0,
                                          self.out_features - self.total_features2), 'constant',
                                         0.)
            self.weight.data -= T(full_quanta_weights)
            self.frozen_merged.set(False)

    def gen_einsum_expr_train(self):
        """
        Generate the einsum expression for the tensorized weights during training.
        """
        d = self.d
        current_symbols_inds = list(range(d))

        eq = '...'
        for i in current_symbols_inds:
            eq += oe.get_symbol(i)

        for (dim1, dim2) in itertools.combinations(range(-1, -d - 1, -1), 2):
            symbol_ind1 = current_symbols_inds[dim1]
            symbol_ind2 = current_symbols_inds[dim2]
            symbol_ind3 = symbol_ind1 + d
            symbol_ind4 = symbol_ind2 + d
            eq += ',' + oe.get_symbol(symbol_ind4) + oe.get_symbol(symbol_ind3) + oe.get_symbol(
                symbol_ind2) + oe.get_symbol(
                symbol_ind1)  # reverse order because dim1 is toward the end than dim2 and because of matrix multiplication order convention. Note that this is different from the forward function
            current_symbols_inds[dim1] = symbol_ind3
            current_symbols_inds[dim2] = symbol_ind4

        eq += '->...'
        for i in current_symbols_inds:
            eq += oe.get_symbol(i)

        shapes = [(100,) + tuple(self.per_dim_features)]  # may need to change the 100 to some other value
        for (dim1, dim2) in itertools.combinations(range(-1, -d - 1, -1), 2):
            shapes.append((self.per_dim_features[dim2], self.per_dim_features[dim1], self.per_dim_features[dim2],
                           self.per_dim_features[dim1]))

        optimize = 'optimal' if d <= 4 else 'branch-all' if d <= 5 else 'branch-2' if d <= 7 else 'auto'
        expr = oe.contract_expression(eq, *shapes, optimize=optimize)

        self.einsum_eq_train = eq
        self.einsum_expr_train = expr

    def gen_einsum_expr_eval(self):
        """
        Generate the einsum expression for the tensorized weights during evaluation.
        """
        d = self.d
        current_symbols_inds = list(range(d))
        init_symbols_inds = [i for i in current_symbols_inds]  # copy

        eq = ''

        for (dim1, dim2) in itertools.combinations(range(-1, -d - 1, -1), 2):
            symbol_ind1 = current_symbols_inds[dim1]
            symbol_ind2 = current_symbols_inds[dim2]
            symbol_ind3 = symbol_ind1 + d
            symbol_ind4 = symbol_ind2 + d
            eq += ',' + oe.get_symbol(symbol_ind4) + oe.get_symbol(symbol_ind3) + oe.get_symbol(
                symbol_ind2) + oe.get_symbol(symbol_ind1)  # reverse order because dim1 is toward the end than dim2
            current_symbols_inds[dim1] = symbol_ind3
            current_symbols_inds[dim2] = symbol_ind4

        eq += '->'
        for i in current_symbols_inds:
            eq += oe.get_symbol(i)
        for i in init_symbols_inds:
            eq += oe.get_symbol(
                i)  # note that this is also the reverse order, so it is the usual matrix multiplication order which is (fan_out, fan_in)
        eq = eq[1:]

        shapes = []
        for (dim1, dim2) in itertools.combinations(range(-1, -d - 1, -1), 2):
            shapes.append((self.per_dim_features[dim2], self.per_dim_features[dim1], self.per_dim_features[dim2],
                           self.per_dim_features[dim1]))

        optimize = 'optimal' if d <= 4 else 'branch-all' if d <= 5 else 'branch-2' if d <= 7 else 'auto'
        expr = oe.contract_expression(eq, *shapes, optimize=optimize)

        self.einsum_eq_eval = eq
        self.einsum_expr_eval = expr

    def forward_quanta_weights(self, x, quanta_weights):
        """
        assume x is of shape (batch, *per_dim_features)
        """
        return self.einsum_expr_train(x, *[quanta_weights[f'{dim1} {dim2}'].to(x.dtype) for (dim1, dim2) in
                                           itertools.combinations(range(-1, -self.d - 1, -1), 2)])

    def forward_sum_opposite(self, x: torch.Tensor):
        assert self.initialize_mode == 'sum_opposite_freeze_one', f'this function is only for sum_opposite_freeze_one, but got {self.initialize_mode=}'
        assert not self.sum_mode, f'this function only works for sum_mode=False, but got {self.sum_mode=}'
        if not self.frozen_merged:
            self.merge_frozen_weights()  # make sure the frozen weights are merged
        previous_dtype = self.weight.dtype

        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.d > 1 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias.to(previous_dtype) if self.bias is not None else None)
            x = self.quanta_dropout(x)
            x = F.pad(x, (0, self.total_features - self.in_features), 'constant', 0.)
            x_shape = x.shape
            x = x.view(-1, *self.per_dim_features)

            # then deal with weight
            x = self.forward_quanta_weights(x, self.quanta_weights).reshape(*x_shape[:-1], self.total_features2)

            result += F.pad(x, (0, self.out_features - self.total_features2), 'constant', 0.)

            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias.to(previous_dtype) if self.bias is not None else None)

    def forward(self, x: torch.Tensor):
        if self.initialize_mode == 'sum_opposite_freeze_one':
            return self.forward_sum_opposite(x)
        else:
            raise NotImplementedError(
                f'initialize_mode={self.initialize_mode} not implemented, only sum_opposite_freeze_one is implemented at the moment')
