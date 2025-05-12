from typing import *
from collections import OrderedDict

from ..core import Tensor, _get_d


class Module:
    """
    base class for neural network modules.
    """
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._d = _get_d(device=self.device)  # backend device handler
        self.use_np = self.device == "cpu"
        self.is_training = True

    def _get_tensors(self) -> List[Tensor]:
        """
        Retrieves all tensors defined within the module, including its children modules.
        This includes both trainable parameters and registered buffers.
        """
        tensors: List[Tensor] = []

        for _, value in self.__dict__.items():
            if isinstance(value, Tensor):
                # Include all tensors, whether they require gradients or not
                tensors.append(value)
            elif isinstance(value, (Module, ModuleList, ModuleDict)):
                # Recursively get tensors from child modules
                tensors.extend(value._get_tensors())

        return tensors

    def train(self) -> None:
        """
        sets the module to training mode.
        """
        self.is_training = True

    def eval(self) -> None:
        """
        sets the module to evaluation mode.
        """
        self.is_training = False

    def parameters(self) -> List[Tensor]:
        """
        Retrieves only the trainable parameters (tensors requiring gradients).
        This excludes registered buffers which shouldn't be trained.
        """
        return [t for t in self._get_tensors() if t.requires_grad]

    def zero_grad(self) -> None:
        """
        resets the gradients of all parameters.
        """
        for param in self.parameters():
            param._reset_grad()

    def forward(self, *args: Any, **kwargs: Any):
        """
        to be overridden by child classes for forward computation.
        """
        raise NotImplementedError("the forward method must be implemented in child classes.")

    def state_dict(self, prefix: str = '') -> OrderedDict:
        """
        returns a dictionary containing the state of the module, including parameters and buffers.
        """
        state_dict = OrderedDict()
        for name, value in self.__dict__.items():
            pref = f"{prefix}.{name}" if prefix else name
            if isinstance(value, Tensor):
                state_dict[pref] = value.data
            elif isinstance(value, (Module, ModuleList, ModuleDict)):
                state_dict.update(value.state_dict(prefix=pref))

        return state_dict

    def load_state_dict(self, state_dict: OrderedDict, prefix: str = '') -> None:
        """
        loads the module state from a dictionary, updating parameters and buffers.
        """
        for name, value in self.__dict__.items():
            pref = f"{prefix}.{name}" if prefix else name
            if isinstance(value, Tensor):
                new_value = state_dict.get(pref)
                if new_value is None:
                    raise ValueError(f"the key '{pref}' is missing from the state_dict.")
                value.data[:] = new_value
            elif isinstance(value, (Module, ModuleList, ModuleDict)):
                value.load_state_dict(state_dict, prefix=pref)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        calls the forward method with the provided arguments.
        """
        return self.forward(*args, **kwargs)

    def register_buffer(self, name: str, tensor: Tensor) -> None:
        """
        Registers a buffer that should be saved and restored during save/load but not 
        trained. This is useful for tensors that are a fixed part of the
        module's operation, like attention masks or positional encodings.

        A buffer is different from a parameter because:
        1. It won't be included in parameters() and won't receive gradients
        2. It will be saved and loaded with the model's state
        3. It will be moved to the correct device with the module

        Args:
            name: Name to give the buffer
            tensor: The tensor to register as a buffer
        """
        setattr(self, name, tensor)

class ModuleList(Module):
    """
    a sequential container of modules, where each module is applied in order.
    """
    def __init__(self, modules: List[Module] = None, device: str = "cpu") -> None:
        super().__init__(device=device)
        self._modules = modules or []

    def __getitem__(self, idx: int) -> Module:
        return self._modules[idx]

    def __setitem__(self, idx: int, module: Module) -> None:
        self._modules[idx] = module

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules)

    def append(self, module: Module) -> None:
        self._modules.append(module)

    def extend(self, modules: List[Module]) -> None:
        self._modules.extend(modules)

    def insert(self, index: int, module: Module) -> None:
        self._modules.insert(index, module)

    def parameters(self) -> List[Tensor]:
        return [param for module in self._modules for param in module.parameters()]

    def state_dict(self, prefix: str = '') -> OrderedDict:
        state_dict = OrderedDict()
        for idx, module in enumerate(self._modules):
            pref = f"{prefix}.{idx}" if prefix else str(idx)
            state_dict.update(module.state_dict(prefix=pref))
        return state_dict

    def load_state_dict(self, state_dict: OrderedDict, prefix: str = '') -> None:
        for idx, module in enumerate(self._modules):
            pref = f"{prefix}.{idx}" if prefix else str(idx)
            module.load_state_dict(state_dict, prefix=pref)

    def forward(self, x: Any) -> Any:
        for module in self._modules:
            x = module(x)
        return x


class ModuleDict(Module):
    """
    a dictionary-like container of modules, where modules are stored with keys.
    """
    def __init__(self, modules: Dict[str, Module] = None, device: str = "cpu") -> None:
        super().__init__(device=device)
        self._modules = modules or {}

    def __getitem__(self, key: str) -> Module:
        return self._modules[key]

    def __setitem__(self, key: str, module: Module) -> None:
        self._modules[key] = module

    def __delitem__(self, key: str) -> None:
        del self._modules[key]

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[str]:
        return iter(self._modules)

    def parameters(self) -> List[Tensor]:
        return [param for module in self._modules.values() for param in module.parameters()]

    def state_dict(self, prefix: str = '') -> OrderedDict:
        state_dict = OrderedDict()
        for key, module in self._modules.items():
            pref = f"{prefix}.{key}" if prefix else key
            state_dict.update(module.state_dict(prefix=pref))
        return state_dict

    def load_state_dict(self, state_dict: OrderedDict, prefix: str = '') -> None:
        for key, module in self._modules.items():
            pref = f"{prefix}.{key}" if prefix else key
            module.load_state_dict(state_dict, prefix=pref)

    def forward(self, x: Any) -> Any:
        for module in self._modules.values():
            x = module(x)
        return x
