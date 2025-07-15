import inspect

import torch.nn as nn
import torch.optim as optim

from easydict import EasyDict
from util.config_yaml import merge_new_config
from util.misc import is_seq_of
from typing import Union, List


class Registry:
    """A registry to map strings to classes. In essence, all the registry does is store a dict that maps string to some
    class. In addition, the registry can be a parent/child of other registries and any class in one of these can in theory
    be accessed from the registry. The registry can then be used for e.g. building class objects based on some config file
    Example:
        >>> MODELS = Registry('models')
        >>> @MODELS.register_module()
        >>> class ResNet:
        >>>     pass
        >>> resnet = MODELS.build(dict(NAME='ResNet'))
    Please refer to https://mmcv.readthedocs.io/en/latest/registry.html for (5.8.2023: link 404s)
    advanced useage.
    Args:
        name (str): Registry name.
        build_func(func, optional): Build function to construct instance from
            Registry, func:`build_from_cfg` is used if neither ``parent`` or
            ``build_func`` is specified. If ``parent`` is specified and
            ``build_func`` is not given,  ``build_func`` will be inherited
            from ``parent``. Default: None.
        parent (Registry, optional): Parent registry. The class registered in
            children registry could be built from parent. Default: None.
        scope (str, optional): The scope of registry. It is the key to search
            for children registry. If not specified, scope will be the name of
            the package where class is defined, e.g. mmdet, mmcls, mmseg.
            Default: None.
    """

    def __init__(
        self,
        name: str,
        build_func=None,
        parent: Union["Registry", None] = None,
        scope: Union[str, None] = None,
    ):
        self._name = name
        self._module_dict = dict()  # Dict of modules in this registry
        self._children = (
            dict()
        )  # Dict of any registires that this registry is a parent of
        self._scope = self.infer_scope() if scope is None else scope

        # self.build_func will be set with the following priority:
        # 1. build_func
        # 2. parent.build_func
        # 3. build_from_cfg
        if build_func is None:
            if parent is not None:
                self.build_func = parent.build_func
            else:
                self.build_func = build_from_cfg
        else:
            self.build_func = build_func
        if parent is not None:
            assert isinstance(parent, Registry)
            parent._add_children(self)
            self.parent = parent
        else:
            self.parent = None

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        format_str = (
            self.__class__.__name__ + f"(name={self._name}, items={self._module_dict})"
        )
        return format_str

    @staticmethod
    def infer_scope():
        """Infer the scope of registry.
        The name of the package where registry is defined will be returned.
        Example:
            # in mmdet/models/backbone/resnet.py
            >>> MODELS = Registry('models')
            >>> @MODELS.register_module()
            >>> class ResNet:
            >>>     pass
            The scope of ``ResNet`` will be ``mmdet``.
        Returns:
            scope (str): The inferred scope name.
        """
        # inspect.stack() trace where this function is called, the index-2
        # indicates the frame where `infer_scope()` is called
        filename = inspect.getmodule(inspect.stack()[2][0]).__name__
        split_filename = filename.split(".")
        return split_filename[0]

    @staticmethod
    def split_scope_key(key):
        """Split scope and key.
        The first scope will be split from key.
        Examples:
            >>> Registry.split_scope_key('mmdet.ResNet')
            'mmdet', 'ResNet'
            >>> Registry.split_scope_key('ResNet')
            None, 'ResNet'
        Return:
            scope (str, None): The first scope.
            key (str): The remaining key.
        """
        split_index = key.find(".")
        if split_index != -1:
            return key[:split_index], key[split_index + 1 :]
        else:
            return None, key

    @property
    def name(self):
        return self._name

    @property
    def scope(self):
        return self._scope

    @property
    def module_dict(self):
        return self._module_dict

    @property
    def children(self):
        return self._children

    def get(self, key: str):
        """Get the registry record.
        Args:
            key (str): The class name in string format.
        Returns:
            class: The corresponding class.
        """
        scope, real_key = self.split_scope_key(key)
        if scope is None or scope == self._scope:
            # get from self
            if real_key in self._module_dict:
                return self._module_dict[real_key]
        else:
            # get from self._children
            if scope in self._children:
                return self._children[scope].get(real_key)
            else:
                # goto root
                parent = self.parent
                while parent.parent is not None:
                    # Goes to the parent highest in the hierarchy and then searches through all its children
                    parent = parent.parent
                return parent.get(key)

    def build(self, *args, **kwargs):
        return self.build_func(*args, **kwargs, registry=self)

    def _add_children(self, registry: "Registry"):
        """Add children for a registry.
        The ``registry`` will be added as children based on its scope.
        The parent registry could build objects from children registry.
        Example:
            >>> models = Registry('models')
            >>> mmdet_models = Registry('models', parent=models)
            >>> @mmdet_models.register_module()
            >>> class ResNet:
            >>>     pass
            >>> resnet = models.build(dict(NAME='mmdet.ResNet'))
        """

        assert isinstance(registry, Registry)
        assert registry.scope is not None
        assert registry.scope not in self.children, (
            f"scope {registry.scope} exists in {self.name} registry"
        )
        self.children[registry.scope] = registry

    def _register_module(
        self, module_class, module_name: Union[List[str], str, None] = None, force=False
    ):
        """Register a new module

        Args:
            module_class (class): module object, must be a class.
            module_name (Union[List[str], str, None], optional): Name or list of names to register the module_class under.
                If None, use __name__. Defaults to None.
            force (bool, optional): if True, force registering even if another module is registered under the given name. Defaults to False.

        Raises:
            TypeError: if module_class is not a class.
            KeyError: if module name is already registered and force is False.
        """
        if not inspect.isclass(module_class):
            raise TypeError(f"module must be a class, but got {type(module_class)}")

        if module_name is None:
            module_name = module_class.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if not force and name in self._module_dict:
                raise KeyError(f"{name} is already registered in {self.name}")
            self._module_dict[name] = module_class

    def register_module(
        self, name: Union[str, None] = None, force: bool = False, module=None
    ):
        """Register a module.
        A record will be added to `self._module_dict`, whose key is the class
        name or the specified name, and value is the class itself.
        It can be used as a decorator or a normal function.
        Example:
            >>> backbones = Registry('backbone')
            >>> @backbones.register_module()
            >>> class ResNet:
            >>>     pass
            >>> backbones = Registry('backbone')
            >>> @backbones.register_module(name='mnet')
            >>> class MobileNet:
            >>>     pass
            >>> backbones = Registry('backbone')
            >>> class ResNet:
            >>>     pass
            >>> backbones.register_module(ResNet)
        Args:
            name (Union[str, None]): The module name to be registered. If not specified, the class name will be used.
            force (bool, optional): Whether to override an existing class with
                the same name. Default: False.
            module (type): Module class to be registered.
        """
        if not isinstance(force, bool):
            raise TypeError(f"force must be a boolean, but got {type(force)}")

        # raise the error ahead of time
        if not (name is None or isinstance(name, str) or is_seq_of(name, str)):
            raise TypeError(
                "name must be either of None, an instance of str or a sequence"
                f"  of str, but got {type(name)}"
            )

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(module_class=module, module_name=name, force=force)
            return module

        # use it as a decorator: @x.register_module()
        def _register(cls):
            self._register_module(module_class=cls, module_name=name, force=force)
            return cls

        return _register


def build_from_cfg(cfg: EasyDict, registry: Registry, default_args=None):
    """Build a module from config dict.
    Args:
        cfg (EasyDict): config dict. It should at least contain the key "NAME".
        registry (Registry): The registry to search the type from.
    Returns:
        object: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict, but got {type(cfg)}")
    if "NAME" not in cfg:
        if default_args is None or "NAME" not in default_args:
            raise KeyError(
                '`cfg` or `default_args` must contain the key "NAME", '
                f"but got {cfg}\n{default_args}"
            )
    if not isinstance(registry, Registry):
        raise TypeError(f"registry must be a Registry object, but got {type(registry)}")

    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError(
            f"default_args must be a dict or None, but got {type(default_args)}"
        )

    if default_args is not None:
        cfg = merge_new_config(cfg, default_args)

    obj_type = cfg.get("NAME")

    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f"{obj_type} is not in the {registry.name} registry")
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f"type must be a str or valid type, but got {type(obj_type)}")
    try:
        return obj_cls(cfg)
    except Exception as e:
        # Normal TypeError does not print class name.
        raise type(e)(f"{obj_cls.__name__}: {e}")


def build_optimizer_from_cfg(
    cfg: EasyDict, base_model: nn.Module, registry: Registry, default_args=None
):
    """Build an optimizer form config dict. Slightly differs from the basic build_from_cfg function, since
    a pytorch optimizer also requires the base model parameters

    Args:
        cfg (EasyDict): config dict. It should at least contain the key "NAME".
        registry (Registry): The registry to search the type from.
    Returns:
        object: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict, but got {type(cfg)}")
    if "NAME" not in cfg:
        if default_args is None or "NAME" not in default_args:
            raise KeyError(
                '`cfg` or `default_args` must contain the key "NAME", '
                f"but got {cfg}\n{default_args}"
            )
    if not isinstance(registry, Registry):
        raise TypeError(f"registry must be a Registry object, but got {type(registry)}")

    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError(
            f"default_args must be a dict or None, but got {type(default_args)}"
        )

    if default_args is not None:
        cfg = merge_new_config(cfg, default_args)

    obj_type = cfg.get("NAME")

    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f"{obj_type} is not in the {registry.name} registry")
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f"type must be a str or valid type, but got {type(obj_type)}")
    try:
        return obj_cls(cfg, base_model)
    except Exception as e:
        # Normal TypeError does not print class name.
        raise type(e)(f"{obj_cls.__name__}: {e}")


def build_scheduler_from_cfg(
    cfg: EasyDict, optimizer: optim.Optimizer, registry: Registry, default_args=None
):
    """Build a scheduler form config dict. Slightly differs from the basic build_from_cfg function, since
    a pytorch scheduler also requires an optimizer object

    Args:
        cfg (EasyDict): config dict. It should at least contain the key "NAME".
        registry (Registry): The registry to search the type from.
    Returns:
        object: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict, but got {type(cfg)}")
    if "NAME" not in cfg:
        if default_args is None or "NAME" not in default_args:
            raise KeyError(
                '`cfg` or `default_args` must contain the key "NAME", '
                f"but got {cfg}\n{default_args}"
            )
    if not isinstance(registry, Registry):
        raise TypeError(f"registry must be a Registry object, but got {type(registry)}")

    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError(
            f"default_args must be a dict or None, but got {type(default_args)}"
        )

    if default_args is not None:
        cfg = merge_new_config(cfg, default_args)

    obj_type = cfg.get("NAME")

    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f"{obj_type} is not in the {registry.name} registry")
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f"type must be a str or valid type, but got {type(obj_type)}")
    try:
        return obj_cls(cfg, optimizer)
    except Exception as e:
        # Normal TypeError does not print class name.
        raise type(e)(f"{obj_cls.__name__}: {e}")
