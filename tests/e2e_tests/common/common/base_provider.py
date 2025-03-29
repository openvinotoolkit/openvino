# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

class BaseProviderMeta(type):
    def __new__(mcs, name, bases, attrs, **kwargs):
        cls = super().__new__(mcs, name, bases, attrs)
        # do not create container for abstract provider
        if '_is_base_provider' in attrs:
            return cls
        assert issubclass(cls, BaseProvider), "Do not use metaclass directly"
        cls.register(cls)
        return cls


class BaseProvider(metaclass=BaseProviderMeta):
    _is_base_provider = True
    registry = {}
    __action_name__ = None

    @classmethod
    def register(cls, provider):
        provider_name = getattr(cls, '__action_name__')
        if not provider_name:
            return
        cls.registry[provider_name] = provider

    @classmethod
    def provide(cls, provider, *args, **kwargs):
        if provider not in cls.registry:
            raise ValueError("Requested provider {} not registered".format(provider))
        root_provider = cls.registry[provider]
        root_provider.validate()
        return root_provider(*args, **kwargs)


class StepProviderMeta(type):
    def __new__(mcs, name, bases, attrs, **kwargs):
        cls = super().__new__(mcs, name, bases, attrs)
        # do not create container for abstract provider
        if '_is_base_provider' in attrs:
            return cls
        assert issubclass(cls, BaseStepProvider), "Do not use metaclass directly"
        cls.register(cls)
        return cls


class BaseStepProvider(metaclass=StepProviderMeta):
    _is_base_provider = True
    registry = {}
    __step_name__ = None

    @classmethod
    def register(cls, provider):
        provider_name = getattr(cls, '__step_name__', None)
        if not provider_name:
            return
        cls.registry[provider_name] = provider

    @classmethod
    def provide(cls, provider, *args, **kwargs):
        if provider not in cls.registry:
            raise ValueError("Requested provider {} not registered".format(provider))
        root_provider = cls.registry[provider]
        return root_provider(*args, **kwargs)
