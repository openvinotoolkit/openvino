#!/usr/bin/env python

#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

import posixpath
import re
from abc import ABC, abstractmethod


class Provider(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def name() -> str:
        pass

    @abstractmethod
    def create_model(self, preprocessing_request_data, options=None):
        pass

    @abstractmethod
    def get_model_info(self):
        pass

    @abstractmethod
    def get_tensor_info(self, tensor):
        pass

    @abstractmethod
    def prepare_input_tensors(self, input_files):
        pass

    @abstractmethod
    def infer(self, tensors_collection):
        pass

    @staticmethod
    def canonize_endpoint_name(endpoint_full_name):
        return endpoint_full_name.split('.')[0]


class Context(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def provider_names() -> list:
        pass

    @abstractmethod
    def create_provider(self, model_path: str) -> Provider:
        pass


class ProviderHolder:
    def __init__(self: str, registered_provides: list):
        self.registered_provides = registered_provides

    @staticmethod
    def __make_name__(root_prefix, provider_name):
        return posixpath.join(root_prefix, provider_name)

    @staticmethod
    def __remove_provider_prefix__(provider_name : str):
        packages = provider_name.split("/")
        if len(packages) == 0:
            return provider_name, None
        prefix = packages[0]
        packages = packages[1:]
        provider_name_specific = posixpath.join(*packages)

        return provider_name_specific, prefix

    def name(self) -> list:
        names = []
        for p in self.registered_provides:
            if issubclass(p, Context):
                names.extend(p.provider_names())
            elif issubclass(p, Provider):
                names.append(p.name())
            else:
                raise RuntimeError(
                    f'Class of provider: {p.__class__} is neither "Context" nor "Provider"'
                )
        return names

    def prefixed_names(self, prefix_root):
        return [
            ProviderHolder.__make_name__(prefix_root, p_name) for p_name in self.name()
        ]

    def get_provider_by_name(self, provider_name):
        for p in self.registered_provides:
            if issubclass(p, Context):
                for ap in p.provider_names():
                    if re.search(ap, provider_name):
                        provider_name_specific, _ = (
                            ProviderHolder.__remove_provider_prefix__(provider_name)
                        )
                        return p.get_provider_by_name(provider_name_specific)
            elif issubclass(p, Provider):
                if re.search(p.name(), provider_name):
                    return p, provider_name
            else:
                raise RuntimeError(
                    f'Class of provider: {p.__class__} is neither "Context" nor "Provider"'
                )
        return None, None
