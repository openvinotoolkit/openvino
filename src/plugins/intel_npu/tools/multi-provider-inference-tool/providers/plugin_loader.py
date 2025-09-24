#!/usr/bin/env python

#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import importlib.util
import os
import re
import sys

from pathlib import Path
import providers.interfaces
from schema.validator import JSONSchemaValidator


class ExecutionProvider:
    model_info_schema = JSONSchemaValidator.load_from_file("model")
    input_source_schema = JSONSchemaValidator.load_from_file("input_source")

    def __init__(self, provider_impl):
        self.provider_impl = provider_impl

    def create_model(self, preprocessing_request_data, options=None):
        if len(preprocessing_request_data.preproc_per_io) != 0:
            assert JSONSchemaValidator.is_valid(ExecutionProvider.model_info_schema, preprocessing_request_data.preproc_per_io)
        return self.provider_impl.create_model(preprocessing_request_data, options)

    def get_model_info(self):
        return self.provider_impl.get_model_info()

    def get_tensor_info(self, tensor):
        return self.provider_impl.get_tensor_info(tensor)

    def prepare_input_tensors(self, input_files):
        for data in input_files.values():
            assert JSONSchemaValidator.is_valid(ExecutionProvider.input_source_schema, data)
        return self.provider_impl.prepare_input_tensors(input_files)

    def infer(self, tensors_collection):
        return self.provider_impl.infer(tensors_collection)


class ProviderFactory:
    def __init__(self):
        self.provider_plugins = ProviderFactory.initialize()

    @staticmethod
    def initialize():
        provider_plugins = {}

        plugin_dir = Path(__file__).parent
        plugin_paths = sorted(plugin_dir.glob("**/provider.py"))
        count = 0
        loaded = 0
        loading_errors = {}
        for source_file in plugin_paths:
            try:
                modname = "provider" + str(count)

                spec = importlib.util.spec_from_file_location(modname, source_file)
                module = importlib.util.module_from_spec(spec)
                sys.modules[modname] = module
                spec.loader.exec_module(module)
                provider_plugins[source_file] = module
                loaded += 1
            except Exception as e:
                loading_errors[source_file] = str(e)
                pass
            finally:
                count += 1

        error_description = ""
        if len(loading_errors) != 0:
            for p, e in loading_errors.items():
                error_description += "\tModule: " + str(p) + "\n\t\t" + e + "\n"
        if count == 0:
            raise RuntimeError(f"No any plugin detected")
        if loaded == 0:
            raise RuntimeError(f"Cannot load any plugin, attempted: {count}.\nPlugins loading logs:\n{error_description}")
        if loaded != count:
            ext_description = ""
            if len(error_description):
                ext_description = "\nNot loaded plugins:\n" + error_description
            print(f"Loaded plugins: {loaded}/{count}{ext_description}")
        return provider_plugins

    def get_avaialable_providers(self):
        ret_list = []
        for p in self.provider_plugins.values():
            ret_list.extend(p.provider_names())
        return ret_list

    def create_provider_ctx(self, provider_name: str) -> providers.interfaces.Context:
        available_providers = self.get_avaialable_providers()
        found = False
        for ap in available_providers:
            if re.search(ap, provider_name):
                found = True
                break

        if not found:
            raise RuntimeError(f"Unrecognized provider: {provider_name}. Please enter a correct one, which met a regexp from the list: {available_providers}")
        for p in self.provider_plugins.values():
            for ap in p.provider_names():
                if re.search(ap, provider_name):
                    return p.create(provider_name)
        raise RuntimeError(f"No provider for {provider_name}")

    def create_provider_for_model(self, ctx: providers.interfaces.Context, model_path: str) -> ExecutionProvider:
        return ExecutionProvider(ctx.create_provider(model_path))
