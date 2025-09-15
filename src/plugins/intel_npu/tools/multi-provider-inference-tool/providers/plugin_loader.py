#!/usr/bin/env python

#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

import importlib.util
import os
import re
import sys

from pathlib import Path
import providers.interfaces

class ProviderFactory():
    def __init__(self):
        self.provider_plugins = ProviderFactory.initialize()

    @staticmethod
    def initialize():
        provider_plugins = {}

        plugin_dir = Path(__file__).parent
        plugin_paths = sorted(plugin_dir.glob('**/provider.py'))
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
            for p,e in loading_errors.items():
                error_description += "\tModule: " + str(p) + "\n\t\t" + e + "\n"
        if count == 0:
            raise RuntimeError(f"No any plugin detected")
        if loaded == 0:
            raise RuntimeError(f"Cannot load any plugin, attempted: {count}.\nPlugins loading logs:\n{error_description}")
        if loaded != count:
            ext_description = ""
            if len(error_description):
                ext_description = '\nNot loaded plugins:\n' + error_description
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
            raise RuntimeError(
                f"Unrecognized provider: {provider_name}. Please enter a correct one, which met a regexp from the list: {available_providers}"
            )
        for p in self.provider_plugins.values():
            for ap in p.provider_names():
                if re.search(ap, provider_name):
                    return p.create(provider_name)
        raise RuntimeError(f"No provider for {provider_name}")


    def create_provider_for_model(self, ctx: providers.interfaces.Context, model_path: str) -> providers.interfaces.Provider:
        return ctx.create_provider(model_path)
