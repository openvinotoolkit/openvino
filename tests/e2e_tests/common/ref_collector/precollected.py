# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch

from e2e_tests.common.ref_collector.provider import ClassProvider
from e2e_tests.test_utils.path_utils import resolve_file_path
import numpy as np
import logging as log
import sys


class PrecollectedRefs(ClassProvider):
    """Precollected reference provider."""
    __action_name__ = "precollected"
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    def __init__(self, config):
        self.path = resolve_file_path(config['path'], as_str=True)

    def get_refs(self, **kwargs):
        """Return existing reference results."""
        log.info("Reading references from path {}".format(self.path))
        return dict(np.load(self.path, allow_pickle=True))


class PrecollectedTorchRefs(ClassProvider):
    """Precollected reference provider."""
    __action_name__ = "torch_precollected"
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    def __init__(self, config):
        self.path = resolve_file_path(config['path'], as_str=True)

    def get_refs(self, **kwargs):
        """Return existing reference results."""
        log.info("Reading references from path {}".format(self.path))
        return torch.load(self.path)


class CustomRefCollector(ClassProvider):
    __action_name__ = "custom_ref_collector"

    def __init__(self, config):
        self.execution_function = config["execution_function"]

    def get_refs(self, **kwargs):
        return self.execution_function()
