# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from e2e_tests.test_utils.path_utils import resolve_file_path
from .provider import ClassProvider
import logging as log
import sys


class Pregenerated(ClassProvider):
    """Pregenerated IR provider."""
    __action_name__ = "pregenerated"
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    def __init__(self, config):
        self.xml = resolve_file_path(config.get("xml")) if config.get("xml") else None
        self.bin = resolve_file_path(config.get("bin")) if config.get("bin") else None
        self.ov_model = config.get("ov_model")
        self.mo_log = None

    def get_ir(self, data=None):
        log.info("Reading ie IR from files:\n\t\tXML: {}\n\t\tBIN: {}".format(self.xml, self.bin))
        """Return existing IR."""
        return self.xml
