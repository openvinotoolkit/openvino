"""
Copyright (c) 2019 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from ..utils import get_path
from ..logging import print_info
from ..adapters import Adapter
from ..config import PathField, StringField
from .loaders import Loader
from .launcher import Launcher, LauncherConfig


class DummyLauncherConfig(LauncherConfig):
    """
    Specifies configuration structure for Dummy launcher.
    """

    loader = StringField(choices=Loader.providers)
    data_path = PathField()
    adapter = StringField(choices=Adapter.providers, optional=True)


class DummyLauncher(Launcher):
    """
    Class for using predictions from another tool.
    """

    __provider__ = 'dummy'

    def __init__(self, config_entry: dict, *args, **kwargs):
        super().__init__(config_entry, *args, **kwargs)

        dummy_launcher_config = DummyLauncherConfig('Dummy_Launcher')
        dummy_launcher_config.validate(self.config)

        self.data_path = get_path(self.config['data_path'])

        self._loader = Loader.provide(self.config['loader'], self.data_path)

        print_info("{} predictions objects loaded from {}".format(len(self._loader), self.data_path))

    def predict(self, identifiers, *args, **kwargs):
        return [self._loader[identifier] for identifier in identifiers]

    def release(self):
        pass

    @property
    def batch(self):
        return 1

    @property
    def inputs(self):
        return None

    def get_all_inputs(self):
        return self.inputs

    @property
    def output_blob(self):
        return self.data_path
