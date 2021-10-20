"""
 Copyright (c) 2018-2021 Intel Corporation

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

import os
import platform
from shutil import copytree, copy
import json


# class that works with the components from config
class Component:
    def __init__(self, name, properties, logger):
        self.name = name
        for k, v in properties.items():
            setattr(self, k, str(v, 'utf-8') if isinstance(v, bytes) else v)
        self.available = True
        self.invisible = 'ui_name' not in properties
        self.selected = False
        self.logger = logger
        self.root_dir = os.getenv('INTEL_OPENVINO_DIR',
                                  os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                               os.pardir, os.pardir,
                                                               os.pardir, os.pardir)))

    def is_exist(self):
        self.logger.debug("Checking {} component...".format(self.name))
        for obj in self.files:
            obj = os.path.join(self.root_dir, obj)
            if not (os.path.isfile(obj) or os.path.isdir(obj)):
                self.logger.warning("[{}] Missing {}".format(self.name, obj))
                self.available = False
                self.selected = False
                return False
        return True

    def invert_selection(self):
        self.selected = not self.selected

    def set_value(self, attr, value):
        setattr(self, attr, value)

    def copy_files(self, destination):
        if not self.is_exist():
            raise FileNotFoundError("Files for component {} not found. "
                                    "Please check your OpenVINO installation".
                                    format(self.name))
        else:
            if not os.path.exists(destination):
                os.makedirs(destination)
            for obj in self.files:
                src = os.path.join(self.root_dir, obj.strip('\n'))
                dst = os.path.join(destination, obj.strip('\n'))
                self.logger.debug("[{}] Copy files:: Processing {}...".format(self.name, src))
                if not os.path.exists(os.path.dirname(dst)):
                    os.makedirs(os.path.dirname(dst))
                if os.path.isdir(src):
                    copytree(src, dst, symlinks=True)
                else:
                    copy(src, dst)


class ComponentFactory:
    @staticmethod
    def create_component(name, properties, logger):
        return Component(name, properties, logger)


# class that operating with JSON configs
class ConfigReader:
    def __init__(self, logger):
        logger.info("Determining the current OS for config selection...")
        current_os = platform.system().lower()
        cfg_path = os.path.join(os.path.dirname(__file__), os.pardir,
                                "configs/{}.json".format(current_os))
        if os.path.isfile(cfg_path):
            logger.info("Loading {}.cfg...".format(current_os))
            with open(cfg_path, encoding='utf-8') as main_cfg:
                self.cfg = json.load(main_cfg)
            self.version = self.cfg['version']
            self.components = self.cfg['components']
            logger.info("Successfully loaded.\nConfig version: {}".format(self.version))
        else:
            raise RuntimeError("Config can't be found at {}".format(os.path.abspath(cfg_path)))

