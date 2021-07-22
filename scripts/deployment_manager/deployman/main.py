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

__version__ = '0.6'

import os
import argparse
import tempfile
from pathlib import Path
from typing import List

from deployman.logger import init_logger
from deployman.config import ConfigReader, ComponentFactory, Component
from deployman.ui import UserInterface

logger = init_logger('WARNING')


# main class
class DeploymentManager:
    def __init__(self, args, selected_targets: List[Component], components: List[Component]):
        self.args = args
        self.selected_targets = selected_targets
        self.components = components
        self.dependencies = []
        self.mandatory_components = []

    def get_dependencies(self):
        dependencies_names = []
        logger.debug("Updating dependencies...")
        for target in self.selected_targets:
            if hasattr(target, 'dependencies'):
                dependencies_names.extend(target.dependencies)
        # remove duplications
        dependencies_names = list(dict.fromkeys(dependencies_names))
        for dependency in dependencies_names:
            _target: Component
            for _target in self.components:
                if _target.name != dependency:
                    continue
                if not _target.is_exist():
                    FileNotFoundError("Dependency {} not available.".format(_target.name))
                self.dependencies.append(_target)

    def get_mandatory_component(self):
        for _target in self.components:
            _target: Component
            if hasattr(_target, 'mandatory'):
                if not _target.is_exist():
                    FileNotFoundError("Mandatory component {} not available.".format(_target.name))
                self.mandatory_components.append(_target)

    @staticmethod
    def packing_binaries(archive_name: str, target_dir: str, source_dir: str):
        logger.info('Archiving deploy package')
        if os.name == 'posix':
            archive_path = DeploymentManager.packing_binaries_posix(archive_name, target_dir, source_dir)
        else:
            archive_path = DeploymentManager.packing_binaries_windows(archive_name, target_dir, source_dir)
        logger.setLevel('INFO')
        logger.info("Deployment archive is ready. "
                    "You can find it here:\n\t{}".format(os.path.join(target_dir, archive_path)))

    @staticmethod
    def packing_binaries_posix(archive_name: str, target_dir: str, source_dir: str) -> str:
        extension = 'tar.gz'
        archive_file_name = '{}.{}'.format(archive_name, extension)
        archive_path = os.path.join(target_dir, archive_file_name)

        import tarfile
        with tarfile.open(archive_path, "w:gz") as tar_pac:
            total_files_number = DeploymentManager.count_files_number(source_dir)
            count = 0
            logger.info('Total number of files to add to the package: {}'.format(total_files_number))
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    count += 1
                    full_path = os.path.join(root, file)
                    if not os.path.isfile(full_path):
                        continue
                    relative_path = str(Path(full_path).relative_to(source_dir))
                    logger.info('Add {} {}/{} file to the package'.format(relative_path,
                                                                          count,
                                                                          total_files_number))
                    tar_pac.add(full_path, arcname=relative_path)
        return archive_path

    @staticmethod
    def packing_binaries_windows(archive_name: str, target_dir: str, source_dir: str) -> str:
        extension = 'zip'
        archive_file_name = '{}.{}'.format(archive_name, extension)
        archive_path = os.path.join(target_dir, archive_file_name)

        from zipfile import ZipFile
        with ZipFile(archive_path, "w") as zip_pac:
            total_files_number = DeploymentManager.count_files_number(source_dir)
            count = 0
            logger.info('Total number of files to add to the package: {}'.format(total_files_number))
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    count += 1
                    full_path = os.path.join(root, file)
                    if not os.path.isfile(full_path):
                        continue
                    relative_path = str(Path(full_path).relative_to(source_dir))
                    logger.info('Add {} {}/{} file to the package'.format(relative_path,
                                                                          count,
                                                                          total_files_number))
                    zip_pac.write(os.path.join(root, file), arcname=relative_path)
        return archive_path

    @staticmethod
    def count_files_number(source_dir: str) -> int:
        total_files_number = 0
        for root, dirs, files in os.walk(source_dir):
            total_files_number += len(files)
        return total_files_number

    def process(self):
        # get dependencies if have
        self.get_dependencies()
        # get mandatory components
        self.get_mandatory_component()

        logger.info('Collection information for components')
        with tempfile.TemporaryDirectory() as tmpdirname:
            for target in self.selected_targets:
                target: Component
                target.copy_files(tmpdirname)
            if self.dependencies:
                for dependency in self.dependencies:
                    dependency: Component
                    dependency.copy_files(tmpdirname)
            if self.mandatory_components:
                for target in self.mandatory_components:
                    target: Component
                    target.copy_files(tmpdirname)
            if self.args.user_data and os.path.exists(self.args.user_data):
                from shutil import copytree
                logger.info('Storing user data for deploy package ')
                copytree(self.args.user_data,
                         os.path.join(
                             tmpdirname,
                             os.path.basename(self.args.user_data.rstrip(os.path.sep))),
                         symlinks=True)
            self.packing_binaries(self.args.archive_name,
                                  self.args.output_dir, tmpdirname)


def main():
    # read main config
    cfg = ConfigReader(logger)

    # here we store all components
    components = []

    for component in cfg.components:
        components.append(ComponentFactory.create_component(component,
                                                            cfg.components[component],
                                                            logger))

    # list for only available components
    available_targets = []
    help_msg = ''

    for component in components:
        if component.is_exist() and not component.invisible:
            available_targets.append(component)
            help_msg += "{} - {}\n".format(component.name, component.ui_name)

    parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--targets", nargs="+", help="List of targets."
                                                     "Possible values: \n{}".format(help_msg))
    parser.add_argument("--user_data", type=str, help="Path to user data that will be added to "
                                                      "the deployment package", default=None)
    parser.add_argument("--output_dir", type=str, help="Output directory for deployment archive",
                        default=os.getenv("HOME", os.path.join(os.path.join(
                            os.path.dirname(__file__), os.pardir))))
    parser.add_argument("--archive_name", type=str, help="Name for deployment archive",
                        default="openvino_deploy_package", )
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)

    logger.info("Parsing command line arguments")
    args = parser.parse_args()

    selected_targets = []
    if not available_targets:
        exit("No available targets to packaging detected.\n"
             "Please check your OpenVINO installation.")

    ui = UserInterface(__version__, args, available_targets, logger)
    if not args.targets:
        ui.run()
        selected_targets = ui.get_selected_targets()
        args = ui.args
    else:
        for target in args.targets:
            target_name = target.lower()
            if not any(target_name == _target.name.lower() for _target in available_targets):
                raise ValueError("You input incorrect target. {} is not available.".format(target_name))
            for _target in available_targets:
                if _target.name.lower() == target_name:
                    selected_targets.append(_target)
    _manager = DeploymentManager(args, selected_targets, components)
    _manager.process()
