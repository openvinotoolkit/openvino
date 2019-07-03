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

from pathlib import Path

from ..representation import ReIdentificationAnnotation


def read_directory(directory, query, image_pattern):
    pids = set()
    images = []
    for image in directory.glob("*.jpg"):
        pid, camid = map(int, image_pattern.search(image.name).groups())
        if pid == -1:
            continue

        camid -= 1
        pids.add(pid)

        identifier = str(Path(directory.name) / image.name)
        images.append(ReIdentificationAnnotation(identifier, camid, pid, query))

    return images, pids


def check_dirs(dirs, parent_dir, arg_name='data_dir'):
    for directory in dirs:
        if directory.is_dir():
            continue

        message_pattern = "{directory} not found in {parent_dir}. Check {arg_name} is pointed to a correct directory"
        raise FileNotFoundError(message_pattern.format(directory=directory, parent_dir=parent_dir, arg_name=arg_name))
