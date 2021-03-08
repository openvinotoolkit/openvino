"""
 Copyright (C) 2017-2021 Intel Corporation

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
from platform import system

import telemetry.utils.isip as isip


def save_uid_to_file(file_name: str, uid: str):
    """
    Save the uid to the specified file
    """
    try:
        # create directories recursively first
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

        with open(file_name, 'w') as file:
            file.write(uid)
    except Exception as e:
        print('Failed to generate the UID file: {}'.format(str(e)))
        return False
    return True


def get_or_generate_uid(file_name: str, generator: callable, validator: [callable, None]):
    """
    Get existing UID or generate a new one.
    :param file_name: name of the file with the UID
    :param generator: the function to generate the UID
    :param validator: the function to validate the UID
    :return: existing or a new UID file
    """
    full_path = os.path.join(get_uid_path(), file_name)
    uid = None
    if os.path.exists(full_path):
        with open(full_path, 'r') as file:
            uid = file.readline().strip()

        if uid is not None and (validator is not None and not validator(uid)):
            uid = None

    if uid is None:
        uid = generator()
        save_uid_to_file(full_path, uid)
    return uid


def get_uid_path():
    """
    Returns a directory with the the OpenVINO randomly generated UUID file.

    :return: the directory with the the UUID file
    """
    platform = system()
    subdir = None
    if platform == 'Windows':
        subdir = 'Intel Corporation'
    elif platform in ['Linux', 'Darwin']:
        subdir = '.intel'
    if subdir is None:
        raise Exception('Failed to determine the operation system type')

    return os.path.join(isip.isip_consent_base_dir(), subdir)
