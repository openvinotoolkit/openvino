"""
 Copyright (C) 2018-2020 Intel Corporation

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
import uuid
from platform import system

import telemetry.utils.isip as isip
from telemetry.backend.backend_ga import GABackend
from telemetry.utils.sender import TelemetrySender


class SingletonMetaClass(type):
    def __init__(self, cls_name, super_classes, dic):
        self.__single_instance = None
        super().__init__(cls_name, super_classes, dic)

    def __call__(cls, *args, **kwargs):
        if cls.__single_instance is None:
            cls.__single_instance = super(SingletonMetaClass, cls).__call__(*args, **kwargs)
        return cls.__single_instance


class Telemetry(metaclass=SingletonMetaClass):
    """
    The main class to send telemetry data. It uses singleton pattern. The instance should be initialized with the
    application name, version and tracking id just once. Later the instance can be created without parameters.
    """
    def __init__(self, app_name: str = None, app_version: str = None, tid: [None, str] = None):
        if app_name is not None:
            self.consent = True # For testing purposes # isip.isip_consent() == isip.ISIPConsent.APPROVED
            self.uid = Telemetry._get_or_generate_uid()
            # override default tid
            if tid is not None:
                self.tid = tid
            self.backend = GABackend(self.tid, self.uid, app_name, app_version)
            self.sender = TelemetrySender()
        else:  # use already configured instance
            assert self.sender is not None, 'The first instantiation of the Telemetry should be done with the ' \
                                            'application name and version'

    @staticmethod
    def _is_valid_uuid(uid: str):
        try:
            uuid.UUID(uid, version=4)
        except ValueError:
            return False
        return True

    @staticmethod
    def _generate_uid():
        """
        The function which randomly generates the UID for the user and save it to the specific file in the user's home
        directory.
        :return: the generated UID
        """
        try:
            uid_file = Telemetry._get_uid_file_path()

            # create directories recursively first
            os.makedirs(os.path.dirname(uid_file), exist_ok=True)

            with open(uid_file, 'w') as file:
                uid = str(uuid.uuid4())
                file.write(uid)
                return uid

        except Exception as e:
            print('Failed to generate the UID file: {}'.format(str(e)))
            return None

    @staticmethod
    def _get_or_generate_uid():
        if os.path.exists(Telemetry._get_uid_file_path()):
            uid = None
            with open(Telemetry._get_uid_file_path(), 'r') as file:
                uid = file.readline().strip()

            # if the UUID is not specified or is in incorrect format then generate a new one
            if uid is None or not Telemetry._is_valid_uuid(uid):
                return Telemetry._generate_uid()
            return uid
        else:
            return Telemetry._generate_uid()

    @staticmethod
    def _get_uid_file_path():
        """
        Returns a full path to the file with the OpenVINO randomly generated UUID file.
        :return: the full path to the UUID file.
        """
        platform = system()
        subdir = None
        if platform == 'Windows':
            subdir = 'Intel Corporation'
        elif platform in ['Linux', 'Darwin']:
            subdir = '.intel'
        if subdir is None:
            raise Exception('Failed to determine the operation system type')
        return os.path.join(isip.isip_consent_base_dir(), subdir, 'openvino_uuid')

    def send_event(self, event_category: str, event_action: str, event_label: str, event_value: int = 1, **kwargs):
        """
        Send single event.

        :param event_category: category of the event
        :param event_action: action of the event
        :param event_label: the label associated with the action
        :param event_value: the integer value corresponding to this label
        :param kwargs: additional parameters
        :return: None
        """
        if self.consent:
            self.sender.send(self.backend, self.backend.build_event_message(event_category, event_action, event_label,
                                                                            event_value, **kwargs))

    def bulk_send_event(self, event_category: str, event_action: str, values: dict, **kwargs):
        """
        Send bulk of events with the same category and action but different label/value pairs specified with a "values"
        dictionary.

        :param event_category: category of the event
        :param event_action: action of the event
        :param values: the dictionary with label/value pairs
        :param kwargs: additional parameters
        :return: None
        """
        if self.consent:
            pass

    def send_error(self, error_msg: str, **kwargs):
        if self.consent:
            pass

    def send_stack_trace(self, stack_trace: str, **kwargs):
        if self.consent:
            pass
