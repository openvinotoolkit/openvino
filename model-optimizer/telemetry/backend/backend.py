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

import abc

from telemetry.utils.message import Message


class BackendRegistry:
    """
    The class that stores information about all registered telemetry backends
    """
    r = {}

    @classmethod
    def register_backend(cls, id: str, backend):
        cls.r[id] = backend

    @classmethod
    def get_backend(cls, id: str):
        assert id in cls.r, 'The backend with id "{}" is not registered'.format(id)
        return cls.r.get(id)


class TelemetryBackendMetaClass(abc.ABCMeta):
    def __init__(cls, clsname, bases, methods):
        super().__init__(clsname, bases, methods)
        if cls.id is not None:
            BackendRegistry.register_backend(cls.id, cls)


class TelemetryBackend(metaclass=TelemetryBackendMetaClass):
    id = None

    @abc.abstractmethod
    def __init__(self, tid: str, app_name: str, app_version: str):
        """
        Initializer of the class
        :param tid: database id
        :param app_name: name of the application
        :param app_version: version of the application
        """

    @abc.abstractmethod
    def send(self, message: Message):
        """
        Sends the message to the backend.
        :param message: The Message object to send
        :return: None
        """

    @abc.abstractmethod
    def build_event_message(self, event_category: str, event_action: str, event_label: str, event_value: int = 1,
                            **kwargs):
        """
        Should return the Message object build from the event message.
        """

    @abc.abstractmethod
    def build_error_message(self, error_msg: str, **kwargs):
        """
        Should return the Message object build from the error message.
        """

    @abc.abstractmethod
    def build_stack_trace_message(self, error_msg: str, **kwargs):
        """
        Should return the Message object build from the stack trace message.
        """

    @abc.abstractmethod
    def build_session_start_message(self, **kwargs):
        """
        Should return the Message object corresponding to the session start.
        """

    @abc.abstractmethod
    def build_session_end_message(self, **kwargs):
        """
        Should return the Message object corresponding to the session end.
        """
