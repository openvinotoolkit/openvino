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
import uuid

from telemetry.backend.backend import TelemetryBackend
from telemetry.utils.message import Message, MessageType
from telemetry.utils.guid import get_or_generate_uid


class GABackend(TelemetryBackend):
    backend_url = 'https://www.google-analytics.com/collect'
    id = 'ga'

    def __init__(self, tid: str = None, app_name: str = None, app_version: str = None):
        super(GABackend, self).__init__(tid, app_name, app_version)
        if tid is None:
            tid = 'UA-17808594-29'
        self.tid = tid
        self.uid = get_or_generate_uid('openvino_ga_uid', lambda: str(uuid.uuid4()), is_valid_uuid4)
        self.app_name = app_name
        self.app_version = app_version
        self.default_message_attrs = {
            'v': '1',  # API Version
            'tid': self.tid,
            'cid': self.uid,
            'an': self.app_name,
            'av': self.app_version,
            'ua': 'Opera/9.80 (Windows NT 6.0) Presto/2.12.388 Version/12.14'  # dummy identifier of the browser
        }

    def send(self, message: Message):
        try:
            import requests
            requests.post(self.backend_url, message.attrs, timeout=1.0)
        except Exception:
            pass

    def build_event_message(self, event_category: str, event_action: str, event_label: str, event_value: int = 1,
                            **kwargs):
        data = self.default_message_attrs.copy()
        data.update({
            't': 'event',
            'ec': event_category,
            'ea': event_action,
            'el': event_label,
            'ev': event_value,
        })
        return Message(MessageType.EVENT, data)

    def build_session_start_message(self, **kwargs):
        data = self.default_message_attrs.copy()
        data.update({
            'sc': 'start',
            't': 'event',
            'ec': 'session',
            'ea': 'control',
            'el': 'start',
            'ev': 1,
        })
        return Message(MessageType.SESSION_START, data)

    def build_session_end_message(self, **kwargs):
        data = self.default_message_attrs.copy()
        data.update({
            'sc': 'end',
            't': 'event',
            'ec': 'session',
            'ea': 'control',
            'el': 'end',
            'ev': 1,
        })
        return Message(MessageType.SESSION_END, data)

    def build_error_message(self, error_msg: str, **kwargs):
        pass

    def build_stack_trace_message(self, error_msg: str, **kwargs):
        pass


def is_valid_uuid4(uid: str):
    try:
        uuid.UUID(uid, version=4)
    except ValueError:
        return False
    return True
