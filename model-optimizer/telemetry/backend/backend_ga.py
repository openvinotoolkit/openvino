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
import requests

from telemetry.backend.backend import TelemetryBackend
from telemetry.utils.message import Message, MessageType


class GABackend(TelemetryBackend):
    backend_url = 'https://www.google-analytics.com/collect'

    def __init__(self, tid: str, uid: str, app_name: str, app_version: str):
        super(GABackend, self).__init__(tid, uid, app_name, app_version)

        self.tid = tid
        self.uid = uid
        self.app_name = app_name
        self.app_version = app_version

    def send(self, backend: TelemetryBackend, message: Message):
        print("Sending message: {}".format(message.attrs))
        requests.post(self.backend_url, message.attrs)

    def build_event_message(self, event_category: str, event_action: str, event_label: str, event_value: int = 1, **kwargs):
        data = {
            'v': '1',  # API Version
            'tid': self.tid,
            'cid': self.uid,
            'an': self.app_name,
            'av': self.app_version,
            't': 'event',
            'ec': event_category,
            'ea': event_action,
            'el': event_label,
            'ev': event_value,
            'ua': 'Opera/9.80 (Windows NT 6.0) Presto/2.12.388 Version/12.14'  # identifier of the browser
        }

        # if the session attribute is specified it triggers adding special attribute to track session
        if 'session' in kwargs and kwargs['session'] is not None:
            session = kwargs['session']
            assert session in ('start', 'end')
            data['sc'] = session
        return Message(MessageType.EVENT, data)

    def build_error_message(self, error_msg: str, **kwargs):
        pass

    def build_stack_trace_message(self, error_msg: str, **kwargs):
        pass
