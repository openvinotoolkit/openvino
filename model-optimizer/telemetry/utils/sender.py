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
from concurrent import futures

from telemetry.backend.backend import TelemetryBackend
from telemetry.utils.message import Message

MAX_QUEUE_SIZE = 100  # maximum number of messages in the queue
MAX_TIMEOUT = 10  # maximum timeout in seconds to send the data


class TelemetrySender:
    def __init__(self, max_workers=None):
        self.executor = futures.ThreadPoolExecutor(max_workers=max_workers)
        self.queue_size = 0

    def send(self, backend: TelemetryBackend, message: Message):
        def _future_callback(future):
            self.queue_size -= 1

        if self.queue_size < MAX_QUEUE_SIZE:
            fut = self.executor.submit(backend.send, message)
            fut.add_done_callback(_future_callback)
            self.queue_size += 1
        else:
            print('Dropping message since the queue is full')
            pass  # dropping message
