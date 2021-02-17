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
import threading
from concurrent import futures
from time import sleep

from telemetry.backend.backend import TelemetryBackend
from telemetry.utils.message import Message

MAX_QUEUE_SIZE = 1000


class TelemetrySender:
    def __init__(self, max_workers=None):
        self.executor = futures.ThreadPoolExecutor(max_workers=max_workers)
        self.queue_size = 0
        self._lock = threading.Lock()

    def send(self, backend: TelemetryBackend, message: Message):
        def _future_callback(future):
            with self._lock:
                self.queue_size -= 1

        with self._lock:
            if self.queue_size < MAX_QUEUE_SIZE:
                fut = self.executor.submit(backend.send, message)
                fut.add_done_callback(_future_callback)
                self.queue_size += 1
            else:
                pass  # dropping a message because the queue is full

    def force_shutdown(self, timeout: float):
        """
        Forces all threads to be stopped after timeout. The "shutdown" method of the ThreadPoolExecutor removes only not
        yet scheduled threads and keep running the existing one. In order to stop the already running use some low-level
        attribute. The operation with low-level attributes is wrapped with the try/except to avoid potential crash if
        these attributes will removed or renamed.

        :param timeout: timeout to wait before the shutdown
        :return: None
        """
        try:
            with self._lock:
                if self.queue_size > 0:
                    sleep(timeout)
                self.executor.shutdown(wait=False)
                self.executor._threads.clear()
                futures.thread._threads_queues.clear()
        except Exception:
            pass
