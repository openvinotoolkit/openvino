"""
 Copyright (C) 2018-2021 Intel Corporation

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

import unittest
import time

from telemetry.utils.sender import TelemetrySender


class FakeTelemetryBackend:
    def send(self, param):
        pass


class FakeTelemetryBackendWithSleep:
    def send(self, param):
        time.sleep(1)


class TelemetrySenderStress(unittest.TestCase):
    def test_stress(self):
        """
        Stress tests to schedule a lot of threads which works super fast (do nothing) with sending messages."
        """
        tm = TelemetrySender()
        fake_backend = FakeTelemetryBackend()
        for _ in range(1000000):
            tm.send(fake_backend, None)

    def test_check_shutdown(self):
        """
        Stress test to schedule many threads taking 1 second and then ask to force shutdown. Make sure that the elapsed
        time is small.
        """
        tm = TelemetrySender()
        fake_backend = FakeTelemetryBackendWithSleep()
        # schedule many requests which just wait 1 second
        for _ in range(100000):
            tm.send(fake_backend, None)

        start_time = time.time()
        # ask to shutdown with timeout of 1 second
        tm.force_shutdown(1)
        while len(tm.executor._threads):
            pass
        # check that no more than 3 seconds spent
        self.assertTrue(time.time() - start_time < 3)

    def test_check_shutdown_negative(self):
        """
        Test to check that without forcing shutdown total execution time is expected.
        """
        tm = TelemetrySender(1)  # only one worker thread
        fake_backend = FakeTelemetryBackendWithSleep()
        start_time = time.time()
        # schedule 5 requests which totally should work more than 4 seconds
        for _ in range(5):
            tm.send(fake_backend, None)

        try:
            # wait until all threads finish their work. We use internal ThreadPoolExecutor attribute _work_queue to make
            # sure that all workers completed their work, so the whole code is wrapped to try/except to avoid exceptions
            # if internal implementation is changed in the future
            while tm.executor._work_queue.qsize():
                pass
            self.assertTrue(time.time() - start_time > 4.0)
        except:
            pass
