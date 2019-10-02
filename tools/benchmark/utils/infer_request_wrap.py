"""
 Copyright (C) 2018-2019 Intel Corporation

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

from datetime import datetime
import threading


class InferReqWrap:
    def __init__(self, request, req_id, callback_queue):
        self.req_id = req_id
        self.request = request
        self.request.set_completion_callback(self.callback, self.req_id)
        self.callbackQueue = callback_queue

    def callback(self, status_code, user_data):
        if user_data != self.req_id:
            print('Request ID {} does not correspond to user data {}'.format(self.req_id, user_data))
        elif status_code:
            print('Request {} failed with status code {}'.format(self.req_id, status_code))
        self.callbackQueue(self.req_id, self.request.latency)

    def start_async(self, input_data):
        self.request.async_infer(input_data)

    def infer(self, input_data):
        self.request.infer(input_data)
        self.callbackQueue(self.req_id, self.request.latency)


class InferRequestsQueue:
    def __init__(self, requests):
        self.idleIds = []
        self.requests = []
        self.times = []
        for req_id in range(len(requests)):
            self.requests.append(InferReqWrap(requests[req_id], req_id, self.put_idle_request))
            self.idleIds.append(req_id)
        self.startTime = datetime.max
        self.endTime = datetime.min
        self.cv = threading.Condition()

    def reset_times(self):
        self.times.clear()

    def get_duration_in_seconds(self):
        return (self.endTime - self.startTime).total_seconds()

    def put_idle_request(self, req_id, latency):
        self.cv.acquire()
        self.times.append(latency)
        self.idleIds.append(req_id)
        self.endTime = max(self.endTime, datetime.now())
        self.cv.notify()
        self.cv.release()

    def get_idle_request(self):
        self.cv.acquire()
        while len(self.idleIds) == 0:
            self.cv.wait()
        req_id = self.idleIds.pop()
        self.startTime = min(datetime.now(), self.startTime)
        self.cv.release()
        return self.requests[req_id]

    def wait_all(self):
        self.cv.acquire()
        while len(self.idleIds) != len(self.requests):
            self.cv.wait()
        self.cv.release()
