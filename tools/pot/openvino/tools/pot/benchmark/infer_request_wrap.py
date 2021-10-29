# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import threading
from ..utils.logger import get_logger

logger = get_logger(__name__)


class InferReqWrap:
    def __init__(self, request, index, callback_queue):
        self.id = index
        self.request = request
        self.request.set_completion_callback(self.callback, self.id)
        self.callback_queue = callback_queue

    def callback(self, statusCode, userdata):
        if userdata != self.id:
            logger.info('Request ID {} does not correspond to user data {}'.format(self.id, userdata))
        elif statusCode != 0:
            logger.info('Request {} failed with status code {}'.format(self.id, statusCode))
        self.callback_queue(self.id, self.request.latency)

    def start_async(self, input_data):
        self.request.async_infer(input_data)

    def infer(self, input_data):
        self.request.infer(input_data)
        self.callback_queue(self.id, self.request.latency)


class InferRequestsQueue:
    def __init__(self, requests):
        self.idle_ids = []
        self.requests = []
        self.times = []
        inf_len = len(requests)
        for i in range(inf_len):
            self.requests.append(InferReqWrap(requests[i], i, self.put_idle_request))
            self.idle_ids.append(i)
        self.cv = threading.Condition()

    def reset_times(self):
        self.times.clear()

    def put_idle_request(self, i, latency):
        self.cv.acquire()
        self.times.append(latency)
        self.idle_ids.append(i)
        self.cv.notify()
        self.cv.release()

    def get_idle_request(self):
        self.cv.acquire()
        while len(self.idle_ids) == 0:
            self.cv.wait()
        i = self.idle_ids.pop()
        self.cv.release()
        return self.requests[i]

    def wait_all(self):
        self.cv.acquire()
        while len(self.idle_ids) != len(self.requests):
            self.cv.wait()
        self.cv.release()
