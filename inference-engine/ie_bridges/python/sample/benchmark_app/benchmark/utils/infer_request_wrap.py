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

from ctypes import *
from datetime import datetime
import threading

class InferReqWrap:
    def __init__(self, request, id, callbackQueue):
        self.id = id
        self.request = request
        self.request.set_completion_callback(self.callback, self.id)
        self.callbackQueue = callbackQueue

    def callback(self, statusCode, userdata):
        if (userdata != self.id):
            print("Request ID {} does not correspond to user data {}".format(self.id, userdata))
        elif statusCode != 0:
            print("Request {} failed with status code {}".format(self.id, statusCode))
        self.callbackQueue(self.id, self.request.latency)

    def startAsync(self, input_data):
        self.request.async_infer(input_data)

    def infer(self, input_data):
        self.request.infer(input_data)
        self.callbackQueue(self.id, self.request.latency);

class InferRequestsQueue:
    def __init__(self, requests):
      self.idleIds = []
      self.requests = []
      self.times = []
      for id in range(0, len(requests)):
          self.requests.append(InferReqWrap(requests[id], id, self.putIdleRequest))
          self.idleIds.append(id)
      self.startTime = datetime.max
      self.endTime = datetime.min
      self.cv = threading.Condition()

    def resetTimes(self):
      self.times.clear()

    def getDurationInSeconds(self):
      return (self.endTime - self.startTime).total_seconds()

    def putIdleRequest(self, id, latency):
      self.cv.acquire()
      self.times.append(latency)
      self.idleIds.append(id)
      self.endTime = max(self.endTime, datetime.now())
      self.cv.notify()
      self.cv.release()

    def getIdleRequest(self):
        self.cv.acquire()
        while len(self.idleIds) == 0:
            self.cv.wait()
        id = self.idleIds.pop();
        self.startTime = min(datetime.now(), self.startTime);
        self.cv.release()
        return self.requests[id]

    def waitAll(self):
        self.cv.acquire()
        while len(self.idleIds) != len(self.requests):
            self.cv.wait()
        self.cv.release()
