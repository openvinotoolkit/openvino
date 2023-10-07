# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from threading import Event, Thread
import psutil
import time
import os

class MemConsumption:
    def __init__(self):
        self.g_exitGetMemThread = False
        self.g_EndCollectMem = False
        self.g_maxRssMemConsumption = -1
        self.g_maxSharedMemConsumption = -1
        self.g_event = Event()
        self.g_data_event = Event()
    
    def collect_memory_consumption(self):
        while self.g_exitGetMemThread == False:
            self.g_event.wait()
            while True:
                process = psutil.Process(os.getpid())
                rss_mem_data = process.memory_info().rss / float(2 ** 20)
                try:
                    shared_mem_data = process.memory_info().shared / float(2 ** 20)
                except:
                    shared_mem_data = -1
                if rss_mem_data > self.g_maxRssMemConsumption:
                    self.g_maxRssMemConsumption = rss_mem_data
                if shared_mem_data > self.g_maxSharedMemConsumption:
                    self.g_maxSharedMemConsumption = shared_mem_data
                self.g_data_event.set()
                if self.g_EndCollectMem == True:
                    self.g_event.set()
                    self.g_event.clear()
                    self.g_EndCollectMem = False
                    break
                time.sleep(500/1000)

    def start_collect_memory_consumption(self):
        self.g_EndCollectMem = False
        self.g_event.set()
    
    def end_collect_momory_consumption(self):
        self.g_EndCollectMem = True
        self.g_event.wait()

    def get_max_memory_consumption(self):
        self.g_data_event.wait()
        self.g_data_event.clear()
        return self.g_maxRssMemConsumption, self.g_maxSharedMemConsumption
    
    def clear_max_memory_consumption(self):
        self.g_maxRssMemConsumption = -1
        self.g_maxSharedMemConsumption = -1

    def start_collect_memConsumption_thread(self):
        self.t_mem_thread = Thread(target=self.collect_memory_consumption)
        self.t_mem_thread.start()

    def end_collect_memConsumption_thread(self):
        self.g_event.set()
        self.g_data_event.set()
        self.g_EndCollectMem = True
        self.g_exitGetMemThread = True
        self.t_mem_thread.join()