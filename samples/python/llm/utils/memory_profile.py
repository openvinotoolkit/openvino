# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from threading import Event, Thread
import psutil
import time
import os


class MemConsumption:
    def __init__(self):
        """Initialize MemConsumption."""
        self.g_exit_get_mem_thread = False
        self.g_end_collect_mem = False
        self.g_max_rss_mem_consumption = -1
        self.g_max_shared_mem_consumption = -1
        self.g_event = Event()
        self.g_data_event = Event()

    def collect_memory_consumption(self):
        """Collect the data."""
        while self.g_exit_get_mem_thread is False:
            self.g_event.wait()
            while True:
                process = psutil.Process(os.getpid())
                rss_mem_data = process.memory_info().rss / float(2**20)
                try:
                    shared_mem_data = process.memory_info().shared / float(2**20)
                except Exception:
                    shared_mem_data = -1
                if rss_mem_data > self.g_max_rss_mem_consumption:
                    self.g_max_rss_mem_consumption = rss_mem_data
                if shared_mem_data > self.g_max_shared_mem_consumption:
                    self.g_max_shared_mem_consumption = shared_mem_data
                self.g_data_event.set()
                if self.g_end_collect_mem is True:
                    self.g_event.set()
                    self.g_event.clear()
                    self.g_end_collect_mem = False
                    break
                time.sleep(500 / 1000)

    def start_collect_memory_consumption(self):
        """Start collect."""
        self.g_end_collect_mem = False
        self.g_event.set()

    def end_collect_momory_consumption(self):
        """Stop collect."""
        self.g_end_collect_mem = True
        self.g_event.wait()

    def get_max_memory_consumption(self):
        """Return the data."""
        self.g_data_event.wait()
        self.g_data_event.clear()
        return self.g_max_rss_mem_consumption, self.g_max_shared_mem_consumption

    def clear_max_memory_consumption(self):
        """Clear MemConsumption."""
        self.g_max_rss_mem_consumption = -1
        self.g_max_shared_mem_consumption = -1

    def start_collect_mem_consumption_thread(self):
        """Start the thread."""
        self.t_mem_thread = Thread(target=self.collect_memory_consumption)
        self.t_mem_thread.start()

    def end_collect_mem_consumption_thread(self):
        """End the thread."""
        self.g_event.set()
        self.g_data_event.set()
        self.g_end_collect_mem = True
        self.g_exit_get_mem_thread = True
        self.t_mem_thread.join()
