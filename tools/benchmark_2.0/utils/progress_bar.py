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

from progress.bar import Bar


class ProgressBar:
    def __init__(self, total_num, stream_output=False, progress_enabled=False):
        self.stream_output = stream_output
        self.is_finished = True
        self.progress_enabled = progress_enabled
        self.percent_to_update = 1
        self.cur_progress = 0
        self.total_num = total_num
        self.reset(total_num)

    def add_progress(self, num):
        self.is_finished = False
        if self.progress_enabled:
            self.cur_progress += num
            total_progress = self.bar.max
            if self.cur_progress > total_progress:
                self.cur_progress = total_progress

            prev_progress = self.bar.index
            prev_percent = 100 * prev_progress / total_progress
            cur_percent = 100 * self.cur_progress / total_progress
            if prev_progress == 0 or \
               self.cur_progress == total_progress or \
               prev_percent + self.percent_to_update <= cur_percent:
                self.bar.next(self.cur_progress - self.bar.index)
                if self.stream_output:
                    print()

    def finish(self, num=0):
        if num:
            self.add_progress(num)

        self.is_finished = True
        if self.progress_enabled:
            self.bar.finish()
            print()

    def reset(self, total_num):
        if self.progress_enabled:
            self.bar = Bar('Progress:', max=total_num, fill='.', suffix='%(percent).d%%')

    def new_bar(self, total_num):
        if self.is_finished:
            self.reset(total_num)
        else:
            raise Exception('Cannot create a new bar. Current bar is still in progress')
