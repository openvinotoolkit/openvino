#   Intel® Single Event API
#
#   This file is provided under the BSD 3-Clause license.
#   Copyright (c) 2021, Intel Corporation
#   All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#       Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#       Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#       Neither the name of the Intel Corporation nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
#   IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
#   HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
import os
import csv
import shutil
from sea_runtool import GraphCombiner

class Stat(GraphCombiner):
    def __init__(self, args, tree):
        GraphCombiner.__init__(self, args, tree)

    def get_targets(self):
        return [self.args.output + ".csv"]

    def finish(self):
        GraphCombiner.finish(self)
        delim = ','
        with open(self.get_targets()[-1], 'w') as f:
            writer = csv.writer(f, delimiter=delim)
            writer.writerow(["domain", "name", "min", "max", "avg", "total", "count"])
            for domain, data in self.per_domain.items():
                for task_name, task_data in data['tasks'].items():
                    time = task_data['time']
                    writer.writerow([domain, task_name, min(time), max(time), sum(time) / len(time), sum(time), len(time)])

EXPORTER_DESCRIPTORS = [{
    'format': 'stat',
    'available': True,
    'exporter': Stat
}]
