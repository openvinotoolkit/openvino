import os
import csv
import shutil
from sea_runtool import GraphCombiner

# Supported values are "csv" and "tsv"
FILE_EXTENSION = ".csv"

class Stat(GraphCombiner):
    def __init__(self, args, tree):
        GraphCombiner.__init__(self, args, tree)

    def get_targets(self):
        return [self.args.output + FILE_EXTENSION]

    def finish(self):
        GraphCombiner.finish(self)
        delim = ','
        if FILE_EXTENSION == ".tsv":
            delim = '\t'
        with open(self.get_targets()[-1], 'w') as f:
            writer = csv.writer(f, delimiter=delim)
            writer.writerow(["domain", "name", "min", "max", "avg", "total", "count"])
            for domain, data in self.per_domain.items():
                for task_name, task_data in data['tasks'].items():
                    time = task_data['time']
                    writer.writerow([domain, task_name, min(time), max(time), sum(time) / len(time), sum(time), len(time)])

    @staticmethod
    def join_traces(traces, output, args):  # FIXME: implement real joiner
        sorting = []
        for trace in traces:
            sorting.append((os.path.getsize(trace), trace))
        sorting.sort(key=lambda size_trace: size_trace[0], reverse=True)
        shutil.copyfile(sorting[0][1], output + ".tsv")
        return output + ".tsv"

EXPORTER_DESCRIPTORS = [{
    'format': 'stat',
    'available': True,
    'exporter': Stat
}]
