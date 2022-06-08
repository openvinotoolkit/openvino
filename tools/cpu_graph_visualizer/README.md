# CPU Graph visualizer

A python tools based on openvino API 2.0 to visualize CPU execution graph (runtime model) as graphviz rendered svg embedding in single HTML file, to be open and viewed inside web brower.

```bash
$ python ~/openvino/tools/cpu_graph_visualizer/visualizer.py -h
usage: CPU graph visualizer [-h] [-p] [--raw] [--bf16] model

positional arguments:
  model       Model file path

optional arguments:
  -h, --help  show this help message and exit
  -p, --perf  do profiling
  --raw       also dump raw ngraph model
  --bf16      enable inference precision with bf16
```