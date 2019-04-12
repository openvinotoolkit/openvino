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

import openvino.tools.benchmark as benchmark

if __name__ == '__main__':
    config = benchmark.CommandLineReader.read()
    result = benchmark.Benchmark(config).run()
    print("{0}: {1:.4} ms".format(config.model, result.latency * 1000.0))