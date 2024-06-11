# OpenVINO NPUW Extension to the NPU plugin

Welcome to the source directory of the NPUW extension to the NPU
plugin! Here you will find all the information on how to build and run
inference via NPUW.

## Introduction

NPUW is an extension to the NPU plugin which introduces function
outlining and faster compilation for LLM models.

## Building

The extension is built as part of normal OpenVINO build procedure, as
all the necessities are already bundled in the source tree. An
important note so far is to disable CPPLINT as the code is not
aligned with the OpenVINO guidelines yet:

1. Clone this fork & the right branch:

   ```bash
   git clone --recursive --branch=preview/npuw --depth=1 https://github.com/openvinotoolkit/openvino_private.git
   ```
2. Now build the OpenVINO package:

   ```bash
   cd openvino
   mkdir build-x86_64 && cd build-x86_64
   cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_PLUGINS_XML=ON -DCMAKE_INSTALL_PREFIX=install -DENABLE_CPPLINT=OFF -DPython3_EXECUTABLE=path/to/python.exe -DCYTHON_EXECUTABLE=/path/to/cython.exe ..
   cmake --build . --config Release --target install --parallel
   ```

After you complete these steps, you should get the fully-functioning
OpenVINO+NPU(W) package in your
`/path/to/openvino/openvino/build-x86_64/install` directory.

NOTE: Be careful with `--parallel` option in CMake, based on the
experience it is better to limit the number of parallel jobs (e.g.,
`--parallel 8`) to avoid the build host freezing or running out of
memory.

## Running

Running the NPU with NPUW is easy, just keep using the `NPU`
as the device name in arguments to `compile_model()` / `benchmark_app`
and so on -- but additionally set the `"NPU_USE_NPUW" : "YES"` (Python)
or `ov::intel_npu::use_npuw(true)` (C++) property to them.

Currently it may not show anything special when ran "as-is". It
requires options to show some magic, see below.

## Options

NPUW extension is configured via options, that can be passed to
`compile_model()` as OpenVINO properties.
You may find the table with options/properties and their meanings
in [Option reference](#option-reference).

### Naming conventions

Almost all NPUW options start with `NPUW_` prefix in Python or `ov::intel_npu::npuw::`
namespace in C++. The remaining part is the option name.
All options work on a per model basis and are applied to that compiled model,
to which compilation routine they are passed. However, there are options, like
`NPUW_SUBMODEL_DEVICE` (Python) or `ov::intel_npu::npuw::submodel_device()` (C++),
that accepts values for specific subgraphs of the model. For example,
`"0:NPU,1:CPU"` is a possible value meaning that subgraph 0 should work on NPU
and subgraph 1 - on CPU. So, subgraph level fine-tuning is also available.

### Option reference

Here, for brevity, only text version of properties is preserved.
You can refer to src/plugins/intel_npu/src/al/include/npuw_private_properties.hpp
for full `ov::Property` descriptions.


| Option (Python)            | Possible values                                               | Default Value | Meaning                                                                                                                                    |
|----------------------------|---------------------------------------------------------------|---------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| `"NPU_USE_NPUW"`           | `"YES"`, `"NO"`                                               | `"NO"`        | Set this option to `YES` to utilize NPUW extension.                                                                                        |
| `"NPUW_DEVICES"`           | List of devices                                               | `"NPU,CPU"`   | Device list to try in order.                                                                                                               |
| `"NPUW_SUBMODEL_DEVICE"`   | List of "Subgraph idx: device"                                | `""`          | Force the specific subgraph to specific device. The device must be present in the `"NPUW_DEVICES"` list.                                   |
| `"NPUW_ONLINE_PIPELINE"`   | `"INIT"`, `"JUST"`, `"REP"`                                   | `"REP"`       | Specify which partitioning pipeline to run.                                                                                                |
| `"NPUW_ONLINE_AVOID"`      | E.g., `"Op:Select/NPU,P:RMSNorm/NPU"`                         | `""`          | Forbids operation(s) and/or predefined pattern(s) to compile and run on a specified device.                                                |
| `"NPUW_ONLINE_MIN_SIZE"`   | Integer > `10`                                                | `10`          | Lower boundary of partition graph size the plugin can generate. Used to control fusion term criteria in online partitioning.               |
| `"NPUW_ONLINE_DUMP_PLAN"`  |  `"path-to-file.xml"`                                         | `""`          | Dump online partitioning to the specified file. This partitioning can be reused via `NPUW_PLAN` property later.                            |
| `"NPUW_PLAN"`              | `"path-to-file.xml"`                                          | `""`          | Set plan file to use by offline partitioning.                                                                                              |
| `"NPUW_FOLD"`              | `"YES"`, `"NO"`                                               | `"NO"`        | Perform function call folding if there are repeating blocks in the graph.                                                                  |
| `"NPUW_CWAI"`              | `"YES"`, `"NO"`                                               | `"NO"`        | Cut-off weighs from repeating blocks, but don't do folding. Decompression cut-off may still happen. Conflicts with `"NPUW_FOLD"`           |
| `"NPUW_DCOFF_TYPE"`        | `"f16"`, `"i8"`                                               | `""`          | Promotional data type for weights decompression. Works only with function `"NPUW_FOLD"`ing.                                                |
| `"NPUW_DCOFF_SCALE"`       | `"YES"`, `"NO"`                                               | `"NO"`        | Include weights scaling into the decompression procedure (and exclude it from function bodies). Works only with function `"NPUW_FOLD"`ing. |
| `"NPUW_FUNCALL_FOR_ALL"`   | `"YES"`, `"NO"`                                               | `"NO"`        | Every subgraph will be turned into a function. Warning: May cause performance issues!                                                      |
| `"NPUW_PARALLEL_COMPILE"`  | `"YES"`, `"NO"`                                               | `"NO"`        | Employ parallel subgraph compilation. Disabled by default due to instaibilities.                                                           |
| `"NPUW_FUNCALL_ASYNC"`     | `"YES"`, `"NO"`                                               | `"NO"`        | Pipeline execution of functions (repeating blocks) and their prologues (e.g., where weights decompression may happen).                     |
| `"NPUW_ACC_CHECK"`         | `"YES"`, `"NO"`                                               | `"NO"`        | Enable accuracy check for inference to make infer requests tolerant to accuracy fails                                                      |
| `"NPUW_ACC_THRESH" : 0.01` | Double floating-point value from `0.0` to `1.0`.              | `0.01`        | Threshold for accuracy validators, to indicate that metric returns successfull comparison.                                                 |
| `"NPUW_ACC_DEVICE"`        | Device name                                                   | `""`          | Reference device, giving accurate results for given model(s).                                                                              |
| `"NPUW_DUMP_FULL"`         | `"YES"`, `"NO"`                                               | `"NO"`        | Dump the whole model in its original form (as plugin gets it, before any partitioning is done).                                            |
| `"NPUW_DUMP_SUBS"`         | Comma-separated list of subgraph indices  or `"YES"` for all. | `""`          | Dump the specified subgraph(s) in OpenVINO IR form in the current directory.                                                               |
| `"NPUW_DUMP_SUBS_ON_FAIL"` | Comma-separated list of subgraph indices  or `"YES"` for all. | `""`          | Dump the specified subgraph(s) on disk if a compilation failure happens.                                                                   |
| `"NPUW_DUMP_IO"`           | Comma-separated list of subgraph indices  or `"YES"` for all. | `""`          | Dump input & output tensors for subgraph(s).                                                                                               |
| `"NPUW_DUMP_IO_ITERS"`     | `"YES"`, `"NO"`                                               | `"NO"`        | Dump input & output tensors for subgraph(s) for every iteration. WARNING: may exhaust the disk space quickly.                              |

NOTE: `"YES/NO"`values above are to be used in Python.
For C++ namespace please use `true/false` values
within the `ov::intel_npu::npuw` namespace.

## Logging

There is a possibility to turn on logging via environment variables:
`OPENVINO_NPUW_LOG` and `OPENVINO_NPUW_LOG_LEVEL`. They are both
absolutely the same.

They can accept following values: `ERROR`, `WARNING`, `INFO`,
`DEBUG`, `YES`. Every next level in the list includes the previous
one. `YES` equals to `DEBUG`.

For example: `set OPENVINO_NPUW_LOG=INFO` enables detailed logging
of processes underlying NPUW.

## Contacts

* Dmitry Matveev: [dmitry.matveev@intel.com](dmitry.matveev@intel.com)

[jira:upstream]: https://jira.devtools.intel.com/browse/EISW-112305
