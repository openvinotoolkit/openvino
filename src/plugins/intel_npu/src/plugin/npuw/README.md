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
and so on -- but set `OV_USE_NPUW=YES` in your environment variables
first.

Currently it may not show anything special when ran "as-is". It
requires options to show some magic, see below.

## Options

Currently NPUW extension is configured via environment variables. It is
not very great in terms of usability, but in fact allows us to
configure the NPUW behavior in a non-invasive way, especially when it
is hidden behind a number of abstraction layers in the application or
notebook.

### Naming conventions

Currently all NPUW environment variables start with prefix
`OPENVINO_NPUW_`. The following part is the option name. Some options
may apply only globally, but some options may apply to specific models
(in multi-model applications) or even to specific model
subgraphs. Such models will be highlighed in the below configuration
table.
- If an option `$OPTIONNAME` can be applied to a particular model,
  then it can be set via `OPENVINO_NPUW_$OPTIONNAME_$MODELNAME`, where
  `$MODELNAME` is defined in the OpenVINO model IR (or set via
  `set_friendly_name()` in runtime), e.g.:

  ```bash
  $ head -n2 decoder_model.xml
  <?xml version="1.0"?>
  <net name="torch_jit_dec_int4" version="11">
  $ head -n2 decoder_with_past_model.xml
  <?xml version="1.0"?>
  <net name="Model0_dec_int4" version="11">
  ```
  Here model names are `torch_jit_dec_int4` and `Model0_dec_int4`,
  respectively.
- If an option `$OPTIONNAME` can be applied to a particular sub-model,
  then it can be set via `OPENVINO_NPUW_$OPTIONNAME_$MODELNAME_$IDX`,
  where `$MODELNAME` is the same as defined before, and `$IDX` is a
  numeric *zero-based* index of a submodel.

Examples:

| Option                                     | Meaning                                                                                     |
|--------------------------------------------|---------------------------------------------------------------------------------------------|
| `set OPENVINO_NPUW_LOG=INFO`               | A global option, set the current log level to `INFO`                                        |
| `set OPENVINO_NPUW_DEVICES_Model0=GPU,CPU` | A model-specific option, set device list to try for `Model0` to `GPU,CPU`                   |
| `set OPENVINO_NPUW_DEVICE_Model0_3=CPU`    | A sub-model-specific option, force the device to `CPU` for the fourth subgraph of `Model0`. |


### Option reference

In the table below, for brevity, the following codes are used to
denote the supported scope for an option:

* `G` - this is a global option (applies to entire plugin, all models
  and submodels)
* `M` - this option may be specified per-model.
* `S` - this option may be specifed per-sub-model.

Combinations are also possible.

| Option                          | Scope       | Possible values                    | Default Value | Meaning                                                                                                                                                                 |
|---------------------------------|-------------|------------------------------------|---------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `OPENVINO_NPUW_CWAI`            | `G`,`M`     | `YES`                              | (no)          | Cut-off weighs from repeating blocks, but don't do folding. Decompression cut-off may still happen. Conflicts with `FOLD`.                                              |
| `OPENVINO_NPUW_DCOFF_SCALE`     | `G`,`M`     | `YES`                              | (no)          | Include weights scaling into the decompression procedure (and exclude it from function bodies). Works only with function `FOLD`ing.                                     |
| `OPENVINO_NPUW_DCOFF`           | `G`,`M`     | `i8`, `f16`                        | (no)          | Promotional data type for weights decompression. Works only with function `FOLD`ing.                                                                                    |
| `OPENVINO_NPUW_DEVICES`         | `G`, `M`    | Comma-separated list               | `NPU,CPU`     | Device list to try in order. E.g., `NPU,GPU,CPU`.                                                                                                                       |
| `OPENVINO_NPUW_DEVICE`          | `S`         | OpenVINO device name               | (none)        | Force the specific subgraph to specific device. The device must be present in the `NPUW_DEVICES` list.                                                                  |
| `OPENVINO_NPUW_DUMP_IO`         | `G`,`M`,`S` | `YES`                              | (no)          | Dump input & output tensors for subgraph(s).                                                                                                                            |
| `OPENVINO_NPUW_DUMP_IO_ITERS`   | `G`,`M`,`S` | `YES`                              | (no)          | Dump input & output tensors for subgraph(s) for every iteration. Warning: may exhaust the disk space quickly.                                                           |
| `OPENVINO_NPUW_DUMP_ON_FAIL`    | `G`,`M`,`S` | `YES`                              | (no)          | Dump a submodel on disk if a compilation failure happens.                                                                                                               |
| `OPENVINO_NPUW_DUMP_SUB`        | `G`,`M`,`S` | `YES`                              | (no)          | Dump the specified subgraph(s) in OpenVINO IR form in the current directory.                                                                                            |
| `OPENVINO_NPUW_DUMP_PLAN`       | `G`,`M`     | `YES`                              | (no)          | Dump online partitioning in the current directory. The partitioning can be used as an offline one later (see `OPENVINO_NPUW_PLAN`).                                     |
| `OPENVINO_NPUW_DUMP`            | `G`,`M`     | `YES`                              | (no)          | Dump the whole model in its original form (as plugin gets it, before any partitioning is done).                                                                         |
| `OPENVINO_NPUW_FOLD`            | `G`,`M`     | `YES`                              | (no)          | Perform function call folding if there's repeating blocks in the graph.                                                                                                 |
| `OPENVINO_NPUW_LOG_LEVEL`       | `G`         | (See log levels)                   | (no)          | Set the log verbosity level. Same as `OPENVINO_NPUW_LOG`.                                                                                                               |
| `OPENVINO_NPUW_LOG`             | `G`         | (See log levels)                   | (no)          | Set the log verbosity level.                                                                                                                                            |
| `OPENVINO_NPUW_PARC_GRAPH_SIZE` | `G`,`M`     | Integer >= 10                      | 10            | Lower boundary of partition graph size the plugin can generate. Used to control fusion term criteria. Works with `USE_PARC` only.                                       |
| `OPENVINO_NPUW_PARC_PIPELINE`   | `G`,`M`     | `INIT`,`JUST`                      | `JUST`        | Specify which partitioning pipeline to run.                                                                                                                             |
| `OPENVINO_NPUW_AVOID`           | `G`,`M`     | e.g. `Op:Select/NPU,P:RMSNorm/NPU` | (none)        | Forbids operation(s) and/or predefined pattern(s) compiling and running on a specified device. Only compatible with online partitioning (see `OPENVINO_NPUW_USE_PARC`). |
| `OPENVINO_NPUW_PARC`            | `G`,`M`     | `YES`                              | (no)          | Employ parallel subgraph compilation. Disabled by default due to instaibilities.                                                                                        |
| `OPENVINO_NPUW_PLAN`            | `G`,`M`     | Path to .xml file                  | (no)          | Pre-defined partitioning plan file to use.                                                                                                                              |
| `OPENVINO_NPUW_USE_PARC`        | `G`,`M`     | `YES`                              | (no)          | Use online partitioning by default (`_PLAN` option is ignored in this case).                                                                                            |
| `OPENVINO_NPUW_FUNCALL_ASYNC`   | `G`,`M`     | `YES`                              | (no)          | Pipeline execution of functions (repeating blocks) and their prologues (e.g., where weights decompression may happen).                                                  |

The following log levels are available: `ERROR`, `WARNING`, `INFO`,
`DEBUG`, `YES`. Every next level in the list includes the previous
one. `YES` equals to `DEBUG`.

## Contacts

* Dmitry Matveev: [dmitry.matveev@intel.com](dmitry.matveev@intel.com)

[jira:upstream]: https://jira.devtools.intel.com/browse/EISW-112305
