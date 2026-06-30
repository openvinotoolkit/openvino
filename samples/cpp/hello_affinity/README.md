# Hello Affinity C++ Sample

This sample demonstrates how to apply per-operation affinity hints to a model and print the resulting affinity summary.

Models with static or dynamic inputs are supported if the selected device can compile them.

## Usage

```sh
hello_affinity -m <path_to_model> [-d <device_name>] [-affinity <affinity|path_to_affinity_json>] \
    [--fallback-device <device>] [-hint <performance_hint>] [-shape <shapes>] [-data_shape <shapes>] \
    [-niter <integer>] [-no_warmup]
```

The optional `-affinity` argument accepts either a single device name such as `CPU` or `GPU`, or a JSON file with
`{node_name: device_name}` mappings. If it is omitted, the selected device plugin assigns operations.
`-affinity` is supported only with `-d HETERO:<devices>`.

When a virtual device such as `HETERO:CPU,GPU` is used, the sample validates affinity values against the hardware
devices from the device list and applies the same fallback behavior as `benchmark_app` for partially mapped JSON files.

The sample reads the model, optionally applies affinity settings and prints the resulting assignment summary, compiles
the model, and runs synchronous inference with generated input tensors.

Use `-niter <integer>` or `--niter <integer>` to run multiple synchronous inference iterations with the same generated
input tensors. The default is `1`.

By default, the sample runs one warm-up inference before measuring `-niter` iterations, matching `benchmark_app`. Use
`-no_warmup` or `--no_warmup` to skip it.

Use `-hint <performance_hint>` or `--hint <performance_hint>` to set the device performance mode. Supported values are
`latency`, `throughput` (`tput`), and `none`. If it is omitted, the sample does not set a performance hint and lets
the selected device choose its default mode.

Use `--fallback-device <device>` together with `-affinity` to fill all operations not listed in the JSON file with an
explicit device before compilation.

Use `-shape <shapes>` and `-data_shape <shapes>` with the same shape string format as `benchmark_app` when
reproducing a benchmark command. `-shape` reshapes the model before affinity assignment; `-data_shape` provides
concrete dynamic input shapes for compilation and generated inference tensors.
