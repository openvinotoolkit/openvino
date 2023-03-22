# Intel Software Development Emulator

Intel SDE can be used for emulating CPU architecture, checking for AVX/SSE transitions, bad pointers and data misalignment, etc.

It also supports debugging within emulation.

In general, the tool can be used for all kinds of troubleshooting activities except performance analysis.

See [Documentation](https://www.intel.com/content/www/us/en/developer/articles/tool/software-development-emulator.html) for more information

## Usage examples:

- Emulating Sapphire Rapids CPU for _benchmark_app_ together with blob dumping, for example to debug some accuracy issue:

```sh
OV_CPU_BLOB_DUMP_FORMAT=TEXT OV_CPU_BLOB_DUMP_NODE_TYPE=Convolution \
/path/to/sde -spr -- ./benchmark_app --niter 1 --nstreams 1 -m path/to/model.xml
```

- Running _cpuFuncTests_ on some old architecture, for example Sandy Bridge:

```sh
/path/to/sde -snd -- ./cpuFuncTests
```

- Count AVX/SSE transitions for the current host:

```sh
/path/to/sde -ast -- ./benchmark_app -m path/to/model.xml
```

> **NOTE**: The best way to check for AVX/SSE transitions is to run within Alder Lake emulation:

```sh
/path/to/sde -adl -- ./benchmark_app -m path/to/model.xml
```

## See also

 * [OpenVINO™ README](../../../../README.md)
 * [OpenVINO Core Components](../../../README.md)
 * [OpenVINO Plugins](../../README.md)
 * [OpenVINO CPU Plugin](../README.md)
 * [Developer documentation](../../../../docs/dev/index.md)
