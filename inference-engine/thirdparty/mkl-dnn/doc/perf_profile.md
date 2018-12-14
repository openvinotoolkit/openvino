Performance profiling {#perf_profile}
=====================================

It is often useful to collect information about how much of an application run
time is spent executing Intel(R) MKL-DNN primitives and which of those take
the most time. One of the popular methods to do this is to use profilers like
Linux\* perf or Intel(R) VTune(tm) Amplifier. Currently, Intel MKL-DNN has very
limited support for these tools since it does not annotate code generated at
run-time and thus the profiles cannot properly attribute it. However, Intel
MKL-DNN implements another feature called _verbose mode_ that allows tracing
execution of Intel MKL-DNN primitives and collection of basic statistics like
execution time and primitive parameters.

## Verbose mode

To enable Intel MKL-DNN verbose mode, set `MKLDNN_VERBOSE` environment variable
to `1` (to dump only execution time) or `2` (to dump both execution and
creation time). For example:

```
    $ export MKLDNN_VERBOSE=1
    $ ./benchdnn --conv ic16ih7oc16oh7kh5ph2n"wip"
```

This will produce the following output (the line break was added to fit into
the page width):

```
    mkldnn_verbose,exec,reorder,simple:any,undef,in:f32_nchw out:f32_nChw8c,num:1,2x16x7x7,0.484863
    mkldnn_verbose,exec,reorder,simple:any,undef,in:f32_goihw out:f32_gOIhw8i8o,num:1,1x16x16x5x5,0.494141
    mkldnn_verbose,exec,reorder,simple:any,undef,in:f32_nchw out:f32_nChw8c,num:1,2x16x7x7,0.478027
    mkldnn_verbose,exec,reorder,simple:any,undef,in:f32_x out:f32_x,num:1,16,0.219971
    mkldnn_verbose,exec,convolution,jit:avx2,forward_inference,fsrc:nChw8c fwei:gOIhw8i8o fbia:x \
        fdst:nChw8c,alg:convolution_direct,mb2_g1ic16oc16_ih7oh7kh5sh1dh0ph2_iw7ow7kw5sw1dw0pw2,0.0170898
    mkldnn_verbose,exec,reorder,simple:any,undef,in:f32_nChw8c out:f32_nchw,num:1,2x16x7x7,0.488037
    mkldnn_verbose,exec,reorder,simple:any,undef,in:f32_nChw8c out:f32_nchw,num:1,2x16x7x7,0.00512695
    0:PASSED __REPRO: ic16ih7oc16oh7kh5ph2nwip
    tests:1 passed:1 skipped:0 mistrusted:0 unimplemented:0 failed:0
```

Each line with verbose information is formatted as a comma-separated list
containing:
- `mkldnn_verbose`
- `stage`, e.g. `create` or `exec`
- `primitive-kind`, e.g. `convolution`, `reorder`, `sum`, ...
- primitive implementation name
- propagation-kind, e.g. `forward_training`
- input/output data info, e.g. data type and data format
- auxiliary information, e.g. algorithm or number of input
- problem description
    - for convolution the problem description is dumped in benchdnn friendly format
    - for reorder, sum, and concat problem description is simply logical dims
    - for other primitives the problem description is similar to convolution one
- execution time in milliseconds

To get more information about verbose report format please refer to the
`verbose_templ()` function in the
[src/common/verbose.hpp](https://github.com/intel/mkl-dnn/blob/master/src/common/verbose.hpp)
file.

---
**NOTE**
The format is subject to change

---


---
**WARNING**
Verbose mode has non-negligible performance impact especially if the output
rate is high.

---

## Intel(R) VTune(TM) profiling

To collect performance data of JIT-kernels set `VTUNEROOT` environment variable
to path to VTune before building of Intel MKL-DNN. For example:

```
    $ mkdir -p build && cd build && cmake -DVTUNEROOT=/path/to/vtune .. && make
```

## Dump JIT-kernels
To dump JIT-kernels set MKLDNN_JIT_DUMP environment variable to `1`. For example:

```
    $ export MKLDNN_JIT_DUMP=1
    $ ./simple-net-c
```

This will produce the following output files:
    mkldnn_dump_jit_avx2_conv_fwd_kernel_f32.0.bin
    mkldnn_dump_jit_uni_lrn_fwd_kernel_f32.2.bin
    mkldnn_dump_jit_uni_lrn_fwd_kernel_f32.3.bin
    mkldnn_dump_jit_uni_lrn_fwd_kernel_f32.4.bin
    mkldnn_dump_jit_uni_pool_kernel_f32.5.bin
    mkldnn_dump_jit_uni_relu_kernel_f32.1.bin
    
To open these files any disassembler can be used. For example:

```
    $ xed -ir mkldnn_dump_jit_avx2_conv_fwd_kernel_f32.0.bin
    XDIS 0: PUSH      BASE       53                       push ebx
    XDIS 1: PUSH      BASE       55                       push ebp
    XDIS 2: BINARY    BASE       41                       inc ecx
    XDIS 3: PUSH      BASE       54                       push esp
    XDIS 4: BINARY    BASE       41                       inc ecx
    XDIS 5: PUSH      BASE       55                       push ebp
    XDIS 6: BINARY    BASE       41                       inc ecx
    XDIS 7: PUSH      BASE       56                       push esi
    XDIS 8: BINARY    BASE       41                       inc ecx
    XDIS 9: PUSH      BASE       57                       push edi
    XDIS a: BINARY    BASE       48                       dec eax
    XDIS b: DATAXFER  BASE       8B07                     mov eax, dword ptr [edi]
    XDIS d: BINARY    BASE       48                       dec eax
    XDIS e: DATAXFER  BASE       8B7708                   mov esi, dword ptr [edi+0x8]
    XDIS 11: BINARY    BASE       48                       dec eax
    XDIS 12: DATAXFER  BASE       8B5710                   mov edx, dword ptr [edi+0x10]
    XDIS 15: BINARY    BASE       48                       dec eax
    XDIS 16: DATAXFER  BASE       8B5F18                   mov ebx, dword ptr [edi+0x18]
    XDIS 19: BINARY    BASE       48                       dec eax
    XDIS 1a: DATAXFER  BASE       8B4F40                   mov ecx, dword ptr [edi+0x40]
    XDIS 1d: BINARY    BASE       44                       inc esp
    XDIS 1e: DATAXFER  BASE       8B6F70                   mov ebp, dword ptr [edi+0x70]
    ...
```

[Legal information](@ref legal_information)
