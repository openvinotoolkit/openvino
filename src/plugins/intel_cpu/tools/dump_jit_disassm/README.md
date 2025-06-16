# Dump JIT code disassembler tool

This tool generates JIT dump disassembly (intel syntax) with `file:line_no1` annotations showing the call stack of the C++ jit source code which generates the instruction. By reading the generated disassembly together with the JIT source code it helps developer to understand and debug JIT code.

# Preparing

1. Install dependency

```bash
python3 -m pip install argparse
python3 -m pip install colorama
```

 2. Build CPU plugin with `-DENABLE_DEBUG_CAPS=ON -DCMAKE_BUILD_TYPE=Debug` and install it.

# Usage

Launch OpenVINO CPU plugin through your application with following environment variable set:

```bash
export ONEDNN_JIT_DUMP=1
export ONEDNN_VERBOSE=2
export OV_CPU_DEBUG_LOG=-
```

You can find logs sequences of `dump_debug_traces`/`register_jit_code`/`dump_jit_code` as shown below:

```bash
[ oneDNN ] dump_debug_traces: dnnl_traces_cpu_jit_avx512_core_amx_compute_zp_pbuff_t.121.txt
[ oneDNN ] register_jit_code: /home/dev/tingqian/openvino/src/plugins/intel_cpu/thirdparty/onednn/src/cpu/x64/jit_avx512_core_amx_conv_kernel.hpp, jit_avx512_core_amx_compute_zp_pbuff_t
[ oneDNN ] dump_jit_code: dnnl_dump_cpu_jit_avx512_core_amx_compute_zp_pbuff_t.121.bin
[ DEBUG ] graph.cpp:856 CreatePrimitives() LOADTIME_createPrimitive tl_unet/outD4/Conv2D_1 jit_avx512_amx_I8 [+ 13177.633/88862.552 ms]
onednn_verbose,create:cache_miss,convolution,jit:avx512_core_amx_int8,forward_inference,src_u8::blocked:acdb:f0 wei_s8:p:blocked:ABcd16b16a4b:f8:zpm1 bia_f32::blocked:a:f0 dst_f32::blocked:acdb:f0,attr-zero-points:src0:0:167 attr-post-ops:depthwise_scale_shift+eltwise_tanh+eltwise_linear:296.41:227+eltwise_round_half_to_even+eltwise_clip:-0:255+eltwise_linear:0.964648:-218.975+sum:1:0:f32+eltwise_linear:0.581796:104+eltwise_round_half_to_even+eltwise_clip:0:255+eltwise_linear:0.00601129:-0.625175 ,alg:convolution_direct,mb1_ic96oc3_ih128oh128kh3sh1dh0ph1_iw128ow128kw3sw1dw0pw1,0.685791
```

 - item `[ oneDNN ] dump_debug_traces` shows us the name of the file into which offsets & backtraces are dumpped.
 - item `[ oneDNN ] register_jit_code` shows us the corresponding jit source code.
 - item `[ oneDNN ] dump_jit_code` shows us the name of the file into which jit generated binary code is dumpped.
 - item `[ DEBUG ]` shows us for which layer are these dumps generated.
 - item `onednn_verbose,create` shows us the full description of the primitive generating the jit kernel and the dumps above.

If we want to explorer the JIT code dumpped here, use following command:

```bash
python ~/openvino/src/plugins/intel_cpu/tools/dump_jit_disassm/ dnnl_traces_cpu_jit_avx512_core_amx_compute_zp_pbuff_t.121.txt dnnl_dump_cpu_jit_avx512_core_amx_compute_zp_pbuff_t.121.bin
```

This tool will extract line number debug information using the well-known linux command `addr2line` and disassemble the JIT binary dump using another well-known command `objdump`, so make sure they are correctly installed in your system.

the final output looks like this:

```
0000000000000000 <.data>:
       0:       53                      push   rbx      jit_avx512_core_amx_1x1_conv_kernel.cpp:834
       1:       55                      push   rbp      jit_avx512_core_amx_1x1_conv_kernel.cpp:834
       2:       41 54                   push   r12      jit_avx512_core_amx_1x1_conv_kernel.cpp:834
       4:       41 55                   push   r13      jit_avx512_core_amx_1x1_conv_kernel.cpp:834
       6:       41 56                   push   r14      jit_avx512_core_amx_1x1_conv_kernel.cpp:834
       8:       41 57                   push   r15      jit_avx512_core_amx_1x1_conv_kernel.cpp:834
       a:       bd 00 04 00 00          mov    ebp,0x400
       f:       48 83 ec 08             sub    rsp,0x8
      13:       4c 8b bf 88 00 00 00    mov    r15,QWORD PTR [rdi+0x88] jit_uni_postops_injector.cpp:387 / jit_avx512_core_amx_1x1_conv_kernel.cpp:837
      1a:       4d 8b 37                mov    r14,QWORD PTR [r15]      jit_uni_postops_injector.cpp:389 / jit_avx512_core_amx_1x1_conv_kernel.cpp:837
      1d:       4c 89 34 24             mov    QWORD PTR [rsp],r14      jit_uni_postops_injector.cpp:387 / jit_avx512_core_amx_1x1_conv_kernel.cpp:837
```

# Tips

 - Please note that the line number showed is actually derived from return address of each function in call stack, thus it's the line of code right next to the JIT source that generated the instruction. User should focus on previous valid line of source code for exact mapping.

 - In VSCode, if the final output is displayed in TERMINAL, you can click the `file:line_no` while holdding `Ctrl` key to directly navigate to coresponding source code.

 - `llvm-addr2line` (`llvm-symbolizer`) may work significantly faster than default `addr2line`, consider using `addr2line` tool from LLVM toolchain if applicable. Customization is available using `--addr2line=<path-to-addr2line-tool>` flag.
