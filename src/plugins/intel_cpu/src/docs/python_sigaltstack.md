# Python API AMX environment initialization

Unlike SSE/AVX, AMX feature on Linux is not enabled by default for every process(due to excessively larger register space requirement on [XSAVE][1] area and thus longer context switching latency), user application needs to request permission for such dynamically enabled feature using [XSTATE system call][2].

oneDNN will [do such initialization][3] at the first time when some primitive calls `mayiuse` with amx related cpu_isa, for example:  `mayiuse(avx512_core_bf16_amx_bf16)`. This may happen in followng stages:
 - graph compilation stage: in stream's scheduling thread.
 - prepare parameter stage: also in stream's scheduling thread

This is fine for C/C++ OpenVINO application, but Python application using OpenVINO would fail to use AMX due to following known issue:

 [ Insufficient sigaltstack size used by CPython prevents extensions from using new ISA][4].

In brief, CPython sets it's [signal stack][5] too small in [Modules/faulthandler.c][6], so no enough space to store AMX in [XSAVE][1] area, causing [Linux XSTATE implementation][7] to return ENOSPC. This issue is probably not fixed until Python3.9.

The stream scheduling thread created by pthread_create() [does not inherit the creating thread's alternate signal stack][8], so it's big enough, but [Linux XSTATE][7] API requires all threads in current process to have big enough signal stack since the feature is enabled for whole process, and the CPython main thread doesn't satisfy this requirement due to above issue, so the fix has to be done in CPython thread rather than in stream scheduling thread.

We choose to fix it in constructor/destructor of `Engine` class, a member variable of type `std::shared_ptr<void>` holds a reference to a `CPUSpecialSetup` instance which manages signal stack replacement.

[1]: https://www.moritz.systems/blog/how-debuggers-work-getting-and-setting-x86-registers-part-2/

[2]: https://www.kernel.org/doc/html/latest/x86/xstate.html

[3]:https://github.com/oneapi-src/oneDNN/blob/master/src/cpu/x64/cpu_isa_traits.cpp#L333

[4]: https://bugs.python.org/issue46968

[5]: https://www.gnu.org/software/libc/manual/html_node/Signal-Stack.html

[6]: https://github.com/python/cpython/blob/main/Modules/faulthandler.c#L1359

[7]: https://elixir.bootlin.com/linux/v5.19-rc2/source/arch/x86/kernel/fpu/xstate.c#L1532

[8]: https://man7.org/linux/man-pages/man3/pthread_create.3.html