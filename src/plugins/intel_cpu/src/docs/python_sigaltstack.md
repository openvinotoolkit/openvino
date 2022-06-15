# python API AMX enviroment initialization

Unlike SSE/AVX, AMX feature on Linux is not enabled by default for every process(due to excessively larger space requirement on [XSAVE][1] area), user application needs to request permission for such dynamically enabled feature using [XSTATE API][2].

oneDNN will [do such initialization][3] on the first time some primitive is asking for it by passing amx related cpu_isa to `mayiuse` function, for example:  `mayiuse(avx512_core_bf16_amx_bf16)`. This may happens in any of:
 - graph compilation stage: in stream's scheduling thread.
 - prepare parameter stage: also in stream's scheduling thread

This is generally fine for C/C++ enviroment, but Python API would fail due to following known issue: [ Insufficient sigaltstack size used by CPython prevents extensions from using new ISA][4].

The issue is due to that CPython sets it's [signal stack][5] too small in [Modules/faulthandler.c][6], to allow enough space to store [XSAVE][1] area for AMX, causing [Linux XSTATE implementation][7] return ENOSPC error number. This issue if fixed for Python3.9, but not for lower version of python release.

The stream schedualing thread created by pthread_create() [does not inherit the creating thread's alternate signal stack][8], so it's signal stack has big enough size, but [Linux XSTATE][7] API requires all threads of current process to have big enough signal stack since the feature is enabled for whole process, and the Python API runnning inside CPython main thread dosn't satisfy this requirement due to above issue, so the fix has to be done in CPython thread rather than in stream scheduling thread.

We choose to do it inside Engine class's constructor/destructor in the plugin.cpp for now, `std::shared_ptr<void> specialSetup` member varible holds a reference to `CPUSpecialSetup` which manages signal stack init in it's constructor & destructor.

[1]: https://www.moritz.systems/blog/how-debuggers-work-getting-and-setting-x86-registers-part-2/

[2]: https://www.kernel.org/doc/html/latest/x86/xstate.html

[3]:https://github.com/oneapi-src/oneDNN/blob/master/src/cpu/x64/cpu_isa_traits.cpp#L333

[4]: https://bugs.python.org/issue46968

[5]: https://www.gnu.org/software/libc/manual/html_node/Signal-Stack.html

[6]: https://github.com/python/cpython/blob/main/Modules/faulthandler.c#L1359

[7]: https://elixir.bootlin.com/linux/v5.19-rc2/source/arch/x86/kernel/fpu/xstate.c#L1532

[8]: https://man7.org/linux/man-pages/man3/pthread_create.3.html