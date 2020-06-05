Intel® Instrumentation and Tracing Technology (ITT) and Just-In-Time (JIT) API
==================================================================================

This ITT/JIT open source profiling API includes:

  - Instrumentation and Tracing Technology (ITT) API
  - Just-In-Time (JIT) Profiling API

The Instrumentation and Tracing Technology (ITT) API enables your application
to generate and control the collection of trace data during its execution 
across different Intel tools.

ITT API consists of two parts: a _static part_ and a _dynamic part_. The
_dynamic part_ is specific for a tool and distributed only with a particular
tool. The _static part_ is a common part shared between tools. Currently, the
static part of ITT API is distributed as a static library and released under
a BSD/GPLv2 dual license with every tool supporting ITT API.

### Build

To build the library:
 - On Windows, Linux and OSX: requires [cmake](https://cmake.org) to be set in `PATH`
 - Run `python buildall.py`
 - Windows: requires Visual Studio installed or requires [Ninja](https://github.com/ninja-build/ninja/releases) to be set in `PATH`

### Run

To load the library:
 - On Windows and Linux: Set environment variable `INTEL_LIBITTNOTIFY32`/`INTEL_LIBITTNOTIFY64` to the full path pointing to `libittnotify[32/64].[lib/a]`
 - On OSX: Set environment variable `DYLD_INSERT_LIBRARIES` to the full path to `libittnotify.dylib`

### License

All code in the repo is dual licensed under GPLv2 and 3-Clause BSD licenses
