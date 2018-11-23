Here's the place for your question, suggestion, a feature request or brief
description of the problem. If you are submitting a defect report please fill
all the sections below. For everything else feel free to remove everything
below the line.

-----------------------------------------------------------------------------

### Environment
Intel MKL-DNN includes hardware-specific optimizations and may behave
differently on depending on the compiler and build environment. Include
the following information to help reproduce the issue:
* CPU make and model (try `lscpu`; if your `lscpu` does not list CPU flags,
  try running `cat /proc/cpuinfo | grep flags | sort -u`)
* OS version (`uname -a`)
* Compiler version (`gcc --version`)
* MKLROOT value (`echo MKLROOT=$MKLROOT`)
* CMake version (`cmake --version`)
* CMake output log
* git hash (`git log -1 --format=%H`)

### Steps to reproduce
Please check that the issue is reproducible with the latest revision on
master. Include all the steps to reproduce the issue. A short C/C++ program
or modified unit tests demonstrating the issue will greatly help
with the investigation.

### Actual behavior
Describe the behavior you see.

### Expected behavior
Describe the behavior you expect.
