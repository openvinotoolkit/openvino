# Intel(R) Math Kernel Library for Deep Neural Networks (Intel(R) MKL-DNN)
![v0.17 beta](https://img.shields.io/badge/v0.17-beta-orange.svg)

Intel(R) Math Kernel Library for Deep Neural Networks (Intel(R) MKL-DNN) is
an open source performance library for deep learning applications. The library
accelerates deep learning applications and framework on Intel(R) architecture. 
Intel(R) MKL-DNN contains vectorized and threaded building blocks which you can
use to implement deep neural networks (DNN) with C and C++ interfaces.

DNN functionality optimized for Intel architecture is also included in 
[Intel(R) Math Kernel Library (Intel(R) MKL)](https://software.intel.com/en-us/mkl/features/deep-neural-networks).
API in this implementation is not compatible with Intel MKL-DNN and does not
include certain new and experimental features.

This release contains performance critical functions that improve performance of
of the following deep learning topologies and variations of these.

| Application                               | Example topology
|:---                                       |:---
| Image recognition                         | AlexNet, VGG, GoogleNet, ResNet, MobileNet
| Image segmenation                         | FCN, SegNet, MaskRCNN, U-Net
| Volumetric segmentation                   | 3D-Unet
| Object detection                          | SSD, Faster R-CNN, Yolo
| Neural Machine Translation (experimental) | GNMT
| Speech Recognition (experimental)         | DeepSpeech
| Adversarial Networks                      | DCGAN, 3DGAN
| Reinforcement Learning                    | A3C
| Text-to-Speech                            | WaveNet

Intel MKL-DNN is used in the following software products:
* [Caffe\* Optimized for Intel Architecture](https://github.com/intel/caffe)
* [Chainer\*](https://chainer.org)
* [DeepBench](https://github.com/baidu-research/DeepBench)
* [PaddlePaddle\*](http://www.paddlepaddle.org)
* [Tensorflow\*](https://www.tensorflow.org)
* [Microsoft\* Cognitive Toolkit (CNTK)](https://docs.microsoft.com/en-us/cognitive-toolkit)
* [Apache\* MXNet](https://mxnet.apache.org)
* [OpenVINO(TM) toolkit](https://01.org/openvinotoolkit)
* [Intel(R) Nervana(TM) Graph](https://github.com/NervanaSystems/ngraph)
* [Menoh\*](https://github.com/pfnet-research/menoh)
* [DeepLearning4J\*](https://deeplearning4j.org)
* [BigDL](https://github.com/intel-analytics/BigDL)

## License
Intel MKL-DNN is licensed under
[Apache License Version 2.0](http://www.apache.org/licenses/LICENSE-2.0). This
software includes the following third party components:
* [Xbyak](https://github.com/herumi/xbyak) distributed under [3-clause BSD licence](src/cpu/xbyak/COPYRIGHT)
* [gtest](https://github.com/google/googletest) distributed under [3-clause BSD license](tests/gtests/gtest/LICENSE)

## Documentation
* [Introduction](https://intel.github.io/mkl-dnn) explains programming model
and basic concepts
* [Reference manual](https://intel.github.io/mkl-dnn/modules.html) provides
detailed functionality description
* [Examples](https://github.com/intel/mkl-dnn/tree/master/examples) 
demonstrate use of C and C++ APIs in simple topologies
* [Tutorial](https://software.intel.com/en-us/articles/intel-mkl-dnn-part-1-library-overview-and-installation) 
provides step by step installation instructions and an example walkthrough

## Support
Please submit your questions, feature requests and bug reports on
[GitHub issues](https://github.com/intel/mkl-dnn/issues) page.

**WARNING** The following functionality has preview status and might change
without prior notification in future releases:
* Convolutions with `s16` data type in source, weights or destination
* Convolutions and auxiliary primitives for 3D spatial data
* RNN, LSTM and GRU primitives
* Intel Threading Building Blocks (Intel TBB\*) support

## How to Contribute
We welcome community contributions to Intel MKL-DNN. If you have an idea how to improve the library:

* Share your proposal via
 [GitHub issues](https://github.com/intel/mkl-dnn/issues).
* Ensure you can build the product and run all the examples with your patch
* In the case of a larger feature, create a test
* Submit a [pull request](https://github.com/intel/mkl-dnn/pulls)

We will review your contribution and, if any additional fixes or modifications
are necessary, may provide feedback to guide you. When accepted, your pull
request will be merged the repository.

## System Requirements
Intel MKL-DNN supports Intel(R) 64 architecture and compatible architectures.
The library is optimized for the systems based on
* Intel Atom(R) processor with Intel(R) SSE4.1 support
* 4th, 5th, 6th, 7th and 8th generation Intel(R) Core processor
* Intel(R) Xeon(R) processor E5 v3 family (formerly Haswell)
* Intel Xeon processor E5 v4 family (formerly Broadwell)
* Intel Xeon Platinum processor family (formerly Skylake)
* Intel(R) Xeon Phi(TM) processor x200 product family (formerly Knights Landing)
* Intel Xeon Phi processor x205 product family (formerly Knights Mill)

and compatible processors.

The software dependencies are:
* [Cmake](https://cmake.org/download/) 2.8.0 or later
* [Doxygen](http://www.stack.nl/~dimitri/doxygen/download.html#srcbin) 1.8.5 or later
* C++ compiler with C++11 standard support
* Optional dependencies:
  * GNU OpenMP\*, LLVM OpenMP\*, or Intel OpenMP
  * Threading Building Blocks (TBB)
  * Intel MKL or Intel MKL small libraries

> **Note**
> Building Intel MKL-DNN with optinal dependencies may introduce additional
> runtime dependencies for the library. Please refer to corresponding 
> software system requirements for details.

The software was validated on RedHat\* Enterprise Linux 7 with
* GNU\* Compiler Collection 4.8, 5.4, 6.1, 7.2 and 8.1
* Clang\* 3.8.0
* [Intel(R) C/C++ Compiler](https://software.intel.com/en-us/intel-parallel-studio-xe)
  17.0, 18.0 and 19.0

on Windows Server\* 2012 R2 with
* Microsoft\* Visual C++ 14.0 (Visual Studio 2015)
* [Intel(R) C/C++ Compiler](https://software.intel.com/en-us/intel-parallel-studio-xe)
  17.0 and 19.0

on macOS\* 10.13 (High Sierra) with
* Apple LLVM version 9.2 (XCode 9.2)
* [Intel C/C++ Compiler](https://software.intel.com/en-us/intel-parallel-studio-xe)
  18.0 and 19.0

The implementation uses OpenMP\* 4.0 SIMD extensions. We recommend using
Intel(R) Compiler for the best performance results.

## Installation

Download [Intel MKL-DNN source code](https://github.com/intel/mkl-dnn/archive/master.zip)
or clone the repository to your system

```
	git clone https://github.com/intel/mkl-dnn.git
```

Ensure that all software dependencies are in place and have at least minimal
supported version.

Intel MKL-DNN can take advantage of optimized
matrix-matrix multiplication (GEMM) function from Intel MKL. The dynamic
library with this functionality is included in the repository. If you choose 
to build Intel MKL-DNN with the binary dependency download Intel MKL small
libraries using provided script

###### Linux/macOS
```
	cd scripts && ./prepare_mkl.sh && cd ..
```

###### Windows
```
	cd scripts && call prepare_mkl.bat && cd ..
```

or manually from [GitHub release section](https://github.com/intel/mkl-dnn/releases)
and unpack it to the `external` directory in the repository root. Intel MKL-DNN
can also be built with full Intel MKL, if the latter is installed on the system.
You might need to set `MKLROOT` environment variable to the path where full
Intel MKL is installed to help cmake locate the library.

You can choose to build Intel MKL-DNN without binary dependency. The resulting
version will be fully functional, however performance of convolutions relying
on GEMM-based algorithm, inner product, and mkldnn_?gemm functionality may be
suboptimal.

> **Note**
>
> Using Intel MKL small libraries currently work for Intel MKL-DNN built with
> OpenMP\* only. Building with Intel TBB requires either full Intel MKL library
> or standalone build.
>
> Using Intel MKL or Intel MKL small libraries will introduce additional
> runtime dependencies. Please refer to Intel MKL 
> [system requirements](https://software.intel.com/en-us/articles/intel-math-kernel-library-intel-mkl-2019-system-requirements)
> for additional information.

Intel MKL-DNN uses a CMake-based build system

```
	mkdir -p build && cd build && cmake $CMAKE_OPTIONS .. && make
```

Here `$CMAKE_OPTIONS` are options to control the build. Along with the standard
cmake options such as `CMAKE_INSTALL_PREFIX` or `CMAKE_BUILD_TYPE`,
user can also pass Intel MKL-DNN specific ones:

|Option                 | Possible Values (defaults in bold)   | Description
|:---                   |:---                                  | :---
|MKLDNN_LIBRARY_TYPE    | **SHARED**, STATIC                   | Defines resulting library type
|MKLDNN_THREADING       | **OMP**, OMP:INTEL, OMP:COMP, TBB    | Defines threading type
|MKLDNN_USE_MKL         | **DEF**, NONE, ML, FULL, FULL:STATIC | Defines binary dependency on Intel MKL
|WITH_EXAMPLE           | **ON**, OFF                          | Controls building examples
|WITH_TEST              | **ON**, OFF                          | Controls building tests
|ARCH_OPT_FLAGS (\*)    | *compiler flags*                     | Specifies compiler optimization flags
|VTUNEROOT              | *path*                               | Enables integration with Intel(R) Vtune(tm) Amplifier

Please check [cmake/options.cmake](cmake/options.cmake) for more options
and details.

> (\*) **WARNING**
>
> By default Intel MKL-DNN is built specifically for the processor type of the
> compiling machine (e.g. `-march=native` in case of GCC). While this option
> gives better performance, the resulting library can only be run on systems
> that are instruction-set compatible with the compiling machine.
>
> Hence if Intel MKL-DNN is to be shipped to other platforms (e.g. built by
> Linux distribution maintainers) consider setting ARCH_OPT_FLAGS to "".

Intel MKL-DNN includes unit tests implemented using the googletest framework. To validate your build, run:

```
	make test
```

Documentation is provided inline and can be generated in HTML format with Doxygen:

```
	make doc
```

Documentation will reside in `build/reference/html` folder.

Finally,
```
	make install
```
will place the  header files, libraries and documentation in `/usr/local`. To change
the installation path, use the option `-DCMAKE_INSTALL_PREFIX=<prefix>` when invoking CMake.

## Linking your application

Intel MKL-DNN includes several header files providing C and C++ APIs for
the functionality and one or several dynamic libraries depending on how
Intel MKL-DNN was built. The minimal installation:

|File                   | Description
|:---                   |:---
|include/mkldnn.h       | C header
|include/mkldnn.hpp     | C++ header
|include/mkldnn_types.h | auxiliary C header
|lib/libmkldnn.so       | Intel MKL-DNN dynamic library
|lib/libmkldnn.a        | Intel MKL-DNN static library (if built with `MKLDNN_LIBRARY_TYPE=STATIC`)


#### Intel MKL-DNN with OpenMP

If Intel MKL-DNN is built with Intel MKL small libraries the following extra
libraries would be installed:

|File                   | Description
|:---                   |:---
|lib/libiomp5.so        | Intel OpenMP* runtime library
|lib/libmklml_gnu.so    | Intel MKL small library for GNU* OpenMP runtime
|lib/libmklml_intel.so  | Intel MKL small library for Intel(R) OpenMP runtime

Intel MKL-DNN uses OpenMP\* for parallelism and requires an OpenMP runtime
library to work. As different OpenMP runtimes may not be binary compatible
it's important to ensure that only one OpenMP runtime is used throughout the
application. Having more than one OpenMP runtime initialized may lead to
undefined behavior resulting in incorrect results or crashes.

Intel MKL-DNN library built with binary dependency will link against Intel OpenMP
runtime included with Intel MKL small libraries package. Intel OpenMP runtime
is binary compatible with GNU OpenMP and CLANG OpenMP runtimes and is 
recommended for the best performance results. Here are example linklines for 
GNU C++ compiler and Intel C++ compiler.
```
	g++ -std=c++11 -I${MKLDNNROOT}/include -L${MKLDNNROOT}/lib simple_net.cpp -lmkldnn -lmklml_intel -liomp5
```
```
	icpc -std=c++11 -qopenmp -I${MKLDNNROOT}/include -L${MKLDNNROOT}/lib simple_net.cpp -lmkldnn -lmklml_intel
```
Using GNU compiler with `-fopenmp` and `-liomp5` options will link the 
application with both Intel and GNU OpenMP runtime libraries. This will lead
to undefined behavior of the application.

Intel MKL-DNN library built standalone will use OpenMP runtime supplied by
the compiler, so as long as both the library and the application use the
same compiler correct OpenMP runtime will be used. 
```
	g++ -std=c++11 -fopenmp -I${MKLDNNROOT}/include -L${MKLDNNROOT}/lib simple_net.cpp -lmkldnn
```
```
	icpc -std=c++11 -qopenmp -I${MKLDNNROOT}/include -L${MKLDNNROOT}/lib simple_net.cpp -lmkldnn
```

#### Intel MKL-DNN with Intel TBB

Intel MKL-DNN built with Intel TBB doesn't require special handling:
```
	g++ -std=c++11 -I${MKLDNNROOT}/include -L${MKLDNNROOT}/lib simple_net.cpp -lmkldnn -ltbb
```

Please note that Intel MKL-DNN requires Intel TBB 2017 or above.
Also, Intel MKL-DNN has limited optimizations done for Intel TBB
and has some functional limitations if built with Intel TBB.

Functional limitations:
* Convolution with Winograd algorithm is not supported

Performance limitations (mostly less parallelism than in case of OpenMP):
* Batch normalization
* Convolution backward by weights
* mkldnn_sgemm

> **WARNING**
>
> If the library is built with full Intel MKL user is expected to set
> `MKL_THREADING_LAYER` environment variable to either `tbb` or `sequential`
> to force Intel MKL to use Intel TBB for parallelization or to be sequential
> respectively. Without this setting Intel MKL (RT library) by default would
> try to use OpenMP for parallelization.

--------

[Legal Information](doc/legal_information.md)
