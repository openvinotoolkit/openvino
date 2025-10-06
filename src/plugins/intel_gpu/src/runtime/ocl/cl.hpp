//
// Copyright (c) 2008-2024 The Khronos Group Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

/*! \file
 *
 *   \brief C++ bindings for OpenCL 1.0, OpenCL 1.1, OpenCL 1.2,
 *       OpenCL 2.0, OpenCL 2.1, OpenCL 2.2, and OpenCL 3.0.
 *   \author Lee Howes and Bruce Merry
 *
 *   Derived from the OpenCL 1.x C++ bindings written by
 *   Benedict R. Gaster, Laurent Morichetti and Lee Howes
 *   With additions and fixes from:
 *       Brian Cole, March 3rd 2010 and April 2012
 *       Matt Gruenke, April 2012.
 *       Bruce Merry, February 2013.
 *       Tom Deakin and Simon McIntosh-Smith, July 2013
 *       James Price, 2015-
 *   \version 2.2.0
 *   \date 2019-09-18
 *
 *   Optional extension support
 *
 *         cl_khr_d3d10_sharing
 *         #define CL_HPP_USE_DX_INTEROP
 *         cl_khr_il_program
 *         #define CL_HPP_USE_IL_KHR
 *         cl_khr_sub_groups
 *         #define CL_HPP_USE_CL_SUB_GROUPS_KHR
 *
 *   Doxygen documentation for this header is available here:
 *
 *       http://khronosgroup.github.io/OpenCL-CLHPP/
 *
 *   The latest version of this header can be found on the GitHub releases page:
 *
 *       https://github.com/KhronosGroup/OpenCL-CLHPP/releases
 *
 *   Bugs and patches can be submitted to the GitHub repository:
 *
 *       https://github.com/KhronosGroup/OpenCL-CLHPP
 */

/*! \mainpage
 * \section intro Introduction
 * For many large applications C++ is the language of choice and so it seems
 * reasonable to define C++ bindings for OpenCL.
 *
 * The interface is contained with a single C++ header file \em opencl.hpp and all
 * definitions are contained within the namespace \em cl. There is no additional
 * requirement to include \em cl.h and to use either the C++ or original C
 * bindings; it is enough to simply include \em opencl.hpp.
 *
 * The bindings themselves are lightweight and correspond closely to the
 * underlying C API. Using the C++ bindings introduces no additional execution
 * overhead.
 *
 * There are numerous compatibility, portability and memory management
 * fixes in the new header as well as additional OpenCL 2.0 features.
 * As a result the header is not directly backward compatible and for this
 * reason we release it as opencl.hpp rather than a new version of cl.hpp.
 * 
 *
 * \section compatibility Compatibility
 * Due to the evolution of the underlying OpenCL API the 2.0 C++ bindings
 * include an updated approach to defining supported feature versions
 * and the range of valid underlying OpenCL runtime versions supported.
 *
 * The combination of preprocessor macros CL_HPP_TARGET_OPENCL_VERSION and 
 * CL_HPP_MINIMUM_OPENCL_VERSION control this range. These are three digit
 * decimal values representing OpenCL runtime versions. The default for 
 * the target is 300, representing OpenCL 3.0.  The minimum is defined as 200.
 * These settings would use 2.0 and newer API calls only.
 * If backward compatibility with a 1.2 runtime is required, the minimum
 * version may be set to 120.
 *
 * Note that this is a compile-time setting, and so affects linking against
 * a particular SDK version rather than the versioning of the loaded runtime.
 *
 * The earlier versions of the header included basic vector and string 
 * classes based loosely on STL versions. These were difficult to 
 * maintain and very rarely used. For the 2.0 header we now assume
 * the presence of the standard library unless requested otherwise.
 * We use std::array, std::vector, std::shared_ptr and std::string 
 * throughout to safely manage memory and reduce the chance of a 
 * recurrance of earlier memory management bugs.
 *
 * These classes are used through typedefs in the cl namespace: 
 * cl::array, cl::vector, cl::pointer and cl::string.
 * In addition cl::allocate_pointer forwards to std::allocate_shared
 * by default.
 * In all cases these standard library classes can be replaced with 
 * custom interface-compatible versions using the CL_HPP_NO_STD_ARRAY, 
 * CL_HPP_NO_STD_VECTOR, CL_HPP_NO_STD_UNIQUE_PTR and 
 * CL_HPP_NO_STD_STRING macros.
 *
 * The OpenCL 1.x versions of the C++ bindings included a size_t wrapper
 * class to interface with kernel enqueue. This caused unpleasant interactions
 * with the standard size_t declaration and led to namespacing bugs.
 * In the 2.0 version we have replaced this with a std::array-based interface.
 * However, the old behaviour can be regained for backward compatibility
 * using the CL_HPP_ENABLE_SIZE_T_COMPATIBILITY macro.
 *
 * Finally, the program construction interface used a clumsy vector-of-pairs
 * design in the earlier versions. We have replaced that with a cleaner 
 * vector-of-vectors and vector-of-strings design. However, for backward 
 * compatibility old behaviour can be regained with the
 * CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY macro.
 * 
 * In OpenCL 2.0 OpenCL C is not entirely backward compatibility with 
 * earlier versions. As a result a flag must be passed to the OpenCL C
 * compiled to request OpenCL 2.0 compilation of kernels with 1.2 as
 * the default in the absence of the flag.
 * In some cases the C++ bindings automatically compile code for ease.
 * For those cases the compilation defaults to OpenCL C 2.0.
 * If this is not wanted, the CL_HPP_CL_1_2_DEFAULT_BUILD macro may
 * be specified to assume 1.2 compilation.
 * If more fine-grained decisions on a per-kernel bases are required
 * then explicit build operations that take the flag should be used.
 *
 *
 * \section parameterization Parameters
 * This header may be parameterized by a set of preprocessor macros.
 *
 * - CL_HPP_TARGET_OPENCL_VERSION
 *
 *   Defines the target OpenCL runtime version to build the header
 *   against. Defaults to 300, representing OpenCL 3.0.
 *
 * - CL_HPP_MINIMUM_OPENCL_VERSION
 *
 *   Defines the minimum OpenCL runtime version to build the header
 *   against. Defaults to 200, representing OpenCL 2.0.
 *
 * - CL_HPP_NO_STD_STRING
 *
 *   Do not use the standard library string class. cl::string is not
 *   defined and may be defined by the user before opencl.hpp is
 *   included.
 *
 * - CL_HPP_NO_STD_VECTOR
 *
 *   Do not use the standard library vector class. cl::vector is not
 *   defined and may be defined by the user before opencl.hpp is
 *   included.
 *
 * - CL_HPP_NO_STD_ARRAY
 *
 *   Do not use the standard library array class. cl::array is not
 *   defined and may be defined by the user before opencl.hpp is
 *   included.
 *
 * - CL_HPP_NO_STD_UNIQUE_PTR
 *
 *   Do not use the standard library unique_ptr class. cl::pointer and
 *   the cl::allocate_pointer functions are not defined and may be
 *   defined by the user before opencl.hpp is included.
 *
 * - CL_HPP_ENABLE_EXCEPTIONS
 *
 *   Enable exceptions for use in the C++ bindings header. This is the
 *   preferred error handling mechanism but is not required.
 *
 * - CL_HPP_ENABLE_SIZE_T_COMPATIBILITY
 *
 *   Backward compatibility option to support cl.hpp-style size_t
 *   class.  Replaces the updated std::array derived version and
 *   removal of size_t from the namespace. Note that in this case the
 *   new size_t class is placed in the cl::compatibility namespace and
 *   thus requires an additional using declaration for direct backward
 *   compatibility.
 *
 * - CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY
 *
 *   Enable older vector of pairs interface for construction of
 *   programs.
 *
 * - CL_HPP_CL_1_2_DEFAULT_BUILD
 *
 *   Default to OpenCL C 1.2 compilation rather than OpenCL C 2.0
 *   applies to use of cl::Program construction and other program
 *   build variants.
 *
 *
 * - CL_HPP_USE_CL_SUB_GROUPS_KHR
 *
 *   Enable the cl_khr_subgroups extension.
 *
 * - CL_HPP_USE_DX_INTEROP
 *
 *   Enable the cl_khr_d3d10_sharing extension.
 *
 * - CL_HPP_USE_IL_KHR
 *
 *   Enable the cl_khr_il_program extension.
 *
 *
 * \section example Example
 *
 * The following example shows a general use case for the C++
 * bindings, including support for the optional exception feature and
 * also the supplied vector and string classes, see following sections for
 * decriptions of these features.
 * 
 * Note: the C++ bindings use std::call_once and therefore may need to be
 * compiled using special command-line options (such as "-pthread") on some
 * platforms!
 *
 * \code
    #define CL_HPP_ENABLE_EXCEPTIONS
    #define CL_HPP_TARGET_OPENCL_VERSION 200

    #include <CL/opencl.hpp>
    #include <iostream>
    #include <vector>
    #include <memory>
    #include <algorithm>

    const int numElements = 32;

    int main(void)
    {
        // Filter for a 2.0 or newer platform and set it as the default
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        cl::Platform plat;
        for (auto &p : platforms) {
            std::string platver = p.getInfo<CL_PLATFORM_VERSION>();
            if (platver.find("OpenCL 2.") != std::string::npos ||
                platver.find("OpenCL 3.") != std::string::npos) {
                // Note: an OpenCL 3.x platform may not support all required features!
                plat = p;
            }
        }
        if (plat() == 0) {
            GPU_DEBUG_LOG << "No OpenCL 2.0 or newer platform found.\n";
            return -1;
        }

        cl::Platform newP = cl::Platform::setDefault(plat);
        if (newP != plat) {
            GPU_DEBUG_LOG << "Error setting default platform.\n";
            return -1;
        }

        // C++11 raw string literal for the first kernel
        std::string kernel1{R"CLC(
            global int globalA;
            kernel void updateGlobal()
            {
              globalA = 75;
            }
        )CLC"};

        // Raw string literal for the second kernel
        std::string kernel2{R"CLC(
            typedef struct { global int *bar; } Foo;
            kernel void vectorAdd(global const Foo* aNum, global const int *inputA, global const int *inputB,
                                  global int *output, int val, write_only pipe int outPipe, queue_t childQueue)
            {
              output[get_global_id(0)] = inputA[get_global_id(0)] + inputB[get_global_id(0)] + val + *(aNum->bar);
              write_pipe(outPipe, &val);
              queue_t default_queue = get_default_queue();
              ndrange_t ndrange = ndrange_1D(get_global_size(0)/2, get_global_size(0)/2);

              // Have a child kernel write into third quarter of output
              enqueue_kernel(default_queue, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange,
                ^{
                    output[get_global_size(0)*2 + get_global_id(0)] =
                      inputA[get_global_size(0)*2 + get_global_id(0)] + inputB[get_global_size(0)*2 + get_global_id(0)] + globalA;
                });

              // Have a child kernel write into last quarter of output
              enqueue_kernel(childQueue, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange,
                ^{
                    output[get_global_size(0)*3 + get_global_id(0)] =
                      inputA[get_global_size(0)*3 + get_global_id(0)] + inputB[get_global_size(0)*3 + get_global_id(0)] + globalA + 2;
                });
            }
        )CLC"};

        std::vector<std::string> programStrings;
        programStrings.push_back(kernel1);
        programStrings.push_back(kernel2);

        cl::Program vectorAddProgram(programStrings);
        try {
            vectorAddProgram.build("-cl-std=CL2.0");
        }
        catch (...) {
            // Print build info for all devices
            cl_int buildErr = CL_SUCCESS;
            auto buildInfo = vectorAddProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
            for (auto &pair : buildInfo) {
                std::cerr << pair.second << std::endl << std::endl;
            }

            return 1;
        }

        typedef struct { int *bar; } Foo;

        // Get and run kernel that initializes the program-scope global
        // A test for kernels that take no arguments
        auto program2Kernel =
            cl::KernelFunctor<>(vectorAddProgram, "updateGlobal");
        program2Kernel(
            cl::EnqueueArgs(
            cl::NDRange(1)));

        //////////////////
        // SVM allocations

        auto anSVMInt = cl::allocate_svm<int, cl::SVMTraitCoarse<>>();
        *anSVMInt = 5;
        cl::SVMAllocator<Foo, cl::SVMTraitCoarse<cl::SVMTraitReadOnly<>>> svmAllocReadOnly;
        auto fooPointer = cl::allocate_pointer<Foo>(svmAllocReadOnly);
        fooPointer->bar = anSVMInt.get();
        cl::SVMAllocator<int, cl::SVMTraitCoarse<>> svmAlloc;
        std::vector<int, cl::SVMAllocator<int, cl::SVMTraitCoarse<>>> inputA(numElements, 1, svmAlloc);
        cl::coarse_svm_vector<int> inputB(numElements, 2, svmAlloc);

        //////////////
        // Traditional cl_mem allocations

        std::vector<int> output(numElements, 0xdeadbeef);
        cl::Buffer outputBuffer(output.begin(), output.end(), false);
        cl::Pipe aPipe(sizeof(cl_int), numElements / 2);

        // Default command queue, also passed in as a parameter
        cl::DeviceCommandQueue defaultDeviceQueue = cl::DeviceCommandQueue::makeDefault(
            cl::Context::getDefault(), cl::Device::getDefault());

        auto vectorAddKernel =
            cl::KernelFunctor<
                decltype(fooPointer)&,
                int*,
                cl::coarse_svm_vector<int>&,
                cl::Buffer,
                int,
                cl::Pipe&,
                cl::DeviceCommandQueue
                >(vectorAddProgram, "vectorAdd");

        // Ensure that the additional SVM pointer is available to the kernel
        // This one was not passed as a parameter
        vectorAddKernel.setSVMPointers(anSVMInt);

        cl_int error;
        vectorAddKernel(
            cl::EnqueueArgs(
                cl::NDRange(numElements/2),
                cl::NDRange(numElements/2)),
            fooPointer,
            inputA.data(),
            inputB,
            outputBuffer,
            3,
            aPipe,
            defaultDeviceQueue,
            error
            );

        cl::copy(outputBuffer, output.begin(), output.end());

        cl::Device d = cl::Device::getDefault();

        GPU_DEBUG_LOG << "Output:\n";
        for (int i = 1; i < numElements; ++i) {
            GPU_DEBUG_LOG << "\t" << output[i] << "\n";
        }
        GPU_DEBUG_LOG << "\n\n";

        return 0;
    }
 *
 * \endcode
 *
 */
#ifndef CL_HPP_
#define CL_HPP_
/* Handle deprecated preprocessor definitions. In each case, we only check for
 * the old name if the new name is not defined, so that user code can define
 * both and hence work with either version of the bindings.
 */
#if !defined(CL_HPP_USE_DX_INTEROP) && defined(USE_DX_INTEROP)
# pragma message("opencl.hpp: USE_DX_INTEROP is deprecated. Define CL_HPP_USE_DX_INTEROP instead")
# define CL_HPP_USE_DX_INTEROP
#endif
#if !defined(CL_HPP_ENABLE_EXCEPTIONS) && defined(__CL_ENABLE_EXCEPTIONS)
# pragma message("opencl.hpp: __CL_ENABLE_EXCEPTIONS is deprecated. Define CL_HPP_ENABLE_EXCEPTIONS instead")
# define CL_HPP_ENABLE_EXCEPTIONS
#endif
#if !defined(CL_HPP_NO_STD_VECTOR) && defined(__NO_STD_VECTOR)
# pragma message("opencl.hpp: __NO_STD_VECTOR is deprecated. Define CL_HPP_NO_STD_VECTOR instead")
# define CL_HPP_NO_STD_VECTOR
#endif
#if !defined(CL_HPP_NO_STD_STRING) && defined(__NO_STD_STRING)
# pragma message("opencl.hpp: __NO_STD_STRING is deprecated. Define CL_HPP_NO_STD_STRING instead")
# define CL_HPP_NO_STD_STRING
#endif
#if defined(VECTOR_CLASS)
# pragma message("opencl.hpp: VECTOR_CLASS is deprecated. Alias cl::vector instead")
#endif
#if defined(STRING_CLASS)
# pragma message("opencl.hpp: STRING_CLASS is deprecated. Alias cl::string instead.")
#endif
#if !defined(CL_HPP_USER_OVERRIDE_ERROR_STRINGS) && defined(__CL_USER_OVERRIDE_ERROR_STRINGS)
# pragma message("opencl.hpp: __CL_USER_OVERRIDE_ERROR_STRINGS is deprecated. Define CL_HPP_USER_OVERRIDE_ERROR_STRINGS instead")
# define CL_HPP_USER_OVERRIDE_ERROR_STRINGS
#endif

/* Warn about features that are no longer supported
 */
#if defined(__USE_DEV_VECTOR)
# pragma message("opencl.hpp: __USE_DEV_VECTOR is no longer supported. Expect compilation errors")
#endif
#if defined(__USE_DEV_STRING)
# pragma message("opencl.hpp: __USE_DEV_STRING is no longer supported. Expect compilation errors")
#endif

/* Detect which version to target */
#if !defined(CL_HPP_TARGET_OPENCL_VERSION)
# pragma message("opencl.hpp: CL_HPP_TARGET_OPENCL_VERSION is not defined. It will default to 300 (OpenCL 3.0)")
# define CL_HPP_TARGET_OPENCL_VERSION 300
#endif
#if CL_HPP_TARGET_OPENCL_VERSION != 100 && \
    CL_HPP_TARGET_OPENCL_VERSION != 110 && \
    CL_HPP_TARGET_OPENCL_VERSION != 120 && \
    CL_HPP_TARGET_OPENCL_VERSION != 200 && \
    CL_HPP_TARGET_OPENCL_VERSION != 210 && \
    CL_HPP_TARGET_OPENCL_VERSION != 220 && \
    CL_HPP_TARGET_OPENCL_VERSION != 300
# pragma message("opencl.hpp: CL_HPP_TARGET_OPENCL_VERSION is not a valid value (100, 110, 120, 200, 210, 220 or 300). It will be set to 300 (OpenCL 3.0).")
# undef CL_HPP_TARGET_OPENCL_VERSION
# define CL_HPP_TARGET_OPENCL_VERSION 300
#endif

/* Forward target OpenCL version to C headers if necessary */
#if defined(CL_TARGET_OPENCL_VERSION)
/* Warn if prior definition of CL_TARGET_OPENCL_VERSION is lower than
 * requested C++ bindings version */
#if CL_TARGET_OPENCL_VERSION < CL_HPP_TARGET_OPENCL_VERSION
# pragma message("CL_TARGET_OPENCL_VERSION is already defined as is lower than CL_HPP_TARGET_OPENCL_VERSION")
#endif
#else
# define CL_TARGET_OPENCL_VERSION CL_HPP_TARGET_OPENCL_VERSION
#endif

#if !defined(CL_HPP_MINIMUM_OPENCL_VERSION)
# define CL_HPP_MINIMUM_OPENCL_VERSION 200
#endif
#if CL_HPP_MINIMUM_OPENCL_VERSION != 100 && \
    CL_HPP_MINIMUM_OPENCL_VERSION != 110 && \
    CL_HPP_MINIMUM_OPENCL_VERSION != 120 && \
    CL_HPP_MINIMUM_OPENCL_VERSION != 200 && \
    CL_HPP_MINIMUM_OPENCL_VERSION != 210 && \
    CL_HPP_MINIMUM_OPENCL_VERSION != 220 && \
    CL_HPP_MINIMUM_OPENCL_VERSION != 300
# pragma message("opencl.hpp: CL_HPP_MINIMUM_OPENCL_VERSION is not a valid value (100, 110, 120, 200, 210, 220 or 300). It will be set to 100")
# undef CL_HPP_MINIMUM_OPENCL_VERSION
# define CL_HPP_MINIMUM_OPENCL_VERSION 100
#endif
#if CL_HPP_MINIMUM_OPENCL_VERSION > CL_HPP_TARGET_OPENCL_VERSION
# error "CL_HPP_MINIMUM_OPENCL_VERSION must not be greater than CL_HPP_TARGET_OPENCL_VERSION"
#endif

#if CL_HPP_MINIMUM_OPENCL_VERSION <= 100 && !defined(CL_USE_DEPRECATED_OPENCL_1_0_APIS)
# define CL_USE_DEPRECATED_OPENCL_1_0_APIS
#endif
#if CL_HPP_MINIMUM_OPENCL_VERSION <= 110 && !defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
# define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#endif
#if CL_HPP_MINIMUM_OPENCL_VERSION <= 120 && !defined(CL_USE_DEPRECATED_OPENCL_1_2_APIS)
# define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif
#if CL_HPP_MINIMUM_OPENCL_VERSION <= 200 && !defined(CL_USE_DEPRECATED_OPENCL_2_0_APIS)
# define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#endif
#if CL_HPP_MINIMUM_OPENCL_VERSION <= 210 && !defined(CL_USE_DEPRECATED_OPENCL_2_1_APIS)
# define CL_USE_DEPRECATED_OPENCL_2_1_APIS
#endif
#if CL_HPP_MINIMUM_OPENCL_VERSION <= 220 && !defined(CL_USE_DEPRECATED_OPENCL_2_2_APIS)
# define CL_USE_DEPRECATED_OPENCL_2_2_APIS
#endif

#ifdef _WIN32

#include <malloc.h>

#if defined(CL_HPP_USE_DX_INTEROP)
#include <CL/cl_d3d10.h>
#include <CL/cl_dx9_media_sharing.h>
#endif
#endif // _WIN32

#if defined(_MSC_VER)
#include <intrin.h>
#endif // _MSC_VER 
 
 // Check for a valid C++ version

// Need to do both tests here because for some reason __cplusplus is not 
// updated in visual studio
#if (!defined(_MSC_VER) && __cplusplus < 201103L) || (defined(_MSC_VER) && _MSC_VER < 1700)
#error Visual studio 2013 or another C++11-supporting compiler required
#endif
#include <iostream>
#include "intel_gpu/runtime/debug_configuration.hpp"
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/opencl.h>
#else
#include "opencl.h"
#endif // !__APPLE__

#if __cplusplus >= 201703L
# define CL_HPP_DEFINE_STATIC_MEMBER_ inline
#elif defined(_MSC_VER)
# define CL_HPP_DEFINE_STATIC_MEMBER_ __declspec(selectany)
#elif defined(__MINGW32__)
# define CL_HPP_DEFINE_STATIC_MEMBER_ __attribute__((selectany))
#else
# define CL_HPP_DEFINE_STATIC_MEMBER_ __attribute__((weak))
#endif // !_MSC_VER

// Define deprecated prefixes and suffixes to ensure compilation
// in case they are not pre-defined
#if !defined(CL_API_PREFIX__VERSION_1_1_DEPRECATED)
#define CL_API_PREFIX__VERSION_1_1_DEPRECATED
#endif // #if !defined(CL_API_PREFIX__VERSION_1_1_DEPRECATED)
#if !defined(CL_API_SUFFIX__VERSION_1_1_DEPRECATED)
#define CL_API_SUFFIX__VERSION_1_1_DEPRECATED
#endif // #if !defined(CL_API_SUFFIX__VERSION_1_1_DEPRECATED)

#if !defined(CL_API_PREFIX__VERSION_1_2_DEPRECATED)
#define CL_API_PREFIX__VERSION_1_2_DEPRECATED
#endif // #if !defined(CL_API_PREFIX__VERSION_1_2_DEPRECATED)
#if !defined(CL_API_SUFFIX__VERSION_1_2_DEPRECATED)
#define CL_API_SUFFIX__VERSION_1_2_DEPRECATED
#endif // #if !defined(CL_API_SUFFIX__VERSION_1_2_DEPRECATED)

#if !defined(CL_API_PREFIX__VERSION_2_2_DEPRECATED)
#define CL_API_PREFIX__VERSION_2_2_DEPRECATED
#endif // #if !defined(CL_API_PREFIX__VERSION_2_2_DEPRECATED)
#if !defined(CL_API_SUFFIX__VERSION_2_2_DEPRECATED)
#define CL_API_SUFFIX__VERSION_2_2_DEPRECATED
#endif // #if !defined(CL_API_SUFFIX__VERSION_2_2_DEPRECATED)

#if !defined(CL_CALLBACK)
#define CL_CALLBACK
#endif //CL_CALLBACK

#include <utility>
#include <limits>
#include <iterator>
#include <mutex>
#include <cstring>
#include <functional>
#include "ocl_indirect.hpp"
typedef void* cl_command_buffer_khr;
typedef unsigned long cl_command_buffer_properties_khr;
typedef unsigned long cl_command_buffer_info_khr;
typedef unsigned long cl_sync_point_khr;
typedef unsigned long cl_mutable_command_info_khr;
typedef void* cl_mutable_command_khr;
// === Symulacja funkcji C API (bez call_... - tylko właściwe API OpenCL) ===
//inline cl_int clGetPlatformIDs(cl_uint, cl_platform_id*, cl_uint*) { return CL_SUCCESS; }
//inline cl_int clGetDeviceIDs(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*) { return CL_SUCCESS; }
inline cl_context clCreateContext(const cl_context_properties*, cl_uint, const cl_device_id*, void(*)(const char*, const void*, size_t, void*), void*, cl_int*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return nullptr; }
inline cl_command_queue clCreateCommandQueue(cl_context a, cl_device_id b, cl_command_queue_properties c, cl_int* d) { return call_clCreateCommandQueue(a, b, c, d); }
inline cl_program clCreateProgramWithSource(cl_context a, cl_uint b, const char** c, const size_t* d, cl_int* e) { return call_clCreateProgramWithSource(a, b, c, d, e); }
inline cl_int clBuildProgram(cl_program a, cl_uint b, const cl_device_id* c, const char* d, void(*e)(cl_program, void*), void* f) { return call_clBuildProgram(a, b, c, d, e, f); }
inline cl_kernel clCreateKernel(cl_program, const char*, cl_int*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return nullptr; }
inline cl_mem clCreateBuffer(cl_context, cl_mem_flags, size_t, void*, cl_int*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return nullptr; }
inline cl_int clSetKernelArg(cl_kernel, cl_uint, size_t, const void*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clEnqueueNDRangeKernel(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clFinish(cl_command_queue) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clReleaseMemObject(cl_mem) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clReleaseKernel(cl_kernel) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl;  return CL_SUCCESS; }
inline cl_int clReleaseProgram(cl_program) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clReleaseCommandQueue(cl_command_queue) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clReleaseContext(cl_context) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }

// 🔥 Funkcje z rozszerzenia cl_khr_command_buffer
inline cl_int clCreateCommandBufferKHR(cl_uint, const cl_command_queue*, const cl_command_buffer_properties_khr*, cl_int*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clFinalizeCommandBufferKHR(cl_command_buffer_khr) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clRetainCommandBufferKHR(cl_command_buffer_khr) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clReleaseCommandBufferKHR(cl_command_buffer_khr) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clGetCommandBufferInfoKHR(cl_command_buffer_khr, cl_command_buffer_info_khr, size_t, void*, size_t*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clEnqueueCommandBufferKHR(cl_uint, const cl_command_queue*, cl_command_buffer_khr, cl_uint, const cl_event*, cl_event*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clCommandBarrierWithWaitListKHR(cl_command_buffer_khr, cl_command_queue, const void*, cl_uint, const cl_sync_point_khr*, cl_sync_point_khr*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clCommandCopyBufferKHR(cl_command_buffer_khr, cl_command_queue, const void*, cl_mem, cl_mem, size_t, size_t, size_t, cl_uint, const cl_sync_point_khr*, cl_sync_point_khr*, cl_mutable_command_khr*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }

// 🔥 Funkcje z rozszerzenia cl_khr_semaphore
inline cl_int clCreateSemaphoreKHR(cl_context, const cl_semaphore_properties_khr*, cl_int*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clRetainSemaphoreKHR(cl_semaphore_khr) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clReleaseSemaphoreKHR(cl_semaphore_khr) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clEnqueueWaitSemaphoresKHR(cl_command_queue, cl_uint, const cl_semaphore_khr*, const cl_semaphore_payload_khr*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clEnqueueSignalSemaphoresKHR(cl_command_queue, cl_uint, const cl_semaphore_khr*, const cl_semaphore_payload_khr*, cl_uint, const cl_event*, cl_event*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clGetSemaphoreInfoKHR(cl_semaphore_khr, cl_semaphore_info_khr, size_t, void*, size_t*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }

// 🔥 Funkcje z D3D, IL, itp.
inline cl_int clEnqueueAcquireD3D10ObjectsKHR(cl_command_queue, cl_uint, const cl_mem*, cl_uint, const cl_event*, cl_event*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clEnqueueReleaseD3D10ObjectsKHR(cl_command_queue, cl_uint, const cl_mem*, cl_uint, const cl_event*, cl_event*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clEnqueueAcquireGLObjects(cl_command_queue, cl_uint, const cl_mem*, cl_uint, const cl_event*, cl_event*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clEnqueueReleaseGLObjects(cl_command_queue, cl_uint, const cl_mem*, cl_uint, const cl_event*, cl_event*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }

// 🔥 Funkcje z IL
inline cl_program clCreateProgramWithIL(cl_context, const void*, size_t, cl_int*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }

// 🔥 Funkcje SVM
inline cl_int clSetKernelArgSVMPointer(cl_kernel, cl_uint, const void*) { return CL_SUCCESS; }
inline cl_int clEnqueueSVMMemcpy(cl_command_queue, cl_bool, void*, const void*, size_t, cl_uint, const cl_event*, cl_event*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clEnqueueSVMMap(cl_command_queue, cl_bool, cl_map_flags, void*, size_t, cl_uint, const cl_event*, cl_event*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clEnqueueSVMUnmap(cl_command_queue, void*, cl_uint, const cl_event*, cl_event*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }

// 🔥 Funkcje z profiling, getInfo, itp.
inline cl_int clGetEventInfo(cl_event, cl_event_info, size_t, void*, size_t*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clGetEventProfilingInfo(cl_event, cl_profiling_info, size_t, void*, size_t*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clWaitForEvents(cl_uint, const cl_event*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clFlush(cl_command_queue) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline void* clEnqueueMapBuffer(cl_command_queue, cl_mem, cl_bool, cl_map_flags, size_t, size_t, cl_uint, const cl_event*, cl_event*, cl_int*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clEnqueueUnmapMemObject(cl_command_queue, cl_mem, void*, cl_uint, const cl_event*, cl_event*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clEnqueueReadBuffer(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clEnqueueWriteBuffer(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clEnqueueCopyBuffer(cl_command_queue, cl_mem, cl_mem, size_t, size_t, size_t, cl_uint, const cl_event*, cl_event*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clEnqueueReadImage(cl_command_queue, cl_mem, cl_bool, const size_t*, const size_t*, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clEnqueueWriteImage(cl_command_queue, cl_mem, cl_bool, const size_t*, const size_t*, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clEnqueueCopyImage(cl_command_queue, cl_mem, cl_mem, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clEnqueueCopyImageToBuffer(cl_command_queue, cl_mem, cl_mem, const size_t*, const size_t*, size_t, cl_uint, const cl_event*, cl_event*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl;  return CL_SUCCESS; }
inline cl_int clEnqueueCopyBufferToImage(cl_command_queue, cl_mem, cl_mem, size_t, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clEnqueueMarkerWithWaitList(cl_command_queue, cl_uint, const cl_event*, cl_event*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clEnqueueBarrierWithWaitList(cl_command_queue, cl_uint, const cl_event*, cl_event*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clEnqueueWaitForEvents(cl_command_queue, cl_uint, const cl_event*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clSetMemObjectDestructorCallback(cl_mem, void(*)(cl_mem, void*), void*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
typedef void* cl_command_buffer_khr;
inline cl_program clLinkProgram(cl_context, cl_uint, const cl_device_id*, const char*, cl_uint, const cl_program*, void(*)(cl_program, void*), void*, cl_int*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return nullptr; }
inline cl_int clGetKernelArgInfo(cl_kernel, cl_uint, cl_kernel_arg_info, size_t, void*, size_t*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clGetImageRequirementsInfoEXT(cl_context, const cl_mem_properties*, cl_mem_flags, const cl_image_format*, const cl_image_desc*, cl_image_requirements_info_ext, size_t, void*, size_t*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clCreateSubDevicesEXT(cl_device_id, const cl_device_partition_property*, cl_uint, cl_device_id*, cl_uint*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_context clCreateContextFromType(const cl_context_properties*, cl_device_type, void(*)(const char*, const void*, size_t, void*), void*, cl_int*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return nullptr; }
inline cl_int clGetMemObjectInfo(cl_mem, cl_mem_info, size_t, void*, size_t*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clGetProgramBuildInfo(cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clGetKernelInfo(cl_kernel, cl_kernel_info, size_t, void*, size_t*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clGetKernelWorkGroupInfo(cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t, void*, size_t*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clGetCommandQueueInfo(cl_command_queue, cl_command_queue_info, size_t, void*, size_t*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clGetPlatformInfo(cl_platform_id a, cl_platform_info b , size_t c , void* d, size_t* e) { return call_clGetPlatformInfo(a, b, c, d, e); }
inline cl_int clGetDeviceInfo(cl_device_id a, cl_device_info b, size_t c, void* d, size_t* e) { return call_clGetDeviceInfo(a, b, c, d, e); }
inline cl_int clCreateKernelsInProgram(cl_program a, cl_uint b , cl_kernel* c , cl_uint* d) { return call_clCreateKernelsInProgram(a, b, c, d); }
inline cl_kernel clCloneKernel(cl_kernel, cl_int*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return nullptr; }
inline cl_int clSetKernelExecInfo(cl_kernel, cl_kernel_exec_info, size_t, const void*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_event clCreateUserEvent(cl_context, cl_int*) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl;  return nullptr; }
//
inline cl_int clReleaseDevice(cl_device_id device) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clReleaseEvent(cl_event event) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clRetainDevice(cl_device_id device) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clRetainProgram(cl_program program) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clRetainKernel(cl_kernel kernel) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clRetainCommandQueue(cl_command_queue queue) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clRetainContext(cl_context context) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clGetProgramInfo(cl_program program, cl_program_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clCreateSubDevices(cl_device_id in_device, const cl_device_partition_property* properties, cl_uint num_entries, cl_device_id* out_devices, cl_uint* num_devices) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_int clGetContextInfo(cl_context context, cl_context_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_mem clCreateImage(cl_context context, cl_mem_flags flags, const cl_image_format* image_format, const cl_image_desc* image_desc, void* host_ptr, cl_int* errcode_ret) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return CL_SUCCESS; }
inline cl_program clCreateProgramWithBinary(cl_context context, cl_uint num_devices, const cl_device_id* device_list, const size_t* lengths, const unsigned char** binaries, cl_int* binary_status, cl_int* errcode_ret) { return call_clCreateProgramWithBinary(context, num_devices, device_list, lengths, binaries, binary_status, errcode_ret); }
inline cl_command_queue clCreateCommandQueueWithProperties(cl_context context, cl_device_id device, const cl_queue_properties* properties, cl_int* errcode_ret) { std::cout << "xxx" << std::endl; GPU_DEBUG_LOG << "xxxd " << __FILE__ << ":" << __LINE__ << std::endl; return nullptr; }
// Define a size_type to represent a correctly resolved size_t
#if defined(CL_HPP_ENABLE_SIZE_T_COMPATIBILITY)
namespace cl {
    using size_type = ::size_t;
} // namespace cl
#else // #if defined(CL_HPP_ENABLE_SIZE_T_COMPATIBILITY)
namespace cl {
    using size_type = size_t;
} // namespace cl
#endif // #if defined(CL_HPP_ENABLE_SIZE_T_COMPATIBILITY)


#if defined(CL_HPP_ENABLE_EXCEPTIONS)
#include <exception>
#endif // #if defined(CL_HPP_ENABLE_EXCEPTIONS)

#if !defined(CL_HPP_NO_STD_VECTOR)
#include <vector>
namespace cl {
    template < class T, class Alloc = std::allocator<T> >
    using vector = std::vector<T, Alloc>;
} // namespace cl
#endif // #if !defined(CL_HPP_NO_STD_VECTOR)

#if !defined(CL_HPP_NO_STD_STRING)
#include <string>
namespace cl {
    using string = std::string;
} // namespace cl
#endif // #if !defined(CL_HPP_NO_STD_STRING)

#if CL_HPP_TARGET_OPENCL_VERSION >= 200

#if !defined(CL_HPP_NO_STD_UNIQUE_PTR)
#include <memory>
namespace cl {
    // Replace unique_ptr and allocate_pointer for internal use
    // to allow user to replace them
    template<class T, class D>
    using pointer = std::unique_ptr<T, D>;
} // namespace cl
#endif 
#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
#if !defined(CL_HPP_NO_STD_ARRAY)
#include <array>
namespace cl {
    template < class T, size_type N >
    using array = std::array<T, N>;
} // namespace cl
#endif // #if !defined(CL_HPP_NO_STD_ARRAY)

// Define size_type appropriately to allow backward-compatibility
// use of the old size_t interface class
#if defined(CL_HPP_ENABLE_SIZE_T_COMPATIBILITY)
namespace cl {
    namespace compatibility {
        /*! \brief class used to interface between C++ and
        *  OpenCL C calls that require arrays of size_t values, whose
        *  size is known statically.
        */
        template <int N>
        class size_t
        {
        private:
            size_type data_[N];

        public:
            //! \brief Initialize size_t to all 0s
            size_t()
            {
                for (int i = 0; i < N; ++i) {
                    data_[i] = 0;
                }
            }

            size_t(const array<size_type, N> &rhs)
            {
                for (int i = 0; i < N; ++i) {
                    data_[i] = rhs[i];
                }
            }

            size_type& operator[](int index)
            {
                return data_[index];
            }

            const size_type& operator[](int index) const
            {
                return data_[index];
            }

            //! \brief Conversion operator to T*.
            operator size_type* ()             { return data_; }

            //! \brief Conversion operator to const T*.
            operator const size_type* () const { return data_; }

            operator array<size_type, N>() const
            {
                array<size_type, N> ret;

                for (int i = 0; i < N; ++i) {
                    ret[i] = data_[i];
                }
                return ret;
            }
        };
    } // namespace compatibility

    template<int N>
    using size_t = compatibility::size_t<N>;
} // namespace cl
#endif // #if defined(CL_HPP_ENABLE_SIZE_T_COMPATIBILITY)

// Helper alias to avoid confusing the macros
namespace cl {
    namespace detail {
        using size_t_array = array<size_type, 3>;
    } // namespace detail
} // namespace cl


/*! \namespace cl
 *
 * \brief The OpenCL C++ bindings are defined within this namespace.
 *
 */
namespace cl {

#define CL_HPP_CREATE_CL_EXT_FCN_PTR_ALIAS_(name) \
    using PFN_##name = name##_fn

#define CL_HPP_INIT_CL_EXT_FCN_PTR_(name)                               \
    if (!pfn_##name) {                                                  \
        pfn_##name = (PFN_##name)clGetExtensionFunctionAddress(#name);  \
    }

#define CL_HPP_INIT_CL_EXT_FCN_PTR_PLATFORM_(platform, name)            \
    if (!pfn_##name) {                                                  \
        pfn_##name = (PFN_##name)                                       \
            clGetExtensionFunctionAddressForPlatform(platform, #name);  \
    }

#ifdef cl_khr_external_memory
    enum class ExternalMemoryType : cl_external_memory_handle_type_khr;
#endif

    class Memory;
    class Platform;
    class Program;
    class Device;
    class Context;
    class CommandQueue;
    class DeviceCommandQueue;
    class Memory;
    class Buffer;
    class Pipe;
#ifdef cl_khr_semaphore
    class Semaphore;
#endif
#if defined(cl_khr_command_buffer)
    class CommandBufferKhr;
    class MutableCommandKhr;
#endif // cl_khr_command_buffer

#if defined(CL_HPP_ENABLE_EXCEPTIONS)
    /*! \brief Exception class 
     * 
     *  This may be thrown by API functions when CL_HPP_ENABLE_EXCEPTIONS is defined.
     */
    class Error : public std::exception
    {
    private:
        cl_int err_;
        const char * errStr_;
    public:
        /*! \brief Create a new CL error exception for a given error code
         *  and corresponding message.
         * 
         *  \param err error code value.
         *
         *  \param errStr a descriptive string that must remain in scope until
         *                handling of the exception has concluded.  If set, it
         *                will be returned by what().
         */
        Error(cl_int err, const char * errStr = nullptr) : err_(err), errStr_(errStr)
        {}

        /*! \brief Get error string associated with exception
         *
         * \return A memory pointer to the error message string.
         */
        const char * what() const noexcept override
        {
            if (errStr_ == nullptr) {
                return "empty";
            }
            else {
                return errStr_;
            }
        }

        /*! \brief Get error code associated with exception
         *
         *  \return The error code.
         */
        cl_int err(void) const { return err_; }
    };
#define CL_HPP_ERR_STR_(x) #x
#else
#define CL_HPP_ERR_STR_(x) nullptr
#endif // CL_HPP_ENABLE_EXCEPTIONS


namespace detail
{
#if defined(CL_HPP_ENABLE_EXCEPTIONS)
static inline cl_int errHandler (
    cl_int err,
    const char * errStr = nullptr)
{
    if (err != CL_SUCCESS) {
        throw Error(err, errStr);
    }
    return err;
}
#else
static inline cl_int errHandler (cl_int err, const char * errStr = nullptr)
{
    (void) errStr; // suppress unused variable warning
    return err;
}
#endif // CL_HPP_ENABLE_EXCEPTIONS
}



//! \cond DOXYGEN_DETAIL
#if !defined(CL_HPP_USER_OVERRIDE_ERROR_STRINGS)
#define __GET_DEVICE_INFO_ERR               CL_HPP_ERR_STR_(clGetDeviceInfo)
#define __GET_PLATFORM_INFO_ERR             CL_HPP_ERR_STR_(clGetPlatformInfo)
#define __GET_DEVICE_IDS_ERR                CL_HPP_ERR_STR_(clGetDeviceIDs)
#define __GET_PLATFORM_IDS_ERR              CL_HPP_ERR_STR_(clGetPlatformIDs)
#define __GET_CONTEXT_INFO_ERR              CL_HPP_ERR_STR_(clGetContextInfo)
#define __GET_EVENT_INFO_ERR                CL_HPP_ERR_STR_(clGetEventInfo)
#define __GET_EVENT_PROFILE_INFO_ERR        CL_HPP_ERR_STR_(clGetEventProfileInfo)
#define __GET_MEM_OBJECT_INFO_ERR           CL_HPP_ERR_STR_(clGetMemObjectInfo)
#define __GET_IMAGE_INFO_ERR                CL_HPP_ERR_STR_(clGetImageInfo)
#define __GET_SAMPLER_INFO_ERR              CL_HPP_ERR_STR_(clGetSamplerInfo)
#define __GET_KERNEL_INFO_ERR               CL_HPP_ERR_STR_(clGetKernelInfo)
#if CL_HPP_TARGET_OPENCL_VERSION >= 120
#define __GET_KERNEL_ARG_INFO_ERR           CL_HPP_ERR_STR_(clGetKernelArgInfo)
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
#if CL_HPP_TARGET_OPENCL_VERSION >= 210
#define __GET_KERNEL_SUB_GROUP_INFO_ERR     CL_HPP_ERR_STR_(clGetKernelSubGroupInfo)
#else
#define __GET_KERNEL_SUB_GROUP_INFO_ERR     CL_HPP_ERR_STR_(clGetKernelSubGroupInfoKHR)
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 210
#define __GET_KERNEL_WORK_GROUP_INFO_ERR    CL_HPP_ERR_STR_(clGetKernelWorkGroupInfo)
#define __GET_PROGRAM_INFO_ERR              CL_HPP_ERR_STR_(clGetProgramInfo)
#define __GET_PROGRAM_BUILD_INFO_ERR        CL_HPP_ERR_STR_(clGetProgramBuildInfo)
#define __GET_COMMAND_QUEUE_INFO_ERR        CL_HPP_ERR_STR_(clGetCommandQueueInfo)

#define __CREATE_CONTEXT_ERR                CL_HPP_ERR_STR_(clCreateContext)
#define __CREATE_CONTEXT_FROM_TYPE_ERR      CL_HPP_ERR_STR_(clCreateContextFromType)
#define __GET_SUPPORTED_IMAGE_FORMATS_ERR   CL_HPP_ERR_STR_(clGetSupportedImageFormats)
#if CL_HPP_TARGET_OPENCL_VERSION >= 300
#define __SET_CONTEXT_DESCTRUCTOR_CALLBACK_ERR  CL_HPP_ERR_STR_(clSetContextDestructorCallback)
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 300

#define __CREATE_BUFFER_ERR                 CL_HPP_ERR_STR_(clCreateBuffer)
#define __COPY_ERR                          CL_HPP_ERR_STR_(cl::copy)
#define __CREATE_SUBBUFFER_ERR              CL_HPP_ERR_STR_(clCreateSubBuffer)
#define __CREATE_GL_BUFFER_ERR              CL_HPP_ERR_STR_(clCreateFromGLBuffer)
#define __CREATE_GL_RENDER_BUFFER_ERR       CL_HPP_ERR_STR_(clCreateFromGLBuffer)
#define __GET_GL_OBJECT_INFO_ERR            CL_HPP_ERR_STR_(clGetGLObjectInfo)
#if CL_HPP_TARGET_OPENCL_VERSION >= 120
#define __CREATE_IMAGE_ERR                  CL_HPP_ERR_STR_(clCreateImage)
#define __CREATE_GL_TEXTURE_ERR             CL_HPP_ERR_STR_(clCreateFromGLTexture)
#define __IMAGE_DIMENSION_ERR               CL_HPP_ERR_STR_(Incorrect image dimensions)
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
#define __SET_MEM_OBJECT_DESTRUCTOR_CALLBACK_ERR CL_HPP_ERR_STR_(clSetMemObjectDestructorCallback)

#define __CREATE_USER_EVENT_ERR             CL_HPP_ERR_STR_(clCreateUserEvent)
#define __SET_USER_EVENT_STATUS_ERR         CL_HPP_ERR_STR_(clSetUserEventStatus)
#define __SET_EVENT_CALLBACK_ERR            CL_HPP_ERR_STR_(clSetEventCallback)
#define __WAIT_FOR_EVENTS_ERR               CL_HPP_ERR_STR_(clWaitForEvents)

#define __CREATE_KERNEL_ERR                 CL_HPP_ERR_STR_(clCreateKernel)
#define __SET_KERNEL_ARGS_ERR               CL_HPP_ERR_STR_(clSetKernelArg)
#define __CREATE_PROGRAM_WITH_SOURCE_ERR    CL_HPP_ERR_STR_(clCreateProgramWithSource)
#define __CREATE_PROGRAM_WITH_BINARY_ERR    CL_HPP_ERR_STR_(clCreateProgramWithBinary)
#if CL_HPP_TARGET_OPENCL_VERSION >= 210
#define __CREATE_PROGRAM_WITH_IL_ERR        CL_HPP_ERR_STR_(clCreateProgramWithIL)
#else
#define __CREATE_PROGRAM_WITH_IL_ERR        CL_HPP_ERR_STR_(clCreateProgramWithILKHR)
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 210
#if CL_HPP_TARGET_OPENCL_VERSION >= 120
#define __CREATE_PROGRAM_WITH_BUILT_IN_KERNELS_ERR    CL_HPP_ERR_STR_(clCreateProgramWithBuiltInKernels)
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
#define __BUILD_PROGRAM_ERR                 CL_HPP_ERR_STR_(clBuildProgram)
#if CL_HPP_TARGET_OPENCL_VERSION >= 120
#define __COMPILE_PROGRAM_ERR               CL_HPP_ERR_STR_(clCompileProgram)
#define __LINK_PROGRAM_ERR                  CL_HPP_ERR_STR_(clLinkProgram)
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
#define __CREATE_KERNELS_IN_PROGRAM_ERR     CL_HPP_ERR_STR_(clCreateKernelsInProgram)

#if CL_HPP_TARGET_OPENCL_VERSION >= 200
#define __CREATE_COMMAND_QUEUE_WITH_PROPERTIES_ERR          CL_HPP_ERR_STR_(clCreateCommandQueueWithProperties)
#define __CREATE_SAMPLER_WITH_PROPERTIES_ERR                CL_HPP_ERR_STR_(clCreateSamplerWithProperties)
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
#define __SET_COMMAND_QUEUE_PROPERTY_ERR    CL_HPP_ERR_STR_(clSetCommandQueueProperty)
#define __ENQUEUE_READ_BUFFER_ERR           CL_HPP_ERR_STR_(clEnqueueReadBuffer)
#define __ENQUEUE_READ_BUFFER_RECT_ERR      CL_HPP_ERR_STR_(clEnqueueReadBufferRect)
#define __ENQUEUE_WRITE_BUFFER_ERR          CL_HPP_ERR_STR_(clEnqueueWriteBuffer)
#define __ENQUEUE_WRITE_BUFFER_RECT_ERR     CL_HPP_ERR_STR_(clEnqueueWriteBufferRect)
#define __ENQEUE_COPY_BUFFER_ERR            CL_HPP_ERR_STR_(clEnqueueCopyBuffer)
#define __ENQEUE_COPY_BUFFER_RECT_ERR       CL_HPP_ERR_STR_(clEnqueueCopyBufferRect)
#define __ENQUEUE_FILL_BUFFER_ERR           CL_HPP_ERR_STR_(clEnqueueFillBuffer)
#define __ENQUEUE_READ_IMAGE_ERR            CL_HPP_ERR_STR_(clEnqueueReadImage)
#define __ENQUEUE_WRITE_IMAGE_ERR           CL_HPP_ERR_STR_(clEnqueueWriteImage)
#define __ENQUEUE_COPY_IMAGE_ERR            CL_HPP_ERR_STR_(clEnqueueCopyImage)
#define __ENQUEUE_FILL_IMAGE_ERR            CL_HPP_ERR_STR_(clEnqueueFillImage)
#define __ENQUEUE_COPY_IMAGE_TO_BUFFER_ERR  CL_HPP_ERR_STR_(clEnqueueCopyImageToBuffer)
#define __ENQUEUE_COPY_BUFFER_TO_IMAGE_ERR  CL_HPP_ERR_STR_(clEnqueueCopyBufferToImage)
#define __ENQUEUE_MAP_BUFFER_ERR            CL_HPP_ERR_STR_(clEnqueueMapBuffer)
#define __ENQUEUE_MAP_SVM_ERR               CL_HPP_ERR_STR_(clEnqueueSVMMap)
#define __ENQUEUE_FILL_SVM_ERR              CL_HPP_ERR_STR_(clEnqueueSVMMemFill)
#define __ENQUEUE_COPY_SVM_ERR              CL_HPP_ERR_STR_(clEnqueueSVMMemcpy)
#define __ENQUEUE_UNMAP_SVM_ERR             CL_HPP_ERR_STR_(clEnqueueSVMUnmap)
#define __ENQUEUE_MAP_IMAGE_ERR             CL_HPP_ERR_STR_(clEnqueueMapImage)
#define __ENQUEUE_UNMAP_MEM_OBJECT_ERR      CL_HPP_ERR_STR_(clEnqueueUnMapMemObject)
#define __ENQUEUE_NDRANGE_KERNEL_ERR        CL_HPP_ERR_STR_(clEnqueueNDRangeKernel)
#define __ENQUEUE_NATIVE_KERNEL             CL_HPP_ERR_STR_(clEnqueueNativeKernel)
#if CL_HPP_TARGET_OPENCL_VERSION >= 120
#define __ENQUEUE_MIGRATE_MEM_OBJECTS_ERR   CL_HPP_ERR_STR_(clEnqueueMigrateMemObjects)
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
#if CL_HPP_TARGET_OPENCL_VERSION >= 210
#define __ENQUEUE_MIGRATE_SVM_ERR   CL_HPP_ERR_STR_(clEnqueueSVMMigrateMem)
#define __SET_DEFAULT_DEVICE_COMMAND_QUEUE_ERR   CL_HPP_ERR_STR_(clSetDefaultDeviceCommandQueue)
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 210


#define __ENQUEUE_ACQUIRE_GL_ERR            CL_HPP_ERR_STR_(clEnqueueAcquireGLObjects)
#define __ENQUEUE_RELEASE_GL_ERR            CL_HPP_ERR_STR_(clEnqueueReleaseGLObjects)

#define __CREATE_PIPE_ERR             CL_HPP_ERR_STR_(clCreatePipe)
#define __GET_PIPE_INFO_ERR           CL_HPP_ERR_STR_(clGetPipeInfo)

#define __RETAIN_ERR                        CL_HPP_ERR_STR_(Retain Object)
#define __RELEASE_ERR                       CL_HPP_ERR_STR_(Release Object)
#define __FLUSH_ERR                         CL_HPP_ERR_STR_(clFlush)
#define __FINISH_ERR                        CL_HPP_ERR_STR_(clFinish)
#define __VECTOR_CAPACITY_ERR               CL_HPP_ERR_STR_(Vector capacity error)

#if CL_HPP_TARGET_OPENCL_VERSION >= 210
#define __GET_HOST_TIMER_ERR           CL_HPP_ERR_STR_(clGetHostTimer)
#define __GET_DEVICE_AND_HOST_TIMER_ERR           CL_HPP_ERR_STR_(clGetDeviceAndHostTimer)
#endif
#if CL_HPP_TARGET_OPENCL_VERSION >= 220
#define __SET_PROGRAM_RELEASE_CALLBACK_ERR          CL_HPP_ERR_STR_(clSetProgramReleaseCallback)
#define __SET_PROGRAM_SPECIALIZATION_CONSTANT_ERR   CL_HPP_ERR_STR_(clSetProgramSpecializationConstant)
#endif

#ifdef cl_khr_external_memory
#define __ENQUEUE_ACQUIRE_EXTERNAL_MEMORY_ERR       CL_HPP_ERR_STR_(clEnqueueAcquireExternalMemObjectsKHR)
#define __ENQUEUE_RELEASE_EXTERNAL_MEMORY_ERR       CL_HPP_ERR_STR_(clEnqueueReleaseExternalMemObjectsKHR)
#endif

#ifdef cl_khr_semaphore
#define __GET_SEMAPHORE_KHR_INFO_ERR                CL_HPP_ERR_STR_(clGetSemaphoreInfoKHR)
#define __CREATE_SEMAPHORE_KHR_WITH_PROPERTIES_ERR  CL_HPP_ERR_STR_(clCreateSemaphoreWithPropertiesKHR)
#define __ENQUEUE_WAIT_SEMAPHORE_KHR_ERR            CL_HPP_ERR_STR_(clEnqueueWaitSemaphoresKHR)
#define __ENQUEUE_SIGNAL_SEMAPHORE_KHR_ERR          CL_HPP_ERR_STR_(clEnqueueSignalSemaphoresKHR)
#define __RETAIN_SEMAPHORE_KHR_ERR                  CL_HPP_ERR_STR_(clRetainSemaphoreKHR)
#define __RELEASE_SEMAPHORE_KHR_ERR                 CL_HPP_ERR_STR_(clReleaseSemaphoreKHR)
#endif

#ifdef cl_khr_external_semaphore
#define __GET_SEMAPHORE_HANDLE_FOR_TYPE_KHR_ERR         CL_HPP_ERR_STR_(clGetSemaphoreHandleForTypeKHR)
#endif // cl_khr_external_semaphore

#if defined(cl_khr_command_buffer)
#define __CREATE_COMMAND_BUFFER_KHR_ERR             CL_HPP_ERR_STR_(clCreateCommandBufferKHR)
#define __GET_COMMAND_BUFFER_INFO_KHR_ERR           CL_HPP_ERR_STR_(clGetCommandBufferInfoKHR)
#define __FINALIZE_COMMAND_BUFFER_KHR_ERR           CL_HPP_ERR_STR_(clFinalizeCommandBufferKHR)
#define __ENQUEUE_COMMAND_BUFFER_KHR_ERR            CL_HPP_ERR_STR_(clEnqueueCommandBufferKHR)
#define __COMMAND_BARRIER_WITH_WAIT_LIST_KHR_ERR    CL_HPP_ERR_STR_(clCommandBarrierWithWaitListKHR)
#define __COMMAND_COPY_BUFFER_KHR_ERR               CL_HPP_ERR_STR_(clCommandCopyBufferKHR)
#define __COMMAND_COPY_BUFFER_RECT_KHR_ERR          CL_HPP_ERR_STR_(clCommandCopyBufferRectKHR)
#define __COMMAND_COPY_BUFFER_TO_IMAGE_KHR_ERR      CL_HPP_ERR_STR_(clCommandCopyBufferToImageKHR)
#define __COMMAND_COPY_IMAGE_KHR_ERR                CL_HPP_ERR_STR_(clCommandCopyImageKHR)
#define __COMMAND_COPY_IMAGE_TO_BUFFER_KHR_ERR      CL_HPP_ERR_STR_(clCommandCopyImageToBufferKHR)
#define __COMMAND_FILL_BUFFER_KHR_ERR               CL_HPP_ERR_STR_(clCommandFillBufferKHR)
#define __COMMAND_FILL_IMAGE_KHR_ERR                CL_HPP_ERR_STR_(clCommandFillImageKHR)
#define __COMMAND_NDRANGE_KERNEL_KHR_ERR            CL_HPP_ERR_STR_(clCommandNDRangeKernelKHR)
#define __UPDATE_MUTABLE_COMMANDS_KHR_ERR           CL_HPP_ERR_STR_(clUpdateMutableCommandsKHR)
#define __GET_MUTABLE_COMMAND_INFO_KHR_ERR          CL_HPP_ERR_STR_(clGetMutableCommandInfoKHR)
#define __RETAIN_COMMAND_BUFFER_KHR_ERR             CL_HPP_ERR_STR_(clRetainCommandBufferKHR)
#define __RELEASE_COMMAND_BUFFER_KHR_ERR            CL_HPP_ERR_STR_(clReleaseCommandBufferKHR)
#endif // cl_khr_command_buffer

#if defined(cl_ext_image_requirements_info)
#define __GET_IMAGE_REQUIREMENT_INFO_EXT_ERR            CL_HPP_ERR_STR_(clGetImageRequirementsInfoEXT)
#endif //cl_ext_image_requirements_info

/**
 * CL 1.2 version that uses device fission.
 */
#if CL_HPP_TARGET_OPENCL_VERSION >= 120
#define __CREATE_SUB_DEVICES_ERR            CL_HPP_ERR_STR_(clCreateSubDevices)
#else
#define __CREATE_SUB_DEVICES_ERR            CL_HPP_ERR_STR_(clCreateSubDevicesEXT)
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120

/**
 * Deprecated APIs for 1.2
 */
#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
#define __ENQUEUE_MARKER_ERR                CL_HPP_ERR_STR_(clEnqueueMarker)
#define __ENQUEUE_WAIT_FOR_EVENTS_ERR       CL_HPP_ERR_STR_(clEnqueueWaitForEvents)
#define __ENQUEUE_BARRIER_ERR               CL_HPP_ERR_STR_(clEnqueueBarrier)
#define __UNLOAD_COMPILER_ERR               CL_HPP_ERR_STR_(clUnloadCompiler)
#define __CREATE_GL_TEXTURE_2D_ERR          CL_HPP_ERR_STR_(clCreateFromGLTexture2D)
#define __CREATE_GL_TEXTURE_3D_ERR          CL_HPP_ERR_STR_(clCreateFromGLTexture3D)
#define __CREATE_IMAGE2D_ERR                CL_HPP_ERR_STR_(clCreateImage2D)
#define __CREATE_IMAGE3D_ERR                CL_HPP_ERR_STR_(clCreateImage3D)
#endif // #if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)

/**
 * Deprecated APIs for 2.0
 */
#if defined(CL_USE_DEPRECATED_OPENCL_1_2_APIS)
#define __CREATE_COMMAND_QUEUE_ERR          CL_HPP_ERR_STR_(clCreateCommandQueue)
#define __ENQUEUE_TASK_ERR                  CL_HPP_ERR_STR_(clEnqueueTask)
#define __CREATE_SAMPLER_ERR                CL_HPP_ERR_STR_(clCreateSampler)
#endif // #if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)

/**
 * CL 1.2 marker and barrier commands
 */
#if CL_HPP_TARGET_OPENCL_VERSION >= 120
#define __ENQUEUE_MARKER_WAIT_LIST_ERR                CL_HPP_ERR_STR_(clEnqueueMarkerWithWaitList)
#define __ENQUEUE_BARRIER_WAIT_LIST_ERR               CL_HPP_ERR_STR_(clEnqueueBarrierWithWaitList)
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120

#if CL_HPP_TARGET_OPENCL_VERSION >= 210
#define __CLONE_KERNEL_ERR     CL_HPP_ERR_STR_(clCloneKernel)
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 210

#endif // CL_HPP_USER_OVERRIDE_ERROR_STRINGS
//! \endcond

#ifdef cl_khr_external_memory
CL_HPP_CREATE_CL_EXT_FCN_PTR_ALIAS_(clEnqueueAcquireExternalMemObjectsKHR);
CL_HPP_CREATE_CL_EXT_FCN_PTR_ALIAS_(clEnqueueReleaseExternalMemObjectsKHR);

CL_HPP_DEFINE_STATIC_MEMBER_ PFN_clEnqueueAcquireExternalMemObjectsKHR pfn_clEnqueueAcquireExternalMemObjectsKHR = nullptr;
CL_HPP_DEFINE_STATIC_MEMBER_ PFN_clEnqueueReleaseExternalMemObjectsKHR pfn_clEnqueueReleaseExternalMemObjectsKHR = nullptr;
#endif // cl_khr_external_memory

#ifdef cl_khr_semaphore
CL_HPP_CREATE_CL_EXT_FCN_PTR_ALIAS_(clCreateSemaphoreWithPropertiesKHR);
CL_HPP_CREATE_CL_EXT_FCN_PTR_ALIAS_(clReleaseSemaphoreKHR);
CL_HPP_CREATE_CL_EXT_FCN_PTR_ALIAS_(clRetainSemaphoreKHR);
CL_HPP_CREATE_CL_EXT_FCN_PTR_ALIAS_(clEnqueueWaitSemaphoresKHR);
CL_HPP_CREATE_CL_EXT_FCN_PTR_ALIAS_(clEnqueueSignalSemaphoresKHR);
CL_HPP_CREATE_CL_EXT_FCN_PTR_ALIAS_(clGetSemaphoreInfoKHR);

CL_HPP_DEFINE_STATIC_MEMBER_ PFN_clCreateSemaphoreWithPropertiesKHR pfn_clCreateSemaphoreWithPropertiesKHR  = nullptr;
CL_HPP_DEFINE_STATIC_MEMBER_ PFN_clReleaseSemaphoreKHR              pfn_clReleaseSemaphoreKHR               = nullptr;
CL_HPP_DEFINE_STATIC_MEMBER_ PFN_clRetainSemaphoreKHR               pfn_clRetainSemaphoreKHR                = nullptr;
CL_HPP_DEFINE_STATIC_MEMBER_ PFN_clEnqueueWaitSemaphoresKHR         pfn_clEnqueueWaitSemaphoresKHR          = nullptr;
CL_HPP_DEFINE_STATIC_MEMBER_ PFN_clEnqueueSignalSemaphoresKHR       pfn_clEnqueueSignalSemaphoresKHR        = nullptr;
CL_HPP_DEFINE_STATIC_MEMBER_ PFN_clGetSemaphoreInfoKHR              pfn_clGetSemaphoreInfoKHR               = nullptr;
#endif // cl_khr_semaphore

#ifdef cl_khr_external_semaphore
CL_HPP_CREATE_CL_EXT_FCN_PTR_ALIAS_(clGetSemaphoreHandleForTypeKHR);
CL_HPP_DEFINE_STATIC_MEMBER_ PFN_clGetSemaphoreHandleForTypeKHR     pfn_clGetSemaphoreHandleForTypeKHR      = nullptr;
#endif // cl_khr_external_semaphore

#if defined(cl_khr_command_buffer)
CL_HPP_CREATE_CL_EXT_FCN_PTR_ALIAS_(clCreateCommandBufferKHR);
CL_HPP_CREATE_CL_EXT_FCN_PTR_ALIAS_(clFinalizeCommandBufferKHR);
CL_HPP_CREATE_CL_EXT_FCN_PTR_ALIAS_(clRetainCommandBufferKHR);
CL_HPP_CREATE_CL_EXT_FCN_PTR_ALIAS_(clReleaseCommandBufferKHR);
CL_HPP_CREATE_CL_EXT_FCN_PTR_ALIAS_(clGetCommandBufferInfoKHR);
CL_HPP_CREATE_CL_EXT_FCN_PTR_ALIAS_(clEnqueueCommandBufferKHR);
CL_HPP_CREATE_CL_EXT_FCN_PTR_ALIAS_(clCommandBarrierWithWaitListKHR);
CL_HPP_CREATE_CL_EXT_FCN_PTR_ALIAS_(clCommandCopyBufferKHR);
CL_HPP_CREATE_CL_EXT_FCN_PTR_ALIAS_(clCommandCopyBufferRectKHR);
CL_HPP_CREATE_CL_EXT_FCN_PTR_ALIAS_(clCommandCopyBufferToImageKHR);
CL_HPP_CREATE_CL_EXT_FCN_PTR_ALIAS_(clCommandCopyImageKHR);
CL_HPP_CREATE_CL_EXT_FCN_PTR_ALIAS_(clCommandCopyImageToBufferKHR);
CL_HPP_CREATE_CL_EXT_FCN_PTR_ALIAS_(clCommandFillBufferKHR);
CL_HPP_CREATE_CL_EXT_FCN_PTR_ALIAS_(clCommandFillImageKHR);
CL_HPP_CREATE_CL_EXT_FCN_PTR_ALIAS_(clCommandNDRangeKernelKHR);

CL_HPP_DEFINE_STATIC_MEMBER_ PFN_clCreateCommandBufferKHR pfn_clCreateCommandBufferKHR               = nullptr;
CL_HPP_DEFINE_STATIC_MEMBER_ PFN_clFinalizeCommandBufferKHR pfn_clFinalizeCommandBufferKHR           = nullptr;
CL_HPP_DEFINE_STATIC_MEMBER_ PFN_clRetainCommandBufferKHR pfn_clRetainCommandBufferKHR               = nullptr;
CL_HPP_DEFINE_STATIC_MEMBER_ PFN_clReleaseCommandBufferKHR pfn_clReleaseCommandBufferKHR             = nullptr;
CL_HPP_DEFINE_STATIC_MEMBER_ PFN_clGetCommandBufferInfoKHR pfn_clGetCommandBufferInfoKHR             = nullptr;
CL_HPP_DEFINE_STATIC_MEMBER_ PFN_clEnqueueCommandBufferKHR pfn_clEnqueueCommandBufferKHR             = nullptr;
CL_HPP_DEFINE_STATIC_MEMBER_ PFN_clCommandBarrierWithWaitListKHR pfn_clCommandBarrierWithWaitListKHR = nullptr;
CL_HPP_DEFINE_STATIC_MEMBER_ PFN_clCommandCopyBufferKHR pfn_clCommandCopyBufferKHR                   = nullptr;
CL_HPP_DEFINE_STATIC_MEMBER_ PFN_clCommandCopyBufferRectKHR pfn_clCommandCopyBufferRectKHR           = nullptr;
CL_HPP_DEFINE_STATIC_MEMBER_ PFN_clCommandCopyBufferToImageKHR pfn_clCommandCopyBufferToImageKHR     = nullptr;
CL_HPP_DEFINE_STATIC_MEMBER_ PFN_clCommandCopyImageKHR pfn_clCommandCopyImageKHR                     = nullptr;
CL_HPP_DEFINE_STATIC_MEMBER_ PFN_clCommandCopyImageToBufferKHR pfn_clCommandCopyImageToBufferKHR     = nullptr;
CL_HPP_DEFINE_STATIC_MEMBER_ PFN_clCommandFillBufferKHR pfn_clCommandFillBufferKHR                   = nullptr;
CL_HPP_DEFINE_STATIC_MEMBER_ PFN_clCommandFillImageKHR pfn_clCommandFillImageKHR                     = nullptr;
CL_HPP_DEFINE_STATIC_MEMBER_ PFN_clCommandNDRangeKernelKHR pfn_clCommandNDRangeKernelKHR             = nullptr;
#endif /* cl_khr_command_buffer */

#if defined(cl_khr_command_buffer_mutable_dispatch)
CL_HPP_CREATE_CL_EXT_FCN_PTR_ALIAS_(clUpdateMutableCommandsKHR);
CL_HPP_CREATE_CL_EXT_FCN_PTR_ALIAS_(clGetMutableCommandInfoKHR);

CL_HPP_DEFINE_STATIC_MEMBER_ PFN_clUpdateMutableCommandsKHR pfn_clUpdateMutableCommandsKHR           = nullptr;
CL_HPP_DEFINE_STATIC_MEMBER_ PFN_clGetMutableCommandInfoKHR pfn_clGetMutableCommandInfoKHR           = nullptr;
#endif /* cl_khr_command_buffer_mutable_dispatch */

#if defined(cl_ext_image_requirements_info)
CL_HPP_CREATE_CL_EXT_FCN_PTR_ALIAS_(clGetImageRequirementsInfoEXT);
CL_HPP_DEFINE_STATIC_MEMBER_ PFN_clGetImageRequirementsInfoEXT pfn_clGetImageRequirementsInfoEXT  = nullptr;
#endif

#if defined(cl_ext_device_fission)
CL_HPP_CREATE_CL_EXT_FCN_PTR_ALIAS_(clCreateSubDevicesEXT);
CL_HPP_DEFINE_STATIC_MEMBER_ PFN_clCreateSubDevicesEXT
    pfn_clCreateSubDevicesEXT = nullptr;
#endif

namespace detail {

// Generic getInfoHelper. The final parameter is used to guide overload
// resolution: the actual parameter passed is an int, which makes this
// a worse conversion sequence than a specialization that declares the
// parameter as an int.
template<typename Functor, typename T>
inline cl_int getInfoHelper(Functor f, cl_uint name, T* param, long)
{
    return f(name, sizeof(T), param, nullptr);
}

// Specialized for getInfo<CL_PROGRAM_BINARIES>
// Assumes that the output vector was correctly resized on the way in
template <typename Func>
inline cl_int getInfoHelper(Func f, cl_uint name, vector<vector<unsigned char>>* param, int)
{
    if (name != CL_PROGRAM_BINARIES) {
        return CL_INVALID_VALUE;
    }
    if (param) {
        // Create array of pointers, calculate total size and pass pointer array in
        size_type numBinaries = param->size();
        vector<unsigned char*> binariesPointers(numBinaries);

        for (size_type i = 0; i < numBinaries; ++i)
        {
            binariesPointers[i] = (*param)[i].data();
        }

        cl_int err = f(name, numBinaries * sizeof(unsigned char*), binariesPointers.data(), nullptr);

        if (err != CL_SUCCESS) {
            return err;
        }
    }

    return CL_SUCCESS;
}

// Specialized getInfoHelper for vector params
template <typename Func, typename T>
inline cl_int getInfoHelper(Func f, cl_uint name, vector<T>* param, long)
{
    size_type required;
    cl_int err = f(name, 0, nullptr, &required);
    if (err != CL_SUCCESS) {
        return err;
    }
    const size_type elements = required / sizeof(T);

    // Temporary to avoid changing param on an error
    vector<T> localData(elements);
    err = f(name, required, localData.data(), nullptr);
    if (err != CL_SUCCESS) {
        return err;
    }
    if (param) {
        *param = std::move(localData);
    }

    return CL_SUCCESS;
}

/* Specialization for reference-counted types. This depends on the
 * existence of Wrapper<T>::cl_type, and none of the other types having the
 * cl_type member. Note that simplify specifying the parameter as Wrapper<T>
 * does not work, because when using a derived type (e.g. Context) the generic
 * template will provide a better match.
 */
template <typename Func, typename T>
inline cl_int getInfoHelper(
    Func f, cl_uint name, vector<T>* param, int, typename T::cl_type = 0)
{
    size_type required;
    cl_int err = f(name, 0, nullptr, &required);
    if (err != CL_SUCCESS) {
        return err;
    }

    const size_type elements = required / sizeof(typename T::cl_type);

    vector<typename T::cl_type> value(elements);
    err = f(name, required, value.data(), nullptr);
    if (err != CL_SUCCESS) {
        return err;
    }

    if (param) {
        // Assign to convert CL type to T for each element
        param->resize(elements);

        // Assign to param, constructing with retain behaviour
        // to correctly capture each underlying CL object
        for (size_type i = 0; i < elements; i++) {
            (*param)[i] = T(value[i], true);
        }
    }
    return CL_SUCCESS;
}

// Specialized GetInfoHelper for string params
template <typename Func>
inline cl_int getInfoHelper(Func f, cl_uint name, string* param, long)
{
    size_type required;
    cl_int err = f(name, 0, nullptr, &required);
    if (err != CL_SUCCESS) {
        return err;
    }

    // std::string has a constant data member
    // a char vector does not
    if (required > 0) {
        vector<char> value(required);
        err = f(name, required, value.data(), nullptr);
        if (err != CL_SUCCESS) {
            return err;
        }
        if (param) {
            param->assign(value.begin(), value.end() - 1);
        }
    }
    else if (param) {
        param->assign("");
    }
    return CL_SUCCESS;
}

// Specialized GetInfoHelper for clsize_t params
template <typename Func, size_type N>
inline cl_int getInfoHelper(Func f, cl_uint name, array<size_type, N>* param, long)
{
    size_type required;
    cl_int err = f(name, 0, nullptr, &required);
    if (err != CL_SUCCESS) {
        return err;
    }

    size_type elements = required / sizeof(size_type);
    vector<size_type> value(elements, 0);

    err = f(name, required, value.data(), nullptr);
    if (err != CL_SUCCESS) {
        return err;
    }
    
    // Bound the copy with N to prevent overruns
    // if passed N > than the amount copied
    if (elements > N) {
        elements = N;
    }
    for (size_type i = 0; i < elements; ++i) {
        (*param)[i] = value[i];
    }

    return CL_SUCCESS;
}

template<typename T> struct ReferenceHandler;

/* Specialization for reference-counted types. This depends on the
 * existence of Wrapper<T>::cl_type, and none of the other types having the
 * cl_type member. Note that simplify specifying the parameter as Wrapper<T>
 * does not work, because when using a derived type (e.g. Context) the generic
 * template will provide a better match.
 */
template<typename Func, typename T>
inline cl_int getInfoHelper(Func f, cl_uint name, T* param, int, typename T::cl_type = 0)
{
    typename T::cl_type value;
    cl_int err = f(name, sizeof(value), &value, nullptr);
    if (err != CL_SUCCESS) {
        return err;
    }
    *param = value;
    if (value != nullptr)
    {
        err = param->retain();
        if (err != CL_SUCCESS) {
            return err;
        }
    }
    return CL_SUCCESS;
}

#define CL_HPP_PARAM_NAME_INFO_1_0_(F) \
    F(cl_platform_info, CL_PLATFORM_PROFILE, string) \
    F(cl_platform_info, CL_PLATFORM_VERSION, string) \
    F(cl_platform_info, CL_PLATFORM_NAME, string) \
    F(cl_platform_info, CL_PLATFORM_VENDOR, string) \
    F(cl_platform_info, CL_PLATFORM_EXTENSIONS, string) \
    \
    F(cl_device_info, CL_DEVICE_TYPE, cl_device_type) \
    F(cl_device_info, CL_DEVICE_VENDOR_ID, cl_uint) \
    F(cl_device_info, CL_DEVICE_MAX_COMPUTE_UNITS, cl_uint) \
    F(cl_device_info, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, cl_uint) \
    F(cl_device_info, CL_DEVICE_MAX_WORK_GROUP_SIZE, size_type) \
    F(cl_device_info, CL_DEVICE_MAX_WORK_ITEM_SIZES, cl::vector<size_type>) \
    F(cl_device_info, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, cl_uint) \
    F(cl_device_info, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, cl_uint) \
    F(cl_device_info, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, cl_uint) \
    F(cl_device_info, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, cl_uint) \
    F(cl_device_info, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, cl_uint) \
    F(cl_device_info, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, cl_uint) \
    F(cl_device_info, CL_DEVICE_MAX_CLOCK_FREQUENCY, cl_uint) \
    F(cl_device_info, CL_DEVICE_ADDRESS_BITS, cl_uint) \
    F(cl_device_info, CL_DEVICE_MAX_READ_IMAGE_ARGS, cl_uint) \
    F(cl_device_info, CL_DEVICE_MAX_WRITE_IMAGE_ARGS, cl_uint) \
    F(cl_device_info, CL_DEVICE_MAX_MEM_ALLOC_SIZE, cl_ulong) \
    F(cl_device_info, CL_DEVICE_IMAGE2D_MAX_WIDTH, size_type) \
    F(cl_device_info, CL_DEVICE_IMAGE2D_MAX_HEIGHT, size_type) \
    F(cl_device_info, CL_DEVICE_IMAGE3D_MAX_WIDTH, size_type) \
    F(cl_device_info, CL_DEVICE_IMAGE3D_MAX_HEIGHT, size_type) \
    F(cl_device_info, CL_DEVICE_IMAGE3D_MAX_DEPTH, size_type) \
    F(cl_device_info, CL_DEVICE_IMAGE_SUPPORT, cl_bool) \
    F(cl_device_info, CL_DEVICE_MAX_PARAMETER_SIZE, size_type) \
    F(cl_device_info, CL_DEVICE_MAX_SAMPLERS, cl_uint) \
    F(cl_device_info, CL_DEVICE_MEM_BASE_ADDR_ALIGN, cl_uint) \
    F(cl_device_info, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, cl_uint) \
    F(cl_device_info, CL_DEVICE_SINGLE_FP_CONFIG, cl_device_fp_config) \
    F(cl_device_info, CL_DEVICE_DOUBLE_FP_CONFIG, cl_device_fp_config) \
    F(cl_device_info, CL_DEVICE_HALF_FP_CONFIG, cl_device_fp_config) \
    F(cl_device_info, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, cl_device_mem_cache_type) \
    F(cl_device_info, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, cl_uint)\
    F(cl_device_info, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, cl_ulong) \
    F(cl_device_info, CL_DEVICE_GLOBAL_MEM_SIZE, cl_ulong) \
    F(cl_device_info, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, cl_ulong) \
    F(cl_device_info, CL_DEVICE_MAX_CONSTANT_ARGS, cl_uint) \
    F(cl_device_info, CL_DEVICE_LOCAL_MEM_TYPE, cl_device_local_mem_type) \
    F(cl_device_info, CL_DEVICE_LOCAL_MEM_SIZE, cl_ulong) \
    F(cl_device_info, CL_DEVICE_ERROR_CORRECTION_SUPPORT, cl_bool) \
    F(cl_device_info, CL_DEVICE_PROFILING_TIMER_RESOLUTION, size_type) \
    F(cl_device_info, CL_DEVICE_ENDIAN_LITTLE, cl_bool) \
    F(cl_device_info, CL_DEVICE_AVAILABLE, cl_bool) \
    F(cl_device_info, CL_DEVICE_COMPILER_AVAILABLE, cl_bool) \
    F(cl_device_info, CL_DEVICE_EXECUTION_CAPABILITIES, cl_device_exec_capabilities) \
    F(cl_device_info, CL_DEVICE_PLATFORM, cl::Platform) \
    F(cl_device_info, CL_DEVICE_NAME, string) \
    F(cl_device_info, CL_DEVICE_VENDOR, string) \
    F(cl_device_info, CL_DRIVER_VERSION, string) \
    F(cl_device_info, CL_DEVICE_PROFILE, string) \
    F(cl_device_info, CL_DEVICE_VERSION, string) \
    F(cl_device_info, CL_DEVICE_EXTENSIONS, string) \
    \
    F(cl_context_info, CL_CONTEXT_REFERENCE_COUNT, cl_uint) \
    F(cl_context_info, CL_CONTEXT_DEVICES, cl::vector<Device>) \
    F(cl_context_info, CL_CONTEXT_PROPERTIES, cl::vector<cl_context_properties>) \
    \
    F(cl_event_info, CL_EVENT_COMMAND_QUEUE, cl::CommandQueue) \
    F(cl_event_info, CL_EVENT_COMMAND_TYPE, cl_command_type) \
    F(cl_event_info, CL_EVENT_REFERENCE_COUNT, cl_uint) \
    F(cl_event_info, CL_EVENT_COMMAND_EXECUTION_STATUS, cl_int) \
    \
    F(cl_profiling_info, CL_PROFILING_COMMAND_QUEUED, cl_ulong) \
    F(cl_profiling_info, CL_PROFILING_COMMAND_SUBMIT, cl_ulong) \
    F(cl_profiling_info, CL_PROFILING_COMMAND_START, cl_ulong) \
    F(cl_profiling_info, CL_PROFILING_COMMAND_END, cl_ulong) \
    \
    F(cl_mem_info, CL_MEM_TYPE, cl_mem_object_type) \
    F(cl_mem_info, CL_MEM_FLAGS, cl_mem_flags) \
    F(cl_mem_info, CL_MEM_SIZE, size_type) \
    F(cl_mem_info, CL_MEM_HOST_PTR, void*) \
    F(cl_mem_info, CL_MEM_MAP_COUNT, cl_uint) \
    F(cl_mem_info, CL_MEM_REFERENCE_COUNT, cl_uint) \
    F(cl_mem_info, CL_MEM_CONTEXT, cl::Context) \
    \
    F(cl_image_info, CL_IMAGE_FORMAT, cl_image_format) \
    F(cl_image_info, CL_IMAGE_ELEMENT_SIZE, size_type) \
    F(cl_image_info, CL_IMAGE_ROW_PITCH, size_type) \
    F(cl_image_info, CL_IMAGE_SLICE_PITCH, size_type) \
    F(cl_image_info, CL_IMAGE_WIDTH, size_type) \
    F(cl_image_info, CL_IMAGE_HEIGHT, size_type) \
    F(cl_image_info, CL_IMAGE_DEPTH, size_type) \
    \
    F(cl_sampler_info, CL_SAMPLER_REFERENCE_COUNT, cl_uint) \
    F(cl_sampler_info, CL_SAMPLER_CONTEXT, cl::Context) \
    F(cl_sampler_info, CL_SAMPLER_NORMALIZED_COORDS, cl_bool) \
    F(cl_sampler_info, CL_SAMPLER_ADDRESSING_MODE, cl_addressing_mode) \
    F(cl_sampler_info, CL_SAMPLER_FILTER_MODE, cl_filter_mode) \
    \
    F(cl_program_info, CL_PROGRAM_REFERENCE_COUNT, cl_uint) \
    F(cl_program_info, CL_PROGRAM_CONTEXT, cl::Context) \
    F(cl_program_info, CL_PROGRAM_NUM_DEVICES, cl_uint) \
    F(cl_program_info, CL_PROGRAM_DEVICES, cl::vector<Device>) \
    F(cl_program_info, CL_PROGRAM_SOURCE, string) \
    F(cl_program_info, CL_PROGRAM_BINARY_SIZES, cl::vector<size_type>) \
    F(cl_program_info, CL_PROGRAM_BINARIES, cl::vector<cl::vector<unsigned char>>) \
    \
    F(cl_program_build_info, CL_PROGRAM_BUILD_STATUS, cl_build_status) \
    F(cl_program_build_info, CL_PROGRAM_BUILD_OPTIONS, string) \
    F(cl_program_build_info, CL_PROGRAM_BUILD_LOG, string) \
    \
    F(cl_kernel_info, CL_KERNEL_FUNCTION_NAME, string) \
    F(cl_kernel_info, CL_KERNEL_NUM_ARGS, cl_uint) \
    F(cl_kernel_info, CL_KERNEL_REFERENCE_COUNT, cl_uint) \
    F(cl_kernel_info, CL_KERNEL_CONTEXT, cl::Context) \
    F(cl_kernel_info, CL_KERNEL_PROGRAM, cl::Program) \
    \
    F(cl_kernel_work_group_info, CL_KERNEL_WORK_GROUP_SIZE, size_type) \
    F(cl_kernel_work_group_info, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, cl::detail::size_t_array) \
    F(cl_kernel_work_group_info, CL_KERNEL_LOCAL_MEM_SIZE, cl_ulong) \
    \
    F(cl_command_queue_info, CL_QUEUE_CONTEXT, cl::Context) \
    F(cl_command_queue_info, CL_QUEUE_DEVICE, cl::Device) \
    F(cl_command_queue_info, CL_QUEUE_REFERENCE_COUNT, cl_uint) \
    F(cl_command_queue_info, CL_QUEUE_PROPERTIES, cl_command_queue_properties)


#define CL_HPP_PARAM_NAME_INFO_1_1_(F) \
    F(cl_context_info, CL_CONTEXT_NUM_DEVICES, cl_uint)\
    F(cl_device_info, CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF, cl_uint) \
    F(cl_device_info, CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR, cl_uint) \
    F(cl_device_info, CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT, cl_uint) \
    F(cl_device_info, CL_DEVICE_NATIVE_VECTOR_WIDTH_INT, cl_uint) \
    F(cl_device_info, CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG, cl_uint) \
    F(cl_device_info, CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, cl_uint) \
    F(cl_device_info, CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, cl_uint) \
    F(cl_device_info, CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF, cl_uint) \
    F(cl_device_info, CL_DEVICE_OPENCL_C_VERSION, string) \
    \
    F(cl_mem_info, CL_MEM_ASSOCIATED_MEMOBJECT, cl::Memory) \
    F(cl_mem_info, CL_MEM_OFFSET, size_type) \
    \
    F(cl_kernel_work_group_info, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, size_type) \
    F(cl_kernel_work_group_info, CL_KERNEL_PRIVATE_MEM_SIZE, cl_ulong) \
    \
    F(cl_event_info, CL_EVENT_CONTEXT, cl::Context)

#define CL_HPP_PARAM_NAME_INFO_1_2_(F) \
    F(cl_program_info, CL_PROGRAM_NUM_KERNELS, size_type) \
    F(cl_program_info, CL_PROGRAM_KERNEL_NAMES, string) \
    \
    F(cl_program_build_info, CL_PROGRAM_BINARY_TYPE, cl_program_binary_type) \
    \
    F(cl_kernel_info, CL_KERNEL_ATTRIBUTES, string) \
    \
    F(cl_kernel_arg_info, CL_KERNEL_ARG_ADDRESS_QUALIFIER, cl_kernel_arg_address_qualifier) \
    F(cl_kernel_arg_info, CL_KERNEL_ARG_ACCESS_QUALIFIER, cl_kernel_arg_access_qualifier) \
    F(cl_kernel_arg_info, CL_KERNEL_ARG_TYPE_NAME, string) \
    F(cl_kernel_arg_info, CL_KERNEL_ARG_NAME, string) \
    F(cl_kernel_arg_info, CL_KERNEL_ARG_TYPE_QUALIFIER, cl_kernel_arg_type_qualifier) \
    \
    F(cl_kernel_work_group_info, CL_KERNEL_GLOBAL_WORK_SIZE, cl::detail::size_t_array) \
    \
    F(cl_device_info, CL_DEVICE_LINKER_AVAILABLE, cl_bool) \
    F(cl_device_info, CL_DEVICE_IMAGE_MAX_BUFFER_SIZE, size_type) \
    F(cl_device_info, CL_DEVICE_IMAGE_MAX_ARRAY_SIZE, size_type) \
    F(cl_device_info, CL_DEVICE_PARENT_DEVICE, cl::Device) \
    F(cl_device_info, CL_DEVICE_PARTITION_MAX_SUB_DEVICES, cl_uint) \
    F(cl_device_info, CL_DEVICE_PARTITION_PROPERTIES, cl::vector<cl_device_partition_property>) \
    F(cl_device_info, CL_DEVICE_PARTITION_TYPE, cl::vector<cl_device_partition_property>)  \
    F(cl_device_info, CL_DEVICE_REFERENCE_COUNT, cl_uint) \
    F(cl_device_info, CL_DEVICE_PREFERRED_INTEROP_USER_SYNC, cl_bool) \
    F(cl_device_info, CL_DEVICE_PARTITION_AFFINITY_DOMAIN, cl_device_affinity_domain) \
    F(cl_device_info, CL_DEVICE_BUILT_IN_KERNELS, string) \
    F(cl_device_info, CL_DEVICE_PRINTF_BUFFER_SIZE, size_type) \
    \
    F(cl_image_info, CL_IMAGE_ARRAY_SIZE, size_type) \
    F(cl_image_info, CL_IMAGE_NUM_MIP_LEVELS, cl_uint) \
    F(cl_image_info, CL_IMAGE_NUM_SAMPLES, cl_uint)

#define CL_HPP_PARAM_NAME_INFO_2_0_(F) \
    F(cl_device_info, CL_DEVICE_QUEUE_ON_HOST_PROPERTIES, cl_command_queue_properties) \
    F(cl_device_info, CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES, cl_command_queue_properties) \
    F(cl_device_info, CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE, cl_uint) \
    F(cl_device_info, CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE, cl_uint) \
    F(cl_device_info, CL_DEVICE_MAX_ON_DEVICE_QUEUES, cl_uint) \
    F(cl_device_info, CL_DEVICE_MAX_ON_DEVICE_EVENTS, cl_uint) \
    F(cl_device_info, CL_DEVICE_MAX_PIPE_ARGS, cl_uint) \
    F(cl_device_info, CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS, cl_uint) \
    F(cl_device_info, CL_DEVICE_PIPE_MAX_PACKET_SIZE, cl_uint) \
    F(cl_device_info, CL_DEVICE_SVM_CAPABILITIES, cl_device_svm_capabilities) \
    F(cl_device_info, CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT, cl_uint) \
    F(cl_device_info, CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT, cl_uint) \
    F(cl_device_info, CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT, cl_uint) \
    F(cl_device_info, CL_DEVICE_IMAGE_PITCH_ALIGNMENT, cl_uint) \
    F(cl_device_info, CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT, cl_uint) \
    F(cl_device_info, CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS, cl_uint ) \
    F(cl_device_info, CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE, size_type ) \
    F(cl_device_info, CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE, size_type ) \
    F(cl_profiling_info, CL_PROFILING_COMMAND_COMPLETE, cl_ulong) \
    F(cl_kernel_exec_info, CL_KERNEL_EXEC_INFO_SVM_FINE_GRAIN_SYSTEM, cl_bool) \
    F(cl_kernel_exec_info, CL_KERNEL_EXEC_INFO_SVM_PTRS, void**) \
    F(cl_command_queue_info, CL_QUEUE_SIZE, cl_uint) \
    F(cl_mem_info, CL_MEM_USES_SVM_POINTER, cl_bool) \
    F(cl_program_build_info, CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE, size_type) \
    F(cl_pipe_info, CL_PIPE_PACKET_SIZE, cl_uint) \
    F(cl_pipe_info, CL_PIPE_MAX_PACKETS, cl_uint)

#define CL_HPP_PARAM_NAME_INFO_SUBGROUP_KHR_(F) \
    F(cl_kernel_sub_group_info, CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE_KHR, size_type) \
    F(cl_kernel_sub_group_info, CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE_KHR, size_type)

#define CL_HPP_PARAM_NAME_INFO_IL_KHR_(F) \
    F(cl_device_info, CL_DEVICE_IL_VERSION_KHR, string) \
    F(cl_program_info, CL_PROGRAM_IL_KHR, cl::vector<unsigned char>)

#define CL_HPP_PARAM_NAME_INFO_2_1_(F) \
    F(cl_platform_info, CL_PLATFORM_HOST_TIMER_RESOLUTION, cl_ulong) \
    F(cl_program_info, CL_PROGRAM_IL, cl::vector<unsigned char>) \
    F(cl_device_info, CL_DEVICE_MAX_NUM_SUB_GROUPS, cl_uint) \
    F(cl_device_info, CL_DEVICE_IL_VERSION, string) \
    F(cl_device_info, CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS, cl_bool) \
    F(cl_command_queue_info, CL_QUEUE_DEVICE_DEFAULT, cl::DeviceCommandQueue) \
    F(cl_kernel_sub_group_info, CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE, size_type) \
    F(cl_kernel_sub_group_info, CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE, size_type) \
    F(cl_kernel_sub_group_info, CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT, cl::detail::size_t_array) \
    F(cl_kernel_sub_group_info, CL_KERNEL_MAX_NUM_SUB_GROUPS, size_type) \
    F(cl_kernel_sub_group_info, CL_KERNEL_COMPILE_NUM_SUB_GROUPS, size_type)

#define CL_HPP_PARAM_NAME_INFO_2_2_(F) \
    F(cl_program_info, CL_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT, cl_bool) \
    F(cl_program_info, CL_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT, cl_bool)

#define CL_HPP_PARAM_NAME_DEVICE_FISSION_EXT_(F) \
    F(cl_device_info, CL_DEVICE_PARENT_DEVICE_EXT, cl::Device) \
    F(cl_device_info, CL_DEVICE_PARTITION_TYPES_EXT, cl::vector<cl_device_partition_property_ext>) \
    F(cl_device_info, CL_DEVICE_AFFINITY_DOMAINS_EXT, cl::vector<cl_device_partition_property_ext>) \
    F(cl_device_info, CL_DEVICE_REFERENCE_COUNT_EXT , cl_uint) \
    F(cl_device_info, CL_DEVICE_PARTITION_STYLE_EXT, cl::vector<cl_device_partition_property_ext>)

#define CL_HPP_PARAM_NAME_CL_KHR_EXTENDED_VERSIONING_CL3_SHARED_(F) \
    F(cl_platform_info, CL_PLATFORM_NUMERIC_VERSION_KHR, cl_version_khr) \
    F(cl_platform_info, CL_PLATFORM_EXTENSIONS_WITH_VERSION_KHR, cl::vector<cl_name_version_khr>) \
    \
    F(cl_device_info, CL_DEVICE_NUMERIC_VERSION_KHR, cl_version_khr) \
    F(cl_device_info, CL_DEVICE_EXTENSIONS_WITH_VERSION_KHR, cl::vector<cl_name_version_khr>) \
    F(cl_device_info, CL_DEVICE_ILS_WITH_VERSION_KHR, cl::vector<cl_name_version_khr>) \
    F(cl_device_info, CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION_KHR, cl::vector<cl_name_version_khr>)

#define CL_HPP_PARAM_NAME_CL_KHR_EXTENDED_VERSIONING_KHRONLY_(F) \
    F(cl_device_info, CL_DEVICE_OPENCL_C_NUMERIC_VERSION_KHR, cl_version_khr)

// Note: the query for CL_SEMAPHORE_DEVICE_HANDLE_LIST_KHR is handled specially!
#define CL_HPP_PARAM_NAME_CL_KHR_SEMAPHORE_(F) \
    F(cl_semaphore_info_khr, CL_SEMAPHORE_CONTEXT_KHR, cl::Context) \
    F(cl_semaphore_info_khr, CL_SEMAPHORE_REFERENCE_COUNT_KHR, cl_uint) \
    F(cl_semaphore_info_khr, CL_SEMAPHORE_PROPERTIES_KHR, cl::vector<cl_semaphore_properties_khr>) \
    F(cl_semaphore_info_khr, CL_SEMAPHORE_TYPE_KHR, cl_semaphore_type_khr) \
    F(cl_semaphore_info_khr, CL_SEMAPHORE_PAYLOAD_KHR, cl_semaphore_payload_khr) \
    F(cl_platform_info, CL_PLATFORM_SEMAPHORE_TYPES_KHR,  cl::vector<cl_semaphore_type_khr>) \
    F(cl_device_info, CL_DEVICE_SEMAPHORE_TYPES_KHR,      cl::vector<cl_semaphore_type_khr>) \

#define CL_HPP_PARAM_NAME_CL_KHR_EXTERNAL_MEMORY_(F) \
    F(cl_device_info, CL_DEVICE_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR, cl::vector<cl::ExternalMemoryType>) \
    F(cl_platform_info, CL_PLATFORM_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR, cl::vector<cl::ExternalMemoryType>)

#define CL_HPP_PARAM_NAME_CL_KHR_EXTERNAL_SEMAPHORE_(F) \
    F(cl_platform_info, CL_PLATFORM_SEMAPHORE_IMPORT_HANDLE_TYPES_KHR,  cl::vector<cl_external_semaphore_handle_type_khr>) \
    F(cl_platform_info, CL_PLATFORM_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR,  cl::vector<cl_external_semaphore_handle_type_khr>) \
    F(cl_device_info, CL_DEVICE_SEMAPHORE_IMPORT_HANDLE_TYPES_KHR,      cl::vector<cl_external_semaphore_handle_type_khr>) \
    F(cl_device_info, CL_DEVICE_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR,      cl::vector<cl_external_semaphore_handle_type_khr>) \
    F(cl_semaphore_info_khr, CL_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR,      cl::vector<cl_external_semaphore_handle_type_khr>) \

#define CL_HPP_PARAM_NAME_CL_KHR_EXTERNAL_SEMAPHORE_OPAQUE_FD_EXT(F) \
    F(cl_external_semaphore_handle_type_khr, CL_SEMAPHORE_HANDLE_OPAQUE_FD_KHR, int) \

#define CL_HPP_PARAM_NAME_CL_KHR_EXTERNAL_SEMAPHORE_SYNC_FD_EXT(F) \
    F(cl_external_semaphore_handle_type_khr, CL_SEMAPHORE_HANDLE_SYNC_FD_KHR, int) \

#define CL_HPP_PARAM_NAME_CL_KHR_EXTERNAL_SEMAPHORE_WIN32_EXT(F) \
    F(cl_external_semaphore_handle_type_khr, CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KHR, void*) \
    F(cl_external_semaphore_handle_type_khr, CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KMT_KHR, void*) \

#define CL_HPP_PARAM_NAME_INFO_3_0_(F) \
    F(cl_platform_info, CL_PLATFORM_NUMERIC_VERSION, cl_version) \
    F(cl_platform_info, CL_PLATFORM_EXTENSIONS_WITH_VERSION, cl::vector<cl_name_version>) \
    \
    F(cl_device_info, CL_DEVICE_NUMERIC_VERSION, cl_version) \
    F(cl_device_info, CL_DEVICE_EXTENSIONS_WITH_VERSION, cl::vector<cl_name_version>) \
    F(cl_device_info, CL_DEVICE_ILS_WITH_VERSION, cl::vector<cl_name_version>) \
    F(cl_device_info, CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION, cl::vector<cl_name_version>) \
    F(cl_device_info, CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES, cl_device_atomic_capabilities) \
    F(cl_device_info, CL_DEVICE_ATOMIC_FENCE_CAPABILITIES, cl_device_atomic_capabilities) \
    F(cl_device_info, CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT, cl_bool) \
    F(cl_device_info, CL_DEVICE_OPENCL_C_ALL_VERSIONS, cl::vector<cl_name_version>) \
    F(cl_device_info, CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, size_type) \
    F(cl_device_info, CL_DEVICE_WORK_GROUP_COLLECTIVE_FUNCTIONS_SUPPORT, cl_bool) \
    F(cl_device_info, CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT, cl_bool) \
    F(cl_device_info, CL_DEVICE_OPENCL_C_FEATURES, cl::vector<cl_name_version>) \
    F(cl_device_info, CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES, cl_device_device_enqueue_capabilities) \
    F(cl_device_info, CL_DEVICE_PIPE_SUPPORT, cl_bool) \
    F(cl_device_info, CL_DEVICE_LATEST_CONFORMANCE_VERSION_PASSED, string) \
    \
    F(cl_command_queue_info, CL_QUEUE_PROPERTIES_ARRAY, cl::vector<cl_queue_properties>) \
    F(cl_mem_info, CL_MEM_PROPERTIES, cl::vector<cl_mem_properties>) \
    F(cl_pipe_info, CL_PIPE_PROPERTIES, cl::vector<cl_pipe_properties>) \
    F(cl_sampler_info, CL_SAMPLER_PROPERTIES, cl::vector<cl_sampler_properties>) \

#define CL_HPP_PARAM_NAME_CL_IMAGE_REQUIREMENTS_EXT(F) \
    F(cl_image_requirements_info_ext, CL_IMAGE_REQUIREMENTS_ROW_PITCH_ALIGNMENT_EXT, size_type) \
    F(cl_image_requirements_info_ext, CL_IMAGE_REQUIREMENTS_BASE_ADDRESS_ALIGNMENT_EXT, size_type) \
    F(cl_image_requirements_info_ext, CL_IMAGE_REQUIREMENTS_SIZE_EXT, size_type) \
    F(cl_image_requirements_info_ext, CL_IMAGE_REQUIREMENTS_MAX_WIDTH_EXT, cl_uint) \
    F(cl_image_requirements_info_ext, CL_IMAGE_REQUIREMENTS_MAX_HEIGHT_EXT, cl_uint) \
    F(cl_image_requirements_info_ext, CL_IMAGE_REQUIREMENTS_MAX_DEPTH_EXT, cl_uint) \
    F(cl_image_requirements_info_ext, CL_IMAGE_REQUIREMENTS_MAX_ARRAY_SIZE_EXT, cl_uint) \

#define CL_HPP_PARAM_NAME_CL_IMAGE_REQUIREMENTS_SLICE_PITCH_ALIGNMENT_EXT(F) \
    F(cl_image_requirements_info_ext, CL_IMAGE_REQUIREMENTS_SLICE_PITCH_ALIGNMENT_EXT, size_type) \

#define CL_HPP_PARAM_NAME_CL_INTEL_COMMAND_QUEUE_FAMILIES_(F) \
    F(cl_device_info, CL_DEVICE_QUEUE_FAMILY_PROPERTIES_INTEL, cl::vector<cl_queue_family_properties_intel>) \
    \
    F(cl_command_queue_info, CL_QUEUE_FAMILY_INTEL, cl_uint) \
    F(cl_command_queue_info, CL_QUEUE_INDEX_INTEL, cl_uint)

#define CL_HPP_PARAM_NAME_CL_INTEL_UNIFIED_SHARED_MEMORY_(F) \
    F(cl_device_info, CL_DEVICE_HOST_MEM_CAPABILITIES_INTEL, cl_device_unified_shared_memory_capabilities_intel ) \
    F(cl_device_info, CL_DEVICE_DEVICE_MEM_CAPABILITIES_INTEL, cl_device_unified_shared_memory_capabilities_intel ) \
    F(cl_device_info, CL_DEVICE_SINGLE_DEVICE_SHARED_MEM_CAPABILITIES_INTEL, cl_device_unified_shared_memory_capabilities_intel ) \
    F(cl_device_info, CL_DEVICE_CROSS_DEVICE_SHARED_MEM_CAPABILITIES_INTEL, cl_device_unified_shared_memory_capabilities_intel ) \
    F(cl_device_info, CL_DEVICE_SHARED_SYSTEM_MEM_CAPABILITIES_INTEL, cl_device_unified_shared_memory_capabilities_intel )

template <typename enum_type, cl_int Name>
struct param_traits {};

#define CL_HPP_DECLARE_PARAM_TRAITS_(token, param_name, T) \
struct token;                                        \
template<>                                           \
struct param_traits<detail:: token,param_name>       \
{                                                    \
    enum { value = param_name };                     \
    typedef T param_type;                            \
};

CL_HPP_PARAM_NAME_INFO_1_0_(CL_HPP_DECLARE_PARAM_TRAITS_)
#if CL_HPP_TARGET_OPENCL_VERSION >= 110
CL_HPP_PARAM_NAME_INFO_1_1_(CL_HPP_DECLARE_PARAM_TRAITS_)
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
#if CL_HPP_TARGET_OPENCL_VERSION >= 120
CL_HPP_PARAM_NAME_INFO_1_2_(CL_HPP_DECLARE_PARAM_TRAITS_)
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
#if CL_HPP_TARGET_OPENCL_VERSION >= 200
CL_HPP_PARAM_NAME_INFO_2_0_(CL_HPP_DECLARE_PARAM_TRAITS_)
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
#if CL_HPP_TARGET_OPENCL_VERSION >= 210
CL_HPP_PARAM_NAME_INFO_2_1_(CL_HPP_DECLARE_PARAM_TRAITS_)
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 210
#if CL_HPP_TARGET_OPENCL_VERSION >= 220
CL_HPP_PARAM_NAME_INFO_2_2_(CL_HPP_DECLARE_PARAM_TRAITS_)
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 220
#if CL_HPP_TARGET_OPENCL_VERSION >= 300
CL_HPP_PARAM_NAME_INFO_3_0_(CL_HPP_DECLARE_PARAM_TRAITS_)
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 300

#if defined(cl_khr_subgroups) && CL_HPP_TARGET_OPENCL_VERSION < 210
CL_HPP_PARAM_NAME_INFO_SUBGROUP_KHR_(CL_HPP_DECLARE_PARAM_TRAITS_)
#endif // #if defined(cl_khr_subgroups) && CL_HPP_TARGET_OPENCL_VERSION < 210

#if defined(cl_khr_il_program) && CL_HPP_TARGET_OPENCL_VERSION < 210
CL_HPP_PARAM_NAME_INFO_IL_KHR_(CL_HPP_DECLARE_PARAM_TRAITS_)
#endif // #if defined(cl_khr_il_program) && CL_HPP_TARGET_OPENCL_VERSION < 210


// Flags deprecated in OpenCL 2.0
#define CL_HPP_PARAM_NAME_INFO_1_0_DEPRECATED_IN_2_0_(F) \
    F(cl_device_info, CL_DEVICE_QUEUE_PROPERTIES, cl_command_queue_properties)

#define CL_HPP_PARAM_NAME_INFO_1_1_DEPRECATED_IN_2_0_(F) \
    F(cl_device_info, CL_DEVICE_HOST_UNIFIED_MEMORY, cl_bool)

#define CL_HPP_PARAM_NAME_INFO_1_2_DEPRECATED_IN_2_0_(F) \
    F(cl_image_info, CL_IMAGE_BUFFER, cl::Buffer)

// Include deprecated query flags based on versions
// Only include deprecated 1.0 flags if 2.0 not active as there is an enum clash
#if CL_HPP_TARGET_OPENCL_VERSION > 100 && CL_HPP_MINIMUM_OPENCL_VERSION < 200 && CL_HPP_TARGET_OPENCL_VERSION < 200
CL_HPP_PARAM_NAME_INFO_1_0_DEPRECATED_IN_2_0_(CL_HPP_DECLARE_PARAM_TRAITS_)
#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 110
#if CL_HPP_TARGET_OPENCL_VERSION > 110 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
CL_HPP_PARAM_NAME_INFO_1_1_DEPRECATED_IN_2_0_(CL_HPP_DECLARE_PARAM_TRAITS_)
#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 120
#if CL_HPP_TARGET_OPENCL_VERSION > 120 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
CL_HPP_PARAM_NAME_INFO_1_2_DEPRECATED_IN_2_0_(CL_HPP_DECLARE_PARAM_TRAITS_)
#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 200

#if defined(cl_ext_device_fission)
CL_HPP_PARAM_NAME_DEVICE_FISSION_EXT_(CL_HPP_DECLARE_PARAM_TRAITS_)
#endif // cl_ext_device_fission

#if defined(cl_khr_extended_versioning)
#if CL_HPP_TARGET_OPENCL_VERSION < 300
CL_HPP_PARAM_NAME_CL_KHR_EXTENDED_VERSIONING_CL3_SHARED_(CL_HPP_DECLARE_PARAM_TRAITS_)
#endif // CL_HPP_TARGET_OPENCL_VERSION < 300
CL_HPP_PARAM_NAME_CL_KHR_EXTENDED_VERSIONING_KHRONLY_(CL_HPP_DECLARE_PARAM_TRAITS_)
#endif // cl_khr_extended_versioning

#if defined(cl_khr_semaphore)
CL_HPP_PARAM_NAME_CL_KHR_SEMAPHORE_(CL_HPP_DECLARE_PARAM_TRAITS_)
#if defined(CL_SEMAPHORE_DEVICE_HANDLE_LIST_KHR)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_semaphore_info_khr, CL_SEMAPHORE_DEVICE_HANDLE_LIST_KHR, cl::vector<cl::Device>)
#endif // defined(CL_SEMAPHORE_DEVICE_HANDLE_LIST_KHR)
#endif // defined(cl_khr_semaphore)

#ifdef cl_khr_external_memory
CL_HPP_PARAM_NAME_CL_KHR_EXTERNAL_MEMORY_(CL_HPP_DECLARE_PARAM_TRAITS_)
#endif // cl_khr_external_memory

#if defined(cl_khr_external_semaphore)
CL_HPP_PARAM_NAME_CL_KHR_EXTERNAL_SEMAPHORE_(CL_HPP_DECLARE_PARAM_TRAITS_)
#endif // cl_khr_external_semaphore

#if defined(cl_khr_external_semaphore_opaque_fd)
CL_HPP_PARAM_NAME_CL_KHR_EXTERNAL_SEMAPHORE_OPAQUE_FD_EXT(CL_HPP_DECLARE_PARAM_TRAITS_)
#endif // cl_khr_external_semaphore_opaque_fd
#if defined(cl_khr_external_semaphore_sync_fd)
CL_HPP_PARAM_NAME_CL_KHR_EXTERNAL_SEMAPHORE_SYNC_FD_EXT(CL_HPP_DECLARE_PARAM_TRAITS_)
#endif // cl_khr_external_semaphore_sync_fd
#if defined(cl_khr_external_semaphore_win32)
CL_HPP_PARAM_NAME_CL_KHR_EXTERNAL_SEMAPHORE_WIN32_EXT(CL_HPP_DECLARE_PARAM_TRAITS_)
#endif // cl_khr_external_semaphore_win32

#if defined(cl_khr_device_uuid)
using uuid_array = array<cl_uchar, CL_UUID_SIZE_KHR>;
using luid_array = array<cl_uchar, CL_LUID_SIZE_KHR>;
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_UUID_KHR, uuid_array)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DRIVER_UUID_KHR, uuid_array)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_LUID_VALID_KHR, cl_bool)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_LUID_KHR, luid_array)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_NODE_MASK_KHR, cl_uint)
#endif

#if defined(cl_khr_pci_bus_info)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_PCI_BUS_INFO_KHR, cl_device_pci_bus_info_khr)
#endif

// Note: some headers do not define cl_khr_image2d_from_buffer
#if CL_HPP_TARGET_OPENCL_VERSION < 200
#if defined(CL_DEVICE_IMAGE_PITCH_ALIGNMENT_KHR)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_IMAGE_PITCH_ALIGNMENT_KHR, cl_uint)
#endif
#if defined(CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT_KHR)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT_KHR, cl_uint)
#endif
#endif // CL_HPP_TARGET_OPENCL_VERSION < 200

#if defined(cl_khr_integer_dot_product)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_INTEGER_DOT_PRODUCT_CAPABILITIES_KHR, cl_device_integer_dot_product_capabilities_khr)
#if defined(CL_DEVICE_INTEGER_DOT_PRODUCT_ACCELERATION_PROPERTIES_8BIT_KHR)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_INTEGER_DOT_PRODUCT_ACCELERATION_PROPERTIES_8BIT_KHR, cl_device_integer_dot_product_acceleration_properties_khr)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_INTEGER_DOT_PRODUCT_ACCELERATION_PROPERTIES_4x8BIT_PACKED_KHR, cl_device_integer_dot_product_acceleration_properties_khr)
#endif // defined(CL_DEVICE_INTEGER_DOT_PRODUCT_ACCELERATION_PROPERTIES_8BIT_KHR)
#endif // defined(cl_khr_integer_dot_product)

#if defined(cl_ext_image_requirements_info)
CL_HPP_PARAM_NAME_CL_IMAGE_REQUIREMENTS_EXT(CL_HPP_DECLARE_PARAM_TRAITS_)
#endif // cl_ext_image_requirements_info

#if defined(cl_ext_image_from_buffer)
CL_HPP_PARAM_NAME_CL_IMAGE_REQUIREMENTS_SLICE_PITCH_ALIGNMENT_EXT(CL_HPP_DECLARE_PARAM_TRAITS_)
#endif // cl_ext_image_from_buffer

#ifdef CL_PLATFORM_ICD_SUFFIX_KHR
CL_HPP_DECLARE_PARAM_TRAITS_(cl_platform_info, CL_PLATFORM_ICD_SUFFIX_KHR, string)
#endif

#ifdef CL_DEVICE_PROFILING_TIMER_OFFSET_AMD
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_PROFILING_TIMER_OFFSET_AMD, cl_ulong)
#endif
#ifdef CL_DEVICE_GLOBAL_FREE_MEMORY_AMD
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_GLOBAL_FREE_MEMORY_AMD, vector<size_type>)
#endif
#ifdef CL_DEVICE_SIMD_PER_COMPUTE_UNIT_AMD
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_SIMD_PER_COMPUTE_UNIT_AMD, cl_uint)
#endif
#ifdef CL_DEVICE_SIMD_WIDTH_AMD
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_SIMD_WIDTH_AMD, cl_uint)
#endif
#ifdef CL_DEVICE_SIMD_INSTRUCTION_WIDTH_AMD
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_SIMD_INSTRUCTION_WIDTH_AMD, cl_uint)
#endif
#ifdef CL_DEVICE_WAVEFRONT_WIDTH_AMD
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_WAVEFRONT_WIDTH_AMD, cl_uint)
#endif
#ifdef CL_DEVICE_GLOBAL_MEM_CHANNELS_AMD
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_GLOBAL_MEM_CHANNELS_AMD, cl_uint)
#endif
#ifdef CL_DEVICE_GLOBAL_MEM_CHANNEL_BANKS_AMD
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_GLOBAL_MEM_CHANNEL_BANKS_AMD, cl_uint)
#endif
#ifdef CL_DEVICE_GLOBAL_MEM_CHANNEL_BANK_WIDTH_AMD
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_GLOBAL_MEM_CHANNEL_BANK_WIDTH_AMD, cl_uint)
#endif
#ifdef CL_DEVICE_LOCAL_MEM_SIZE_PER_COMPUTE_UNIT_AMD
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_LOCAL_MEM_SIZE_PER_COMPUTE_UNIT_AMD, cl_uint)
#endif
#ifdef CL_DEVICE_LOCAL_MEM_BANKS_AMD
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_LOCAL_MEM_BANKS_AMD, cl_uint)
#endif
#ifdef CL_DEVICE_BOARD_NAME_AMD
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_BOARD_NAME_AMD, string)
#endif

#ifdef CL_DEVICE_COMPUTE_UNITS_BITFIELD_ARM
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_COMPUTE_UNITS_BITFIELD_ARM, cl_ulong)
#endif
#ifdef CL_DEVICE_JOB_SLOTS_ARM
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_JOB_SLOTS_ARM, cl_uint)
#endif
#ifdef CL_DEVICE_SCHEDULING_CONTROLS_CAPABILITIES_ARM
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_SCHEDULING_CONTROLS_CAPABILITIES_ARM, cl_bitfield)
#endif
#ifdef CL_DEVICE_SUPPORTED_REGISTER_ALLOCATIONS_ARM
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_SUPPORTED_REGISTER_ALLOCATIONS_ARM, vector<cl_uint>)
#endif
#ifdef CL_DEVICE_MAX_WARP_COUNT_ARM
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_MAX_WARP_COUNT_ARM, cl_uint)
#endif
#ifdef CL_KERNEL_MAX_WARP_COUNT_ARM
CL_HPP_DECLARE_PARAM_TRAITS_(cl_kernel_info, CL_KERNEL_MAX_WARP_COUNT_ARM, cl_uint)
#endif
#ifdef CL_KERNEL_EXEC_INFO_WORKGROUP_BATCH_SIZE_ARM
CL_HPP_DECLARE_PARAM_TRAITS_(cl_kernel_exec_info, CL_KERNEL_EXEC_INFO_WORKGROUP_BATCH_SIZE_ARM, cl_uint)
#endif
#ifdef CL_KERNEL_EXEC_INFO_WORKGROUP_BATCH_SIZE_MODIFIER_ARM
CL_HPP_DECLARE_PARAM_TRAITS_(cl_kernel_exec_info, CL_KERNEL_EXEC_INFO_WORKGROUP_BATCH_SIZE_MODIFIER_ARM, cl_int)
#endif
#ifdef CL_KERNEL_EXEC_INFO_WARP_COUNT_LIMIT_ARM
CL_HPP_DECLARE_PARAM_TRAITS_(cl_kernel_exec_info, CL_KERNEL_EXEC_INFO_WARP_COUNT_LIMIT_ARM, cl_uint)
#endif
#ifdef CL_KERNEL_EXEC_INFO_COMPUTE_UNIT_MAX_QUEUED_BATCHES_ARM
CL_HPP_DECLARE_PARAM_TRAITS_(cl_kernel_exec_info, CL_KERNEL_EXEC_INFO_COMPUTE_UNIT_MAX_QUEUED_BATCHES_ARM, cl_uint)
#endif

#ifdef CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, cl_uint)
#endif
#ifdef CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV, cl_uint)
#endif
#ifdef CL_DEVICE_REGISTERS_PER_BLOCK_NV
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_REGISTERS_PER_BLOCK_NV, cl_uint)
#endif
#ifdef CL_DEVICE_WARP_SIZE_NV
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_WARP_SIZE_NV, cl_uint)
#endif
#ifdef CL_DEVICE_GPU_OVERLAP_NV
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_GPU_OVERLAP_NV, cl_bool)
#endif
#ifdef CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV, cl_bool)
#endif
#ifdef CL_DEVICE_INTEGRATED_MEMORY_NV
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_INTEGRATED_MEMORY_NV, cl_bool)
#endif

#if defined(cl_khr_command_buffer)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_COMMAND_BUFFER_CAPABILITIES_KHR, cl_device_command_buffer_capabilities_khr)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_COMMAND_BUFFER_REQUIRED_QUEUE_PROPERTIES_KHR, cl_command_queue_properties)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_command_buffer_info_khr, CL_COMMAND_BUFFER_QUEUES_KHR, cl::vector<CommandQueue>)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_command_buffer_info_khr, CL_COMMAND_BUFFER_NUM_QUEUES_KHR, cl_uint)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_command_buffer_info_khr, CL_COMMAND_BUFFER_REFERENCE_COUNT_KHR, cl_uint)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_command_buffer_info_khr, CL_COMMAND_BUFFER_STATE_KHR, cl_command_buffer_state_khr)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_command_buffer_info_khr, CL_COMMAND_BUFFER_PROPERTIES_ARRAY_KHR, cl::vector<cl_command_buffer_properties_khr>)
#endif /* cl_khr_command_buffer */

#if defined(cl_khr_command_buffer_mutable_dispatch)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_mutable_command_info_khr, CL_MUTABLE_COMMAND_COMMAND_QUEUE_KHR, CommandQueue)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_mutable_command_info_khr, CL_MUTABLE_COMMAND_COMMAND_BUFFER_KHR, CommandBufferKhr)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_mutable_command_info_khr, CL_MUTABLE_COMMAND_COMMAND_TYPE_KHR, cl_command_type)

#if CL_KHR_COMMAND_BUFFER_MUTABLE_DISPATCH_EXTENSION_VERSION > CL_MAKE_VERSION(0, 9, 2)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_mutable_command_info_khr, CL_MUTABLE_COMMAND_PROPERTIES_ARRAY_KHR, cl::vector<cl_command_properties_khr>)
#else
CL_HPP_DECLARE_PARAM_TRAITS_(cl_mutable_command_info_khr, CL_MUTABLE_DISPATCH_PROPERTIES_ARRAY_KHR, cl::vector<cl_ndrange_kernel_command_properties_khr>)
#endif
CL_HPP_DECLARE_PARAM_TRAITS_(cl_mutable_command_info_khr, CL_MUTABLE_DISPATCH_KERNEL_KHR, cl_kernel)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_mutable_command_info_khr, CL_MUTABLE_DISPATCH_DIMENSIONS_KHR, cl_uint)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_mutable_command_info_khr, CL_MUTABLE_DISPATCH_GLOBAL_WORK_OFFSET_KHR, cl::vector<size_type>)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_mutable_command_info_khr, CL_MUTABLE_DISPATCH_GLOBAL_WORK_SIZE_KHR, cl::vector<size_type>)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_mutable_command_info_khr, CL_MUTABLE_DISPATCH_LOCAL_WORK_SIZE_KHR, cl::vector<size_type>)
#endif /* cl_khr_command_buffer_mutable_dispatch */

#if defined(cl_khr_kernel_clock)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_KERNEL_CLOCK_CAPABILITIES_KHR, cl_device_kernel_clock_capabilities_khr)
#endif /* cl_khr_kernel_clock */

#if defined(cl_ext_float_atomics)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_SINGLE_FP_ATOMIC_CAPABILITIES_EXT, cl_device_fp_atomic_capabilities_ext)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_DOUBLE_FP_ATOMIC_CAPABILITIES_EXT, cl_device_fp_atomic_capabilities_ext)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_HALF_FP_ATOMIC_CAPABILITIES_EXT, cl_device_fp_atomic_capabilities_ext)
#endif /* cl_ext_float_atomics */

#if defined(cl_intel_command_queue_families)
CL_HPP_PARAM_NAME_CL_INTEL_COMMAND_QUEUE_FAMILIES_(CL_HPP_DECLARE_PARAM_TRAITS_)
#endif // cl_intel_command_queue_families

#if defined(cl_intel_device_attribute_query)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_IP_VERSION_INTEL, cl_uint)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_ID_INTEL, cl_uint)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_NUM_SLICES_INTEL, cl_uint)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_NUM_SUB_SLICES_PER_SLICE_INTEL, cl_uint)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_NUM_EUS_PER_SUB_SLICE_INTEL, cl_uint)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_NUM_THREADS_PER_EU_INTEL, cl_uint)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_FEATURE_CAPABILITIES_INTEL, cl_device_feature_capabilities_intel)
#endif // cl_intel_device_attribute_query

#if defined(cl_intel_required_subgroup_size)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_SUB_GROUP_SIZES_INTEL, cl::vector<size_type>)
CL_HPP_DECLARE_PARAM_TRAITS_(cl_kernel_work_group_info, CL_KERNEL_SPILL_MEM_SIZE_INTEL, cl_ulong)
#endif // cl_intel_required_subgroup_size

#if defined(cl_intel_unified_shared_memory)
CL_HPP_PARAM_NAME_CL_INTEL_UNIFIED_SHARED_MEMORY_(CL_HPP_DECLARE_PARAM_TRAITS_)
#endif // cl_intel_unified_shared_memory

// Convenience functions

template <typename Func, typename T>
inline cl_int
getInfo(Func f, cl_uint name, T* param)
{
    return getInfoHelper(f, name, param, 0);
}

template <typename Func, typename Arg0>
struct GetInfoFunctor0
{
    Func f_; const Arg0& arg0_;
    cl_int operator ()(
        cl_uint param, size_type size, void* value, size_type* size_ret)
    { return f_(arg0_, param, size, value, size_ret); }
};

template <typename Func, typename Arg0, typename Arg1>
struct GetInfoFunctor1
{
    Func f_; const Arg0& arg0_; const Arg1& arg1_;
    cl_int operator ()(
        cl_uint param, size_type size, void* value, size_type* size_ret)
    { return f_(arg0_, arg1_, param, size, value, size_ret); }
};

template <typename Func, typename Arg0, typename T>
inline cl_int
getInfo(Func f, const Arg0& arg0, cl_uint name, T* param)
{
    GetInfoFunctor0<Func, Arg0> f0 = { f, arg0 };
    return getInfoHelper(f0, name, param, 0);
}

template <typename Func, typename Arg0, typename Arg1, typename T>
inline cl_int
getInfo(Func f, const Arg0& arg0, const Arg1& arg1, cl_uint name, T* param)
{
    GetInfoFunctor1<Func, Arg0, Arg1> f0 = { f, arg0, arg1 };
    return getInfoHelper(f0, name, param, 0);
}


template<typename T>
struct ReferenceHandler
{ };

#if CL_HPP_TARGET_OPENCL_VERSION >= 120
/**
 * OpenCL 1.2 devices do have retain/release.
 */
template <>
struct ReferenceHandler<cl_device_id>
{
    /**
     * Retain the device.
     * \param device A valid device created using createSubDevices
     * \return 
     *   CL_SUCCESS if the function executed successfully.
     *   CL_INVALID_DEVICE if device was not a valid subdevice
     *   CL_OUT_OF_RESOURCES
     *   CL_OUT_OF_HOST_MEMORY
     */
    static cl_int retain(cl_device_id device)
    { return call_clRetainDevice(device); }
    /**
     * Retain the device.
     * \param device A valid device created using createSubDevices
     * \return 
     *   CL_SUCCESS if the function executed successfully.
     *   CL_INVALID_DEVICE if device was not a valid subdevice
     *   CL_OUT_OF_RESOURCES
     *   CL_OUT_OF_HOST_MEMORY
     */
    static cl_int release(cl_device_id device)
    { return call_clReleaseDevice(device); }
};
#else // CL_HPP_TARGET_OPENCL_VERSION >= 120
/**
 * OpenCL 1.1 devices do not have retain/release.
 */
template <>
struct ReferenceHandler<cl_device_id>
{
    // cl_device_id does not have retain().
    static cl_int retain(cl_device_id)
    { return CL_SUCCESS; }
    // cl_device_id does not have release().
    static cl_int release(cl_device_id)
    { return CL_SUCCESS; }
};
#endif // ! (CL_HPP_TARGET_OPENCL_VERSION >= 120)

template <>
struct ReferenceHandler<cl_platform_id>
{
    // cl_platform_id does not have retain().
    static cl_int retain(cl_platform_id)
    { return CL_SUCCESS; }
    // cl_platform_id does not have release().
    static cl_int release(cl_platform_id)
    { return CL_SUCCESS; }
};

template <>
struct ReferenceHandler<cl_context>
{
    static cl_int retain(cl_context context)
    { return call_clRetainContext(context); }
    static cl_int release(cl_context context)
    { return call_clReleaseContext(context); }
};

template <>
struct ReferenceHandler<cl_command_queue>
{
    static cl_int retain(cl_command_queue queue)
    { return ::clRetainCommandQueue(queue); }
    static cl_int release(cl_command_queue queue)
    { return call_clReleaseCommandQueue(queue); }
};

template <>
struct ReferenceHandler<cl_mem>
{
    static cl_int retain(cl_mem memory)
    { return call_clRetainMemObject(memory); }
    static cl_int release(cl_mem memory)
    { return ::clReleaseMemObject(memory); }
};

template <>
struct ReferenceHandler<cl_sampler>
{
    static cl_int retain(cl_sampler sampler)
    { return ::clRetainSampler(sampler); }
    static cl_int release(cl_sampler sampler)
    { return ::clReleaseSampler(sampler); }
};

template <>
struct ReferenceHandler<cl_program>
{
    static cl_int retain(cl_program program)
    { return ::clRetainProgram(program); }
    static cl_int release(cl_program program)
    { return ::clReleaseProgram(program); }
};

template <>
struct ReferenceHandler<cl_kernel>
{
    static cl_int retain(cl_kernel kernel)
    { return ::clRetainKernel(kernel); }
    static cl_int release(cl_kernel kernel)
    { return ::clReleaseKernel(kernel); }
};

template <>
struct ReferenceHandler<cl_event>
{
    static cl_int retain(cl_event event)
    { return call_clRetainEvent(event); }
    static cl_int release(cl_event event)
    { return ::clReleaseEvent(event); }
};

#ifdef cl_khr_semaphore
template <>
struct ReferenceHandler<cl_semaphore_khr>
{
    static cl_int retain(cl_semaphore_khr semaphore)
    { 
        if (pfn_clRetainSemaphoreKHR != nullptr) {
            return pfn_clRetainSemaphoreKHR(semaphore);
        }

        return CL_INVALID_OPERATION;
    }

    static cl_int release(cl_semaphore_khr semaphore)
    {
        if (pfn_clReleaseSemaphoreKHR != nullptr) {
            return pfn_clReleaseSemaphoreKHR(semaphore);
        }

        return CL_INVALID_OPERATION;
    }
};
#endif // cl_khr_semaphore
#if defined(cl_khr_command_buffer)
template <>
struct ReferenceHandler<cl_command_buffer_khr>
{
    static cl_int retain(cl_command_buffer_khr cmdBufferKhr)
    {
        if (pfn_clRetainCommandBufferKHR == nullptr) {
            return detail::errHandler(CL_INVALID_OPERATION, __RETAIN_COMMAND_BUFFER_KHR_ERR);
        }
        return pfn_clRetainCommandBufferKHR(cmdBufferKhr);
    }

    static cl_int release(cl_command_buffer_khr cmdBufferKhr)
    {
        if (pfn_clReleaseCommandBufferKHR == nullptr) {
            return detail::errHandler(CL_INVALID_OPERATION, __RELEASE_COMMAND_BUFFER_KHR_ERR);
        }
        return pfn_clReleaseCommandBufferKHR(cmdBufferKhr);
    }
};

template <>
struct ReferenceHandler<cl_mutable_command_khr>
{
    // cl_mutable_command_khr does not have retain().
    static cl_int retain(cl_mutable_command_khr)
    { return CL_SUCCESS; }
    // cl_mutable_command_khr does not have release().
    static cl_int release(cl_mutable_command_khr)
    { return CL_SUCCESS; }
};
#endif // cl_khr_command_buffer


#if (CL_HPP_TARGET_OPENCL_VERSION >= 120 && CL_HPP_MINIMUM_OPENCL_VERSION < 120) || \
    (CL_HPP_TARGET_OPENCL_VERSION >= 200 && CL_HPP_MINIMUM_OPENCL_VERSION < 200)
// Extracts version number with major in the upper 16 bits, minor in the lower 16
static cl_uint getVersion(const vector<char> &versionInfo)
{
    int highVersion = 0;
    int lowVersion = 0;
    int index = 7;
    while(versionInfo[index] != '.' ) {
        highVersion *= 10;
        highVersion += versionInfo[index]-'0';
        ++index;
    }
    ++index;
    while(versionInfo[index] != ' ' &&  versionInfo[index] != '\0') {
        lowVersion *= 10;
        lowVersion += versionInfo[index]-'0';
        ++index;
    }
    return (highVersion << 16) | lowVersion;
}

static cl_uint getPlatformVersion(cl_platform_id platform)
{
    size_type size = 0;
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, 0, nullptr, &size);

    vector<char> versionInfo(size);
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, size, versionInfo.data(), &size);
    return getVersion(versionInfo);
}

static cl_uint getDevicePlatformVersion(cl_device_id device)
{
    cl_platform_id platform;
    call_clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(platform), &platform, nullptr);
    return getPlatformVersion(platform);
}

static cl_uint getContextPlatformVersion(cl_context context)
{
    // The platform cannot be queried directly, so we first have to grab a
    // device and obtain its context
    size_type size = 0;
    clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, nullptr, &size);
    if (size == 0)
        return 0;
    vector<cl_device_id> devices(size/sizeof(cl_device_id));
    clGetContextInfo(context, CL_CONTEXT_DEVICES, size, devices.data(), nullptr);
    return getDevicePlatformVersion(devices[0]);
}
#endif // CL_HPP_TARGET_OPENCL_VERSION && CL_HPP_MINIMUM_OPENCL_VERSION

template <typename T>
class Wrapper
{
public:
    typedef T cl_type;

protected:
    cl_type object_;

public:
    Wrapper() : object_(nullptr) { }
    
    Wrapper(const cl_type &obj, bool retainObject) : object_(obj) 
    {
        if (retainObject) { 
            detail::errHandler(retain(), __RETAIN_ERR); 
        }
    }

    ~Wrapper()
    {
        if (object_ != nullptr) { release(); }
    }

    Wrapper(const Wrapper<cl_type>& rhs)
    {
        object_ = rhs.object_;
        detail::errHandler(retain(), __RETAIN_ERR);
    }

    Wrapper(Wrapper<cl_type>&& rhs) noexcept
    {
        object_ = rhs.object_;
        rhs.object_ = nullptr;
    }

    Wrapper<cl_type>& operator = (const Wrapper<cl_type>& rhs)
    {
        if (this != &rhs) {
            detail::errHandler(release(), __RELEASE_ERR);
            object_ = rhs.object_;
            detail::errHandler(retain(), __RETAIN_ERR);
        }
        return *this;
    }

    Wrapper<cl_type>& operator = (Wrapper<cl_type>&& rhs)
    {
        if (this != &rhs) {
            detail::errHandler(release(), __RELEASE_ERR);
            object_ = rhs.object_;
            rhs.object_ = nullptr;
        }
        return *this;
    }

    Wrapper<cl_type>& operator = (const cl_type &rhs)
    {
        detail::errHandler(release(), __RELEASE_ERR);
        object_ = rhs;
        return *this;
    }

    const cl_type& operator ()() const { return object_; }

    cl_type& operator ()() { return object_; }

    cl_type get() const { return object_; }

protected:
    template<typename Func, typename U>
    friend inline cl_int getInfoHelper(Func, cl_uint, U*, int, typename U::cl_type);

    cl_int retain() const
    {
        if (object_ != nullptr) {
            return ReferenceHandler<cl_type>::retain(object_);
        }
        else {
            return CL_SUCCESS;
        }
    }

    cl_int release() const
    {
        if (object_ != nullptr) {
            return ReferenceHandler<cl_type>::release(object_);
        }
        else {
            return CL_SUCCESS;
        }
    }
};

template <>
class Wrapper<cl_device_id>
{
public:
    typedef cl_device_id cl_type;

protected:
    cl_type object_;
    bool referenceCountable_;

    static bool isReferenceCountable(cl_device_id device)
    {
        bool retVal = false;
#if CL_HPP_TARGET_OPENCL_VERSION >= 120 && CL_HPP_MINIMUM_OPENCL_VERSION < 120
        if (device != nullptr) {
            int version = getDevicePlatformVersion(device);
            if(version > ((1 << 16) + 1)) {
                retVal = true;
            }
        }
#elif CL_HPP_TARGET_OPENCL_VERSION >= 120
        retVal = true;
#endif // CL_HPP_TARGET_OPENCL_VERSION
        (void)device;
        return retVal;
    }

public:
    Wrapper() : object_(nullptr), referenceCountable_(false) 
    { 
    }
    
    Wrapper(const cl_type &obj, bool retainObject) : 
        object_(obj), 
        referenceCountable_(false) 
    {
        referenceCountable_ = isReferenceCountable(obj); 

        if (retainObject) {
            detail::errHandler(retain(), __RETAIN_ERR);
        }
    }

    ~Wrapper()
    {
        release();
    }
    
    Wrapper(const Wrapper<cl_type>& rhs)
    {
        object_ = rhs.object_;
        referenceCountable_ = isReferenceCountable(object_); 
        detail::errHandler(retain(), __RETAIN_ERR);
    }

    Wrapper(Wrapper<cl_type>&& rhs) noexcept
    {
        object_ = rhs.object_;
        referenceCountable_ = rhs.referenceCountable_;
        rhs.object_ = nullptr;
        rhs.referenceCountable_ = false;
    }

    Wrapper<cl_type>& operator = (const Wrapper<cl_type>& rhs)
    {
        if (this != &rhs) {
            detail::errHandler(release(), __RELEASE_ERR);
            object_ = rhs.object_;
            referenceCountable_ = rhs.referenceCountable_;
            detail::errHandler(retain(), __RETAIN_ERR);
        }
        return *this;
    }

    Wrapper<cl_type>& operator = (Wrapper<cl_type>&& rhs)
    {
        if (this != &rhs) {
            detail::errHandler(release(), __RELEASE_ERR);
            object_ = rhs.object_;
            referenceCountable_ = rhs.referenceCountable_;
            rhs.object_ = nullptr;
            rhs.referenceCountable_ = false;
        }
        return *this;
    }

    Wrapper<cl_type>& operator = (const cl_type &rhs)
    {
        detail::errHandler(release(), __RELEASE_ERR);
        object_ = rhs;
        referenceCountable_ = isReferenceCountable(object_); 
        return *this;
    }

    const cl_type& operator ()() const { return object_; }

    cl_type& operator ()() { return object_; }

    cl_type get() const { return object_; }

protected:
    template<typename Func, typename U>
    friend inline cl_int getInfoHelper(Func, cl_uint, U*, int, typename U::cl_type);

    template<typename Func, typename U>
    friend inline cl_int getInfoHelper(Func, cl_uint, vector<U>*, int, typename U::cl_type);

    cl_int retain() const
    {
        if( object_ != nullptr && referenceCountable_ ) {
            return ReferenceHandler<cl_type>::retain(object_);
        }
        else {
            return CL_SUCCESS;
        }
    }

    cl_int release() const
    {
        if (object_ != nullptr && referenceCountable_) {
            return ReferenceHandler<cl_type>::release(object_);
        }
        else {
            return CL_SUCCESS;
        }
    }
};

template <typename T>
inline bool operator==(const Wrapper<T> &lhs, const Wrapper<T> &rhs)
{
    return lhs() == rhs();
}

template <typename T>
inline bool operator!=(const Wrapper<T> &lhs, const Wrapper<T> &rhs)
{
    return !operator==(lhs, rhs);
}

} // namespace detail
//! \endcond





/*! \stuct ImageFormat
 *  \brief Adds constructors and member functions for cl_image_format.
 *
 *  \see cl_image_format
 */
struct ImageFormat : public cl_image_format
{
    //! \brief Default constructor - performs no initialization.
    ImageFormat(){}

    //! \brief Initializing constructor.
    ImageFormat(cl_channel_order order, cl_channel_type type)
    {
        image_channel_order = order;
        image_channel_data_type = type;
    }

    //! \brief Copy constructor.
    ImageFormat(const ImageFormat &other) { *this = other; }

    //! \brief Assignment operator.
    ImageFormat& operator = (const ImageFormat& rhs)
    {
        if (this != &rhs) {
            this->image_channel_data_type = rhs.image_channel_data_type;
            this->image_channel_order     = rhs.image_channel_order;
        }
        return *this;
    }
};

/*! \brief Class interface for cl_device_id.
 *
 *  \note Copies of these objects are inexpensive, since they don't 'own'
 *        any underlying resources or data structures.
 *
 *  \see cl_device_id
 */
class Device : public detail::Wrapper<cl_device_id>
{
private:
    static std::once_flag default_initialized_;
    static Device default_;
    static cl_int default_error_;

    /*! \brief Create the default context.
    *
    * This sets @c default_ and @c default_error_. It does not throw
    * @c cl::Error.
    */
    static void makeDefault();

    /*! \brief Create the default platform from a provided platform.
    *
    * This sets @c default_. It does not throw
    * @c cl::Error.
    */
    static void makeDefaultProvided(const Device &p) {
        default_ = p;
    }

public:
#ifdef CL_HPP_UNIT_TEST_ENABLE
    /*! \brief Reset the default.
    *
    * This sets @c default_ to an empty value to support cleanup in
    * the unit test framework.
    * This function is not thread safe.
    */
    static void unitTestClearDefault() {
        default_ = Device();
    }
#endif // #ifdef CL_HPP_UNIT_TEST_ENABLE

    //! \brief Default constructor - initializes to nullptr.
    Device() : detail::Wrapper<cl_type>() { }

    /*! \brief Constructor from cl_device_id.
     * 
     *  This simply copies the device ID value, which is an inexpensive operation.
     */
    explicit Device(const cl_device_id &device, bool retainObject = false) : 
        detail::Wrapper<cl_type>(device, retainObject) { }

    /*! \brief Returns the first device on the default context.
     *
     *  \see Context::getDefault()
     */
    static Device getDefault(
        cl_int *errResult = nullptr)
    {
        std::call_once(default_initialized_, makeDefault);
        detail::errHandler(default_error_);
        if (errResult != nullptr) {
            *errResult = default_error_;
        }
        return default_;
    }

    /**
    * Modify the default device to be used by
    * subsequent operations.
    * Will only set the default if no default was previously created.
    * @return updated default device.
    *         Should be compared to the passed value to ensure that it was updated.
    */
    static Device setDefault(const Device &default_device)
    {
        std::call_once(default_initialized_, makeDefaultProvided, std::cref(default_device));
        detail::errHandler(default_error_);
        return default_;
    }

    /*! \brief Assignment operator from cl_device_id.
     * 
     *  This simply copies the device ID value, which is an inexpensive operation.
     */
    Device& operator = (const cl_device_id& rhs)
    {
        detail::Wrapper<cl_type>::operator=(rhs);
        return *this;
    }
 

    //! \brief Wrapper for clGetDeviceInfo().
    template <typename T>
    cl_int getInfo(cl_device_info name, T* param) const
    {
        return detail::errHandler(
            detail::getInfo(&call_clGetDeviceInfo, object_, name, param),
            __GET_DEVICE_INFO_ERR);
    }

    //! \brief Wrapper for clGetDeviceInfo() that returns by value.
    template <cl_device_info name> typename
    detail::param_traits<detail::cl_device_info, name>::param_type
    getInfo(cl_int* err = nullptr) const
    {
        typename detail::param_traits<
            detail::cl_device_info, name>::param_type param;
        cl_int result = getInfo(name, &param);
        if (err != nullptr) {
            *err = result;
        }
        return param;
    }

#if CL_HPP_TARGET_OPENCL_VERSION >= 210
    /**
     * Return the current value of the host clock as seen by the device.
     * The resolution of the device timer may be queried with the
     * CL_DEVICE_PROFILING_TIMER_RESOLUTION query.
     * @return The host timer value.
     */
    cl_ulong getHostTimer(cl_int *error = nullptr)
    {
        cl_ulong retVal = 0;
        cl_int err = 
            clGetHostTimer(this->get(), &retVal);
        detail::errHandler(
            err,
            __GET_HOST_TIMER_ERR);
        if (error) {
            *error = err;
        }
        return retVal;
    }

    /**
     * Return a synchronized pair of host and device timestamps as seen by device.
     * Use to correlate the clocks and get the host timer only using getHostTimer
     * as a lower cost mechanism in between calls.
     * The resolution of the host timer may be queried with the 
     * CL_PLATFORM_HOST_TIMER_RESOLUTION query.
     * The resolution of the device timer may be queried with the
     * CL_DEVICE_PROFILING_TIMER_RESOLUTION query.
     * @return A pair of (device timer, host timer) timer values.
     */
    std::pair<cl_ulong, cl_ulong> getDeviceAndHostTimer(cl_int *error = nullptr)
    {
        std::pair<cl_ulong, cl_ulong> retVal;
        cl_int err =
            clGetDeviceAndHostTimer(this->get(), &(retVal.first), &(retVal.second));
        detail::errHandler(
            err,
            __GET_DEVICE_AND_HOST_TIMER_ERR);
        if (error) {
            *error = err;
        }
        return retVal;
    }
#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 210

#if CL_HPP_TARGET_OPENCL_VERSION >= 120
    //! \brief Wrapper for clCreateSubDevices().
    cl_int createSubDevices(const cl_device_partition_property* properties,
                            vector<Device>* devices);
#endif // defined (CL_HPP_TARGET_OPENCL_VERSION >= 120)

#if defined(cl_ext_device_fission)
    //! \brief Wrapper for clCreateSubDevices().
    cl_int createSubDevices(const cl_device_partition_property_ext* properties,
                            vector<Device>* devices);
#endif // defined(cl_ext_device_fission)
};

using BuildLogType = vector<std::pair<cl::Device, typename detail::param_traits<detail::cl_program_build_info, CL_PROGRAM_BUILD_LOG>::param_type>>;
#if defined(CL_HPP_ENABLE_EXCEPTIONS)
/**
* Exception class for build errors to carry build info
*/
class BuildError : public Error
{
private:
    BuildLogType buildLogs;
public:
    BuildError(cl_int err, const char * errStr, const BuildLogType &vec) : Error(err, errStr), buildLogs(vec)
    {
    }

    BuildLogType getBuildLog() const
    {
        return buildLogs;
    }
};
namespace detail {
    static inline cl_int buildErrHandler(
        cl_int err,
        const char * errStr,
        const BuildLogType &buildLogs)
    {
        if (err != CL_SUCCESS) {
            throw BuildError(err, errStr, buildLogs);
        }
        return err;
    }
} // namespace detail

#else
namespace detail {
    static inline cl_int buildErrHandler(
        cl_int err,
        const char * errStr,
        const BuildLogType &buildLogs)
    {
        (void)buildLogs; // suppress unused variable warning
        (void)errStr;
        return err;
    }
} // namespace detail
#endif // #if defined(CL_HPP_ENABLE_EXCEPTIONS)

CL_HPP_DEFINE_STATIC_MEMBER_ std::once_flag Device::default_initialized_;
CL_HPP_DEFINE_STATIC_MEMBER_ Device Device::default_;
CL_HPP_DEFINE_STATIC_MEMBER_ cl_int Device::default_error_ = CL_SUCCESS;

/*! \brief Class interface for cl_platform_id.
 *
 *  \note Copies of these objects are inexpensive, since they don't 'own'
 *        any underlying resources or data structures.
 *
 *  \see cl_platform_id
 */
class Platform : public detail::Wrapper<cl_platform_id>
{
private:
    static std::once_flag default_initialized_;
    static Platform default_;
    static cl_int default_error_;

    /*! \brief Create the default context.
    *
    * This sets @c default_ and @c default_error_. It does not throw
    * @c cl::Error.
    */
    static void makeDefault() {
        /* Throwing an exception from a call_once invocation does not do
        * what we wish, so we catch it and save the error.
        */
#if defined(CL_HPP_ENABLE_EXCEPTIONS)
        try
#endif
        {
            // If default wasn't passed ,generate one
            // Otherwise set it
            cl_uint n = 0;

            cl_int err = call_clGetPlatformIDs(0, nullptr, &n);
            if (err != CL_SUCCESS) {
                default_error_ = err;
                return;
            }
            if (n == 0) {
                default_error_ = CL_INVALID_PLATFORM;
                return;
            }

            vector<cl_platform_id> ids(n);
            err = call_clGetPlatformIDs(n, ids.data(), nullptr);
            if (err != CL_SUCCESS) {
                default_error_ = err;
                return;
            }

            default_ = Platform(ids[0]);
        }
#if defined(CL_HPP_ENABLE_EXCEPTIONS)
        catch (cl::Error &e) {
            default_error_ = e.err();
        }
#endif
    }

    /*! \brief Create the default platform from a provided platform.
     *
     * This sets @c default_. It does not throw
     * @c cl::Error.
     */
    static void makeDefaultProvided(const Platform &p) {
       default_ = p;
    }
    
public:
#ifdef CL_HPP_UNIT_TEST_ENABLE
    /*! \brief Reset the default.
    *
    * This sets @c default_ to an empty value to support cleanup in
    * the unit test framework.
    * This function is not thread safe.
    */
    static void unitTestClearDefault() {
        default_ = Platform();
    }
#endif // #ifdef CL_HPP_UNIT_TEST_ENABLE

    //! \brief Default constructor - initializes to nullptr.
    Platform() : detail::Wrapper<cl_type>()  { }

    /*! \brief Constructor from cl_platform_id.
     * 
     * \param retainObject will cause the constructor to retain its cl object.
     *                     Defaults to false to maintain compatibility with
     *                     earlier versions.
     *  This simply copies the platform ID value, which is an inexpensive operation.
     */
    explicit Platform(const cl_platform_id &platform, bool retainObject = false) : 
        detail::Wrapper<cl_type>(platform, retainObject) { }

    /*! \brief Assignment operator from cl_platform_id.
     * 
     *  This simply copies the platform ID value, which is an inexpensive operation.
     */
    Platform& operator = (const cl_platform_id& rhs)
    {
        detail::Wrapper<cl_type>::operator=(rhs);
        return *this;
    }

    static Platform getDefault(
        cl_int *errResult = nullptr)
    {
        std::call_once(default_initialized_, makeDefault);
        detail::errHandler(default_error_);
        if (errResult != nullptr) {
            *errResult = default_error_;
        }
        return default_;
    }

    /**
     * Modify the default platform to be used by 
     * subsequent operations.
     * Will only set the default if no default was previously created.
     * @return updated default platform. 
     *         Should be compared to the passed value to ensure that it was updated.
     */
    static Platform setDefault(const Platform &default_platform)
    {
        std::call_once(default_initialized_, makeDefaultProvided, std::cref(default_platform));
        detail::errHandler(default_error_);
        return default_;
    }

    //! \brief Wrapper for clGetPlatformInfo().
    template <typename T>
    cl_int getInfo(cl_platform_info name, T* param) const
    {
        return detail::errHandler(
            detail::getInfo(&::clGetPlatformInfo, object_, name, param),
            __GET_PLATFORM_INFO_ERR);
    }

    //! \brief Wrapper for clGetPlatformInfo() that returns by value.
    template <cl_platform_info name> typename
    detail::param_traits<detail::cl_platform_info, name>::param_type
    getInfo(cl_int* err = nullptr) const
    {
        typename detail::param_traits<
            detail::cl_platform_info, name>::param_type param;
        cl_int result = getInfo(name, &param);
        if (err != nullptr) {
            *err = result;
        }
        return param;
    }

    /*! \brief Gets a list of devices for this platform.
     * 
     *  Wraps clGetDeviceIDs().
     */
    cl_int getDevices(
        cl_device_type type,
        vector<Device>* devices) const
    {
        cl_uint n = 0;
        if( devices == nullptr ) {
            return detail::errHandler(CL_INVALID_ARG_VALUE, __GET_DEVICE_IDS_ERR);
        }
        cl_int err = call_clGetDeviceIDs(object_, type, 0, nullptr, &n);
        if (err != CL_SUCCESS  && err != CL_DEVICE_NOT_FOUND) {
            return detail::errHandler(err, __GET_DEVICE_IDS_ERR);
        }

        vector<cl_device_id> ids(n);
        if (n>0) {
            err = call_clGetDeviceIDs(object_, type, n, ids.data(), nullptr);
            if (err != CL_SUCCESS) {
                return detail::errHandler(err, __GET_DEVICE_IDS_ERR);
            }
        }

        // Cannot trivially assign because we need to capture intermediates 
        // with safe construction
        // We must retain things we obtain from the API to avoid releasing
        // API-owned objects.
        if (devices) {
            devices->resize(ids.size());

            // Assign to param, constructing with retain behaviour
            // to correctly capture each underlying CL object
            for (size_type i = 0; i < ids.size(); i++) {
                (*devices)[i] = Device(ids[i], true);
            }
        }
        return CL_SUCCESS;
    }

#if defined(CL_HPP_USE_DX_INTEROP)
   /*! \brief Get the list of available D3D10 devices.
     *
     *  \param d3d_device_source.
     *
     *  \param d3d_object.
     *
     *  \param d3d_device_set.
     *
     *  \param devices returns a vector of OpenCL D3D10 devices found. The cl::Device
     *  values returned in devices can be used to identify a specific OpenCL
     *  device. If \a devices argument is nullptr, this argument is ignored.
     *
     *  \return One of the following values:
     *    - CL_SUCCESS if the function is executed successfully.
     *
     *  The application can query specific capabilities of the OpenCL device(s)
     *  returned by cl::getDevices. This can be used by the application to
     *  determine which device(s) to use.
     *
     * \note In the case that exceptions are enabled and a return value
     * other than CL_SUCCESS is generated, then cl::Error exception is
     * generated.
     */
    cl_int getDevices(
        cl_d3d10_device_source_khr d3d_device_source,
        void *                     d3d_object,
        cl_d3d10_device_set_khr    d3d_device_set,
        vector<Device>* devices) const
    {
        typedef CL_API_ENTRY cl_int (CL_API_CALL *PFN_clGetDeviceIDsFromD3D10KHR)(
            cl_platform_id platform, 
            cl_d3d10_device_source_khr d3d_device_source, 
            void * d3d_object,
            cl_d3d10_device_set_khr d3d_device_set,
            cl_uint num_entries,
            cl_device_id * devices,
            cl_uint* num_devices);

        if( devices == nullptr ) {
            return detail::errHandler(CL_INVALID_ARG_VALUE, __GET_DEVICE_IDS_ERR);
        }

        static PFN_clGetDeviceIDsFromD3D10KHR pfn_clGetDeviceIDsFromD3D10KHR = nullptr;
#if CL_HPP_TARGET_OPENCL_VERSION >= 120
        CL_HPP_INIT_CL_EXT_FCN_PTR_PLATFORM_(object_, clGetDeviceIDsFromD3D10KHR);
#endif
#if CL_HPP_MINIMUM_OPENCL_VERSION < 120
        CL_HPP_INIT_CL_EXT_FCN_PTR_(clGetDeviceIDsFromD3D10KHR);
#endif

        cl_uint n = 0;
        cl_int err = pfn_clGetDeviceIDsFromD3D10KHR(
            object_, 
            d3d_device_source, 
            d3d_object,
            d3d_device_set, 
            0, 
            nullptr, 
            &n);
        if (err != CL_SUCCESS) {
            return detail::errHandler(err, __GET_DEVICE_IDS_ERR);
        }

        vector<cl_device_id> ids(n);
        err = pfn_clGetDeviceIDsFromD3D10KHR(
            object_, 
            d3d_device_source, 
            d3d_object,
            d3d_device_set,
            n, 
            ids.data(), 
            nullptr);
        if (err != CL_SUCCESS) {
            return detail::errHandler(err, __GET_DEVICE_IDS_ERR);
        }

        // Cannot trivially assign because we need to capture intermediates 
        // with safe construction
        // We must retain things we obtain from the API to avoid releasing
        // API-owned objects.
        if (devices) {
            devices->resize(ids.size());

            // Assign to param, constructing with retain behaviour
            // to correctly capture each underlying CL object
            for (size_type i = 0; i < ids.size(); i++) {
                (*devices)[i] = Device(ids[i], true);
            }
        }
        return CL_SUCCESS;
    }
#endif

    /*! \brief Gets a list of available platforms.
     * 
     *  Wraps clGetPlatformIDs().
     */
    static cl_int get(
        vector<Platform>* platforms)
    {
        cl_uint n = 0;

        if( platforms == nullptr ) {
            return detail::errHandler(CL_INVALID_ARG_VALUE, __GET_PLATFORM_IDS_ERR);
        }

        cl_int err = call_clGetPlatformIDs(0, nullptr, &n);
        if (err != CL_SUCCESS) {
            return detail::errHandler(err, __GET_PLATFORM_IDS_ERR);
        }

        vector<cl_platform_id> ids(n);
        err = call_clGetPlatformIDs(n, ids.data(), nullptr);
        if (err != CL_SUCCESS) {
            return detail::errHandler(err, __GET_PLATFORM_IDS_ERR);
        }

        if (platforms) {
            platforms->resize(ids.size());

            // Platforms don't reference count
            for (size_type i = 0; i < ids.size(); i++) {
                (*platforms)[i] = Platform(ids[i]);
            }
        }
        return CL_SUCCESS;
    }

    /*! \brief Gets the first available platform.
     * 
     *  Wraps clGetPlatformIDs(), returning the first result.
     */
    static cl_int get(
        Platform * platform)
    {
        cl_int err;
        Platform default_platform = Platform::getDefault(&err);
        if (platform) {
            *platform = default_platform;
        }
        return err;
    }

    /*! \brief Gets the first available platform, returning it by value.
     *
     * \return Returns a valid platform if one is available.
     *         If no platform is available will return a null platform.
     * Throws an exception if no platforms are available
     * or an error condition occurs.
     * Wraps clGetPlatformIDs(), returning the first result.
     */
    static Platform get(
        cl_int * errResult = nullptr)
    {
        cl_int err;
        Platform default_platform = Platform::getDefault(&err);
        if (errResult) {
            *errResult = err;
        }
        return default_platform;
    }    
    
#if CL_HPP_TARGET_OPENCL_VERSION >= 120
    //! \brief Wrapper for clUnloadCompiler().
    cl_int
    unloadCompiler()
    {
        return call_clUnloadPlatformCompiler(object_);
    }
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
}; // class Platform

#if CL_HPP_TARGET_OPENCL_VERSION >= 120
   //! \brief Wrapper for clCreateSubDevices().
inline cl_int Device::createSubDevices(const cl_device_partition_property* properties,
                         vector<Device>* devices)
{
    cl_uint n = 0;
    cl_int err = call_clCreateSubDevices(object_, properties, 0, nullptr, &n);
    if (err != CL_SUCCESS)
    {
        return detail::errHandler(err, __CREATE_SUB_DEVICES_ERR);
    }

    vector<cl_device_id> ids(n);
    err = clCreateSubDevices(object_, properties, n, ids.data(), nullptr);
    if (err != CL_SUCCESS)
    {
        return detail::errHandler(err, __CREATE_SUB_DEVICES_ERR);
    }

    // Cannot trivially assign because we need to capture intermediates
    // with safe construction
    if (devices)
    {
        devices->resize(ids.size());

        // Assign to param, constructing with retain behaviour
        // to correctly capture each underlying CL object
        for (size_type i = 0; i < ids.size(); i++)
        {
            // We do not need to retain because this device is being created
            // by the runtime
            (*devices)[i] = Device(ids[i], false);
        }
    }

    return CL_SUCCESS;
}
#endif // defined (CL_HPP_TARGET_OPENCL_VERSION >= 120)

#if defined(cl_ext_device_fission)
   //! \brief Wrapper for clCreateSubDevices().
inline cl_int Device::createSubDevices(const cl_device_partition_property_ext* properties,
                        vector<Device>* devices)
{
#if CL_HPP_TARGET_OPENCL_VERSION >= 120
    cl::Device device(object_);
    cl_platform_id platform = device.getInfo<CL_DEVICE_PLATFORM>()();
    CL_HPP_INIT_CL_EXT_FCN_PTR_PLATFORM_(platform, clCreateSubDevicesEXT);
#endif
#if CL_HPP_MINIMUM_OPENCL_VERSION < 120
    CL_HPP_INIT_CL_EXT_FCN_PTR_(clCreateSubDevicesEXT);
#endif

    cl_uint n = 0;
    cl_int err = pfn_clCreateSubDevicesEXT(object_, properties, 0, nullptr, &n);
    if (err != CL_SUCCESS)
    {
        return detail::errHandler(err, __CREATE_SUB_DEVICES_ERR);
    }

    vector<cl_device_id> ids(n);
    err =
        pfn_clCreateSubDevicesEXT(object_, properties, n, ids.data(), nullptr);
    if (err != CL_SUCCESS)
    {
        return detail::errHandler(err, __CREATE_SUB_DEVICES_ERR);
    }
    // Cannot trivially assign because we need to capture intermediates
    // with safe construction
    if (devices)
    {
        devices->resize(ids.size());

        // Assign to param, constructing with retain behaviour
        // to correctly capture each underlying CL object
        for (size_type i = 0; i < ids.size(); i++)
        {
            // We do not need to retain because this device is being created
            // by the runtime
            (*devices)[i] = Device(ids[i], false);
        }
    }

    return CL_SUCCESS;
}
#endif // defined(cl_ext_device_fission)

CL_HPP_DEFINE_STATIC_MEMBER_ std::once_flag Platform::default_initialized_;
CL_HPP_DEFINE_STATIC_MEMBER_ Platform Platform::default_;
CL_HPP_DEFINE_STATIC_MEMBER_ cl_int Platform::default_error_ = CL_SUCCESS;


/**
 * Deprecated APIs for 1.2
 */
#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
/**
 * Unload the OpenCL compiler.
 * \note Deprecated for OpenCL 1.2. Use Platform::unloadCompiler instead.
 */
inline CL_API_PREFIX__VERSION_1_1_DEPRECATED cl_int
UnloadCompiler() CL_API_SUFFIX__VERSION_1_1_DEPRECATED;
inline cl_int
UnloadCompiler()
{
    return ::clUnloadCompiler();
}
#endif // #if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)


#if defined(cl_ext_image_requirements_info)
enum ImageRequirementsInfoExt : cl_image_requirements_info_ext
{
    RowPitchAlign = CL_IMAGE_REQUIREMENTS_ROW_PITCH_ALIGNMENT_EXT,
    BaseAddAlign = CL_IMAGE_REQUIREMENTS_BASE_ADDRESS_ALIGNMENT_EXT,
    Size = CL_IMAGE_REQUIREMENTS_SIZE_EXT,
    MaxWidth = CL_IMAGE_REQUIREMENTS_MAX_WIDTH_EXT,
    MaxHeight = CL_IMAGE_REQUIREMENTS_MAX_HEIGHT_EXT,
    MaxDepth = CL_IMAGE_REQUIREMENTS_MAX_DEPTH_EXT,
    MaxArraySize = CL_IMAGE_REQUIREMENTS_MAX_ARRAY_SIZE_EXT,
#if defined(cl_ext_image_from_buffer)
    SlicePitchAlign = CL_IMAGE_REQUIREMENTS_SLICE_PITCH_ALIGNMENT_EXT,
#endif
};

#endif // cl_ext_image_requirements_info


/*! \brief Class interface for cl_context.
 *
 *  \note Copies of these objects are shallow, meaning that the copy will refer
 *        to the same underlying cl_context as the original.  For details, see
 *        clRetainContext() and clReleaseContext().
 *
 *  \see cl_context
 */
class Context 
    : public detail::Wrapper<cl_context>
{
private:
    static std::once_flag default_initialized_;
    static Context default_;
    static cl_int default_error_;

    /*! \brief Create the default context from the default device type in the default platform.
     *
     * This sets @c default_ and @c default_error_. It does not throw
     * @c cl::Error.
     */
    static void makeDefault() {
        /* Throwing an exception from a call_once invocation does not do
         * what we wish, so we catch it and save the error.
         */
#if defined(CL_HPP_ENABLE_EXCEPTIONS)
        try
#endif
        {
#if !defined(__APPLE__) && !defined(__MACOS)
            const Platform &p = Platform::getDefault();
            cl_platform_id defaultPlatform = p();
            cl_context_properties properties[3] = {
                CL_CONTEXT_PLATFORM, (cl_context_properties)defaultPlatform, 0
            };
#else // #if !defined(__APPLE__) && !defined(__MACOS)
            cl_context_properties *properties = nullptr;
#endif // #if !defined(__APPLE__) && !defined(__MACOS)

            default_ = Context(
                CL_DEVICE_TYPE_DEFAULT,
                properties,
                nullptr,
                nullptr,
                &default_error_);
        }
#if defined(CL_HPP_ENABLE_EXCEPTIONS)
        catch (cl::Error &e) {
            default_error_ = e.err();
        }
#endif
    }


    /*! \brief Create the default context from a provided Context.
     *
     * This sets @c default_. It does not throw
     * @c cl::Error.
     */
    static void makeDefaultProvided(const Context &c) {
        default_ = c;
    }

#if defined(cl_ext_image_requirements_info)
    struct ImageRequirementsInfo {

        ImageRequirementsInfo(cl_mem_flags f, const cl_mem_properties* mem_properties, const ImageFormat* format, const cl_image_desc* desc)
        {
            flags = f;
            properties = mem_properties;
            image_format = format;
            image_desc = desc;
        }

        cl_mem_flags flags = 0;
        const cl_mem_properties* properties;
        const ImageFormat* image_format;
        const cl_image_desc* image_desc;
    };

    static cl_int getImageRequirementsInfoExtHelper(const Context &context,
        const ImageRequirementsInfo &info,
        cl_image_requirements_info_ext param_name,
        size_type param_value_size,
        void* param_value,
        size_type* param_value_size_ret)
    {

#if CL_HPP_TARGET_OPENCL_VERSION >= 120
        Device device = context.getInfo<CL_CONTEXT_DEVICES>().at(0);
        cl_platform_id platform = device.getInfo<CL_DEVICE_PLATFORM>()();
        CL_HPP_INIT_CL_EXT_FCN_PTR_PLATFORM_(platform, clGetImageRequirementsInfoEXT);
#else
        CL_HPP_INIT_CL_EXT_FCN_PTR_(clGetImageRequirementsInfoEXT);
#endif

        if (pfn_clGetImageRequirementsInfoEXT == nullptr) {
            return detail::errHandler(CL_INVALID_OPERATION, __GET_IMAGE_REQUIREMENT_INFO_EXT_ERR);
        }

        return detail::errHandler(
            pfn_clGetImageRequirementsInfoEXT(context(), info.properties,
                info.flags, info.image_format, info.image_desc, param_name,
                param_value_size, param_value, param_value_size_ret),
            __GET_IMAGE_REQUIREMENT_INFO_EXT_ERR);
    }
#endif // cl_ext_image_requirements_info
    
public:
#ifdef CL_HPP_UNIT_TEST_ENABLE
    /*! \brief Reset the default.
    *
    * This sets @c default_ to an empty value to support cleanup in
    * the unit test framework.
    * This function is not thread safe.
    */
    static void unitTestClearDefault() {
        default_ = Context();
    }
#endif // #ifdef CL_HPP_UNIT_TEST_ENABLE

    /*! \brief Constructs a context including a list of specified devices.
     *
     *  Wraps clCreateContext().
     */
    Context(
        const vector<Device>& devices,
        const cl_context_properties* properties = nullptr,
        void (CL_CALLBACK * notifyFptr)(
            const char *,
            const void *,
            size_type,
            void *) = nullptr,
        void* data = nullptr,
        cl_int* err = nullptr)
    {
        cl_int error;

        size_type numDevices = devices.size();
        vector<cl_device_id> deviceIDs(numDevices);

        for( size_type deviceIndex = 0; deviceIndex < numDevices; ++deviceIndex ) {
            deviceIDs[deviceIndex] = (devices[deviceIndex])();
        }

        object_ = call_clCreateContext(
            properties, (cl_uint) numDevices,
            deviceIDs.data(),
            notifyFptr, data, &error);

        detail::errHandler(error, __CREATE_CONTEXT_ERR);
        if (err != nullptr) {
            *err = error;
        }
    }

    /*! \brief Constructs a context including a specific device.
     *
     *  Wraps clCreateContext().
     */
    Context(
        const Device& device,
        const cl_context_properties* properties = nullptr,
        void (CL_CALLBACK * notifyFptr)(
            const char *,
            const void *,
            size_type,
            void *) = nullptr,
        void* data = nullptr,
        cl_int* err = nullptr)
    {
        cl_int error;

        cl_device_id deviceID = device();

        object_ = call_clCreateContext(
            properties, 1,
            &deviceID,
            notifyFptr, data, &error);

        detail::errHandler(error, __CREATE_CONTEXT_ERR);
        if (err != nullptr) {
            *err = error;
        }
    }
    
    /*! \brief Constructs a context including all or a subset of devices of a specified type.
     *
     *  Wraps clCreateContextFromType().
     */
    Context(
        cl_device_type type,
        const cl_context_properties* properties = nullptr,
        void (CL_CALLBACK * notifyFptr)(
            const char *,
            const void *,
            size_type,
            void *) = nullptr,
        void* data = nullptr,
        cl_int* err = nullptr)
    {
        cl_int error;

#if !defined(__APPLE__) && !defined(__MACOS)
        cl_context_properties prop[4] = {CL_CONTEXT_PLATFORM, 0, 0, 0 };

        if (properties == nullptr) {
            // Get a valid platform ID as we cannot send in a blank one
            vector<Platform> platforms;
            error = Platform::get(&platforms);
            if (error != CL_SUCCESS) {
                detail::errHandler(error, __CREATE_CONTEXT_FROM_TYPE_ERR);
                if (err != nullptr) {
                    *err = error;
                }
                return;
            }

            // Check the platforms we found for a device of our specified type
            cl_context_properties platform_id = 0;
            for (unsigned int i = 0; i < platforms.size(); i++) {

                vector<Device> devices;

#if defined(CL_HPP_ENABLE_EXCEPTIONS)
                try {
#endif

                    error = platforms[i].getDevices(type, &devices);

#if defined(CL_HPP_ENABLE_EXCEPTIONS)
                } catch (cl::Error& e) {
                    error = e.err();
                }
    // Catch if exceptions are enabled as we don't want to exit if first platform has no devices of type
    // We do error checking next anyway, and can throw there if needed
#endif

                // Only squash CL_SUCCESS and CL_DEVICE_NOT_FOUND
                if (error != CL_SUCCESS && error != CL_DEVICE_NOT_FOUND) {
                    detail::errHandler(error, __CREATE_CONTEXT_FROM_TYPE_ERR);
                    if (err != nullptr) {
                        *err = error;
                    }
                }

                if (devices.size() > 0) {
                    platform_id = (cl_context_properties)platforms[i]();
                    break;
                }
            }

            if (platform_id == 0) {
                detail::errHandler(CL_DEVICE_NOT_FOUND, __CREATE_CONTEXT_FROM_TYPE_ERR);
                if (err != nullptr) {
                    *err = CL_DEVICE_NOT_FOUND;
                }
                return;
            }

            prop[1] = platform_id;
            properties = &prop[0];
        }
#endif
        object_ = ::clCreateContextFromType(
            properties, type, notifyFptr, data, &error);

        detail::errHandler(error, __CREATE_CONTEXT_FROM_TYPE_ERR);
        if (err != nullptr) {
            *err = error;
        }
    }


    /*! \brief Returns a singleton context including all devices of CL_DEVICE_TYPE_DEFAULT.
     *
     *  \note All calls to this function return the same cl_context as the first.
     */
    static Context getDefault(cl_int * err = nullptr) 
    {
        std::call_once(default_initialized_, makeDefault);
        detail::errHandler(default_error_);
        if (err != nullptr) {
            *err = default_error_;
        }
        return default_;
    }

    /**
     * Modify the default context to be used by
     * subsequent operations.
     * Will only set the default if no default was previously created.
     * @return updated default context.
     *         Should be compared to the passed value to ensure that it was updated.
     */
    static Context setDefault(const Context &default_context)
    {
        std::call_once(default_initialized_, makeDefaultProvided, std::cref(default_context));
        detail::errHandler(default_error_);
        return default_;
    }

    //! \brief Default constructor - initializes to nullptr.
    Context() : detail::Wrapper<cl_type>() { }

    /*! \brief Constructor from cl_context - takes ownership.
     * 
     *  This effectively transfers ownership of a refcount on the cl_context
     *  into the new Context object.
     */
    explicit Context(const cl_context& context, bool retainObject = false) : 
        detail::Wrapper<cl_type>(context, retainObject) { }

    /*! \brief Assignment operator from cl_context - takes ownership.
     * 
     *  This effectively transfers ownership of a refcount on the rhs and calls
     *  clReleaseContext() on the value previously held by this instance.
     */
    Context& operator = (const cl_context& rhs)
    {
        detail::Wrapper<cl_type>::operator=(rhs);
        return *this;
    }

    //! \brief Wrapper for clGetContextInfo().
    template <typename T>
    cl_int getInfo(cl_context_info name, T* param) const
    {
        return detail::errHandler(
            detail::getInfo(&::clGetContextInfo, object_, name, param),
            __GET_CONTEXT_INFO_ERR);
    }

    //! \brief Wrapper for clGetContextInfo() that returns by value.
    template <cl_context_info name> typename
    detail::param_traits<detail::cl_context_info, name>::param_type
    getInfo(cl_int* err = nullptr) const
    {
        typename detail::param_traits<
            detail::cl_context_info, name>::param_type param;
        cl_int result = getInfo(name, &param);
        if (err != nullptr) {
            *err = result;
        }
        return param;
    }

    /*! \brief Gets a list of supported image formats.
     *  
     *  Wraps clGetSupportedImageFormats().
     */
    cl_int getSupportedImageFormats(
        cl_mem_flags flags,
        cl_mem_object_type type,
        vector<ImageFormat>* formats) const
    {
        cl_uint numEntries;
        
        if (!formats) {
            return CL_SUCCESS;
        }

        cl_int err = ::clGetSupportedImageFormats(
           object_, 
           flags,
           type, 
           0, 
           nullptr, 
           &numEntries);
        if (err != CL_SUCCESS) {
            return detail::errHandler(err, __GET_SUPPORTED_IMAGE_FORMATS_ERR);
        }

        if (numEntries > 0) {
            vector<ImageFormat> value(numEntries);
            err = ::clGetSupportedImageFormats(
                object_,
                flags,
                type,
                numEntries,
                (cl_image_format*)value.data(),
                nullptr);
            if (err != CL_SUCCESS) {
                return detail::errHandler(err, __GET_SUPPORTED_IMAGE_FORMATS_ERR);
            }

            formats->assign(value.begin(), value.end());
        }
        else {
            // If no values are being returned, ensure an empty vector comes back
            formats->clear();
        }

        return CL_SUCCESS;
    }

#if defined(cl_ext_image_requirements_info)
    template <typename T>
    cl_int getImageRequirementsInfoExt(cl_image_requirements_info_ext name,
        T* param,
        cl_mem_flags flags = 0,
        const cl_mem_properties* properties = nullptr,
        const ImageFormat* image_format = nullptr,
        const cl_image_desc* image_desc = nullptr) const
    {
        ImageRequirementsInfo imageInfo = {flags, properties, image_format, image_desc};

        return detail::errHandler(
            detail::getInfo(
                Context::getImageRequirementsInfoExtHelper, *this, imageInfo, name, param),
                __GET_IMAGE_REQUIREMENT_INFO_EXT_ERR);
    }

    template <cl_image_requirements_info_ext type> typename
    detail::param_traits<detail::cl_image_requirements_info_ext, type>::param_type
        getImageRequirementsInfoExt(cl_mem_flags flags = 0,
            const cl_mem_properties* properties = nullptr,
            const ImageFormat* image_format = nullptr,
            const cl_image_desc* image_desc = nullptr,
            cl_int* err = nullptr) const
    {
        typename detail::param_traits<
        detail::cl_image_requirements_info_ext, type>::param_type param;
        cl_int result = getImageRequirementsInfoExt(type, &param, flags, properties, image_format, image_desc);
        if (err != nullptr) {
            *err = result;
        }
        return param;
    }
#endif // cl_ext_image_requirements_info

#if CL_HPP_TARGET_OPENCL_VERSION >= 300
    /*! \brief  Registers a destructor callback function with a context.
     *
     *  Wraps clSetContextDestructorCallback().
     * 
     * Each call to this function registers the specified callback function on
     * a destructor callback stack associated with context. The registered
     * callback functions are called in the reverse order in which they were registered.
     * If a context callback function was specified when context was created,
     * it will not be called after any context destructor callback is called.
     */
    cl_int setDestructorCallback(
        void (CL_CALLBACK * pfn_notify)(cl_context, void *),
        void * user_data = nullptr)
    {
        return detail::errHandler(
            ::clSetContextDestructorCallback(
                object_,
                pfn_notify,
                user_data),
                __SET_CONTEXT_DESCTRUCTOR_CALLBACK_ERR);
    }
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 300
};

inline void Device::makeDefault()
{
    /* Throwing an exception from a call_once invocation does not do
    * what we wish, so we catch it and save the error.
    */
#if defined(CL_HPP_ENABLE_EXCEPTIONS)
    try
#endif
    {
        cl_int error = 0;

        Context context = Context::getDefault(&error);
        detail::errHandler(error, __CREATE_CONTEXT_ERR);

        if (error != CL_SUCCESS) {
            default_error_ = error;
        }
        else {
            default_ = context.getInfo<CL_CONTEXT_DEVICES>()[0];
            default_error_ = CL_SUCCESS;
        }
    }
#if defined(CL_HPP_ENABLE_EXCEPTIONS)
    catch (cl::Error &e) {
        default_error_ = e.err();
    }
#endif
}

CL_HPP_DEFINE_STATIC_MEMBER_ std::once_flag Context::default_initialized_;
CL_HPP_DEFINE_STATIC_MEMBER_ Context Context::default_;
CL_HPP_DEFINE_STATIC_MEMBER_ cl_int Context::default_error_ = CL_SUCCESS;

/*! \brief Class interface for cl_event.
 *
 *  \note Copies of these objects are shallow, meaning that the copy will refer
 *        to the same underlying cl_event as the original.  For details, see
 *        clRetainEvent() and clReleaseEvent().
 *
 *  \see cl_event
 */
class Event : public detail::Wrapper<cl_event>
{
public:
    //! \brief Default constructor - initializes to nullptr.
    Event() : detail::Wrapper<cl_type>() { }

    /*! \brief Constructor from cl_event - takes ownership.
     * 
     * \param retainObject will cause the constructor to retain its cl object.
     *                     Defaults to false to maintain compatibility with
     *                     earlier versions.
     *  This effectively transfers ownership of a refcount on the cl_event
     *  into the new Event object.
     */
    explicit Event(const cl_event& event, bool retainObject = false) : 
        detail::Wrapper<cl_type>(event, retainObject) { }

    /*! \brief Assignment operator from cl_event - takes ownership.
     *
     *  This effectively transfers ownership of a refcount on the rhs and calls
     *  clReleaseEvent() on the value previously held by this instance.
     */
    Event& operator = (const cl_event& rhs)
    {
        detail::Wrapper<cl_type>::operator=(rhs);
        return *this;
    }

    //! \brief Wrapper for clGetEventInfo().
    template <typename T>
    cl_int getInfo(cl_event_info name, T* param) const
    {
        return detail::errHandler(
            detail::getInfo(&call_clGetEventInfo, object_, name, param),
            __GET_EVENT_INFO_ERR);
    }

    //! \brief Wrapper for clGetEventInfo() that returns by value.
    template <cl_event_info name> typename
    detail::param_traits<detail::cl_event_info, name>::param_type
    getInfo(cl_int* err = nullptr) const
    {
        typename detail::param_traits<
            detail::cl_event_info, name>::param_type param;
        cl_int result = getInfo(name, &param);
        if (err != nullptr) {
            *err = result;
        }
        return param;
    }

    //! \brief Wrapper for clGetEventProfilingInfo().
    template <typename T>
    cl_int getProfilingInfo(cl_profiling_info name, T* param) const
    {
        return detail::errHandler(detail::getInfo(
            &call_clGetEventProfilingInfo, object_, name, param),
            __GET_EVENT_PROFILE_INFO_ERR);
    }

    //! \brief Wrapper for clGetEventProfilingInfo() that returns by value.
    template <cl_profiling_info name> typename
    detail::param_traits<detail::cl_profiling_info, name>::param_type
    getProfilingInfo(cl_int* err = nullptr) const
    {
        typename detail::param_traits<
            detail::cl_profiling_info, name>::param_type param;
        cl_int result = getProfilingInfo(name, &param);
        if (err != nullptr) {
            *err = result;
        }
        return param;
    }

    /*! \brief Blocks the calling thread until this event completes.
     * 
     *  Wraps clWaitForEvents().
     */
    cl_int wait() const
    {
        return detail::errHandler(
            call_clWaitForEvents(1, &object_),
            __WAIT_FOR_EVENTS_ERR);
    }

#if CL_HPP_TARGET_OPENCL_VERSION >= 110
    /*! \brief Registers a user callback function for a specific command execution status.
     *
     *  Wraps clSetEventCallback().
     */
    cl_int setCallback(
        cl_int type,
        void (CL_CALLBACK * pfn_notify)(cl_event, cl_int, void *),
        void * user_data = nullptr)
    {
        return detail::errHandler(
            call_clSetEventCallback(
                object_,
                type,
                pfn_notify,
                user_data), 
            __SET_EVENT_CALLBACK_ERR);
    }
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110

    /*! \brief Blocks the calling thread until every event specified is complete.
     * 
     *  Wraps clWaitForEvents().
     */
    static cl_int
    waitForEvents(const vector<Event>& events)
    {
        static_assert(sizeof(cl::Event) == sizeof(cl_event),
        "Size of cl::Event must be equal to size of cl_event");

        return detail::errHandler(
            ::clWaitForEvents(
                (cl_uint) events.size(), (events.size() > 0) ? (cl_event*)&events.front() : nullptr),
            __WAIT_FOR_EVENTS_ERR);
    }
};

#if CL_HPP_TARGET_OPENCL_VERSION >= 110
/*! \brief Class interface for user events (a subset of cl_event's).
 * 
 *  See Event for details about copy semantics, etc.
 */
class UserEvent : public Event
{
public:
    /*! \brief Constructs a user event on a given context.
     *
     *  Wraps clCreateUserEvent().
     */
    UserEvent(
        const Context& context,
        cl_int * err = nullptr)
    {
        cl_int error;
        object_ = ::clCreateUserEvent(
            context(),
            &error);

        detail::errHandler(error, __CREATE_USER_EVENT_ERR);
        if (err != nullptr) {
            *err = error;
        }
    }

    //! \brief Default constructor - initializes to nullptr.
    UserEvent() : Event() { }

    /*! \brief Sets the execution status of a user event object.
     *
     *  Wraps clSetUserEventStatus().
     */
    cl_int setStatus(cl_int status)
    {
        return detail::errHandler(
            call_clSetUserEventStatus(object_,status), 
            __SET_USER_EVENT_STATUS_ERR);
    }
};
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110

/*! \brief Blocks the calling thread until every event specified is complete.
 * 
 *  Wraps clWaitForEvents().
 */
inline static cl_int
WaitForEvents(const vector<Event>& events)
{
    return detail::errHandler(
        ::clWaitForEvents(
            (cl_uint) events.size(), (events.size() > 0) ? (cl_event*)&events.front() : nullptr),
        __WAIT_FOR_EVENTS_ERR);
}

/*! \brief Class interface for cl_mem.
 *
 *  \note Copies of these objects are shallow, meaning that the copy will refer
 *        to the same underlying cl_mem as the original.  For details, see
 *        clRetainMemObject() and clReleaseMemObject().
 *
 *  \see cl_mem
 */
class Memory : public detail::Wrapper<cl_mem>
{
public:
    //! \brief Default constructor - initializes to nullptr.
    Memory() : detail::Wrapper<cl_type>() { }

    /*! \brief Constructor from cl_mem - takes ownership.
     *
     *  Optionally transfer ownership of a refcount on the cl_mem
     *  into the new Memory object.
     *
     * \param retainObject will cause the constructor to retain its cl object.
     *                     Defaults to false to maintain compatibility with
     *                     earlier versions.
     *
     *  See Memory for further details.
     */
    explicit Memory(const cl_mem& memory, bool retainObject) :
        detail::Wrapper<cl_type>(memory, retainObject) { }

    /*! \brief Assignment operator from cl_mem - takes ownership.
     *
     *  This effectively transfers ownership of a refcount on the rhs and calls
     *  clReleaseMemObject() on the value previously held by this instance.
     */
    Memory& operator = (const cl_mem& rhs)
    {
        detail::Wrapper<cl_type>::operator=(rhs);
        return *this;
    }

    //! \brief Wrapper for clGetMemObjectInfo().
    template <typename T>
    cl_int getInfo(cl_mem_info name, T* param) const
    {
        return detail::errHandler(
            detail::getInfo(&call_clGetMemObjectInfo, object_, name, param),
            __GET_MEM_OBJECT_INFO_ERR);
    }

    //! \brief Wrapper for clGetMemObjectInfo() that returns by value.
    template <cl_mem_info name> typename
    detail::param_traits<detail::cl_mem_info, name>::param_type
    getInfo(cl_int* err = nullptr) const
    {
        typename detail::param_traits<
            detail::cl_mem_info, name>::param_type param;
        cl_int result = getInfo(name, &param);
        if (err != nullptr) {
            *err = result;
        }
        return param;
    }

#if CL_HPP_TARGET_OPENCL_VERSION >= 110
    /*! \brief Registers a callback function to be called when the memory object
     *         is no longer needed.
     *
     *  Wraps clSetMemObjectDestructorCallback().
     *
     *  Repeated calls to this function, for a given cl_mem value, will append
     *  to the list of functions called (in reverse order) when memory object's
     *  resources are freed and the memory object is deleted.
     *
     *  \note
     *  The registered callbacks are associated with the underlying cl_mem
     *  value - not the Memory class instance.
     */
    cl_int setDestructorCallback(
        void (CL_CALLBACK * pfn_notify)(cl_mem, void *),
        void * user_data = nullptr)
    {
        return detail::errHandler(
            ::clSetMemObjectDestructorCallback(
                object_,
                pfn_notify,
                user_data), 
            __SET_MEM_OBJECT_DESTRUCTOR_CALLBACK_ERR);
    }
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110

};

// Pre-declare copy functions
class Buffer;
template< typename IteratorType >
cl_int copy( IteratorType startIterator, IteratorType endIterator, cl::Buffer &buffer );
template< typename IteratorType >
cl_int copy( const cl::Buffer &buffer, IteratorType startIterator, IteratorType endIterator );
template< typename IteratorType >
cl_int copy( const CommandQueue &queue, IteratorType startIterator, IteratorType endIterator, cl::Buffer &buffer );
template< typename IteratorType >
cl_int copy( const CommandQueue &queue, const cl::Buffer &buffer, IteratorType startIterator, IteratorType endIterator );


#if CL_HPP_TARGET_OPENCL_VERSION >= 200
namespace detail
{
    class SVMTraitNull
    {
    public:
        static cl_svm_mem_flags getSVMMemFlags()
        {
            return 0;
        }
    };
} // namespace detail

template<class Trait = detail::SVMTraitNull>
class SVMTraitReadWrite
{
public:
    static cl_svm_mem_flags getSVMMemFlags()
    {
        return CL_MEM_READ_WRITE |
            Trait::getSVMMemFlags();
    }
};

template<class Trait = detail::SVMTraitNull>
class SVMTraitReadOnly
{
public:
    static cl_svm_mem_flags getSVMMemFlags()
    {
        return CL_MEM_READ_ONLY |
            Trait::getSVMMemFlags();
    }
};

template<class Trait = detail::SVMTraitNull>
class SVMTraitWriteOnly
{
public:
    static cl_svm_mem_flags getSVMMemFlags()
    {
        return CL_MEM_WRITE_ONLY |
            Trait::getSVMMemFlags();
    }
};

template<class Trait = SVMTraitReadWrite<>>
class SVMTraitCoarse
{
public:
    static cl_svm_mem_flags getSVMMemFlags()
    {
        return Trait::getSVMMemFlags();
    }
};

template<class Trait = SVMTraitReadWrite<>>
class SVMTraitFine
{
public:
    static cl_svm_mem_flags getSVMMemFlags()
    {
        return CL_MEM_SVM_FINE_GRAIN_BUFFER |
            Trait::getSVMMemFlags();
    }
};

template<class Trait = SVMTraitReadWrite<>>
class SVMTraitAtomic
{
public:
    static cl_svm_mem_flags getSVMMemFlags()
    {
        return
            CL_MEM_SVM_FINE_GRAIN_BUFFER |
            CL_MEM_SVM_ATOMICS |
            Trait::getSVMMemFlags();
    }
};

// Pre-declare SVM map function
template<typename T>
inline cl_int enqueueMapSVM(
    T* ptr,
    cl_bool blocking,
    cl_map_flags flags,
    size_type size,
    const vector<Event>* events = nullptr,
    Event* event = nullptr);

/**
 * STL-like allocator class for managing SVM objects provided for convenience.
 *
 * Note that while this behaves like an allocator for the purposes of constructing vectors and similar objects,
 * care must be taken when using with smart pointers.
 * The allocator should not be used to construct a unique_ptr if we are using coarse-grained SVM mode because
 * the coarse-grained management behaviour would behave incorrectly with respect to reference counting.
 *
 * Instead the allocator embeds a Deleter which may be used with unique_ptr and is used
 * with the allocate_shared and allocate_ptr supplied operations.
 */
template<typename T, class SVMTrait>
class SVMAllocator {
private:
    Context context_;

public:
    typedef T value_type;
    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef value_type& reference;
    typedef const value_type& const_reference;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;

    template<typename U>
    struct rebind
    {
        typedef SVMAllocator<U, SVMTrait> other;
    };

    template<typename U, typename V>
    friend class SVMAllocator;

    SVMAllocator() :
        context_(Context::getDefault())
    {
    }

    explicit SVMAllocator(cl::Context context) :
        context_(context)
    {
    }


    SVMAllocator(const SVMAllocator &other) :
        context_(other.context_)
    {
    }

    template<typename U>
    SVMAllocator(const SVMAllocator<U, SVMTrait> &other) :
        context_(other.context_)
    {
    }

    ~SVMAllocator()
    {
    }

    pointer address(reference r) noexcept
    {
        return std::addressof(r);
    }

    const_pointer address(const_reference r) noexcept
    {
        return std::addressof(r);
    }

    /**
     * Allocate an SVM pointer.
     *
     * If the allocator is coarse-grained, this will take ownership to allow
     * containers to correctly construct data in place. 
     */
    pointer allocate(
        size_type size,
        typename cl::SVMAllocator<void, SVMTrait>::const_pointer = 0,
        bool map = true)
    {
        // Allocate memory with default alignment matching the size of the type
        void* voidPointer =
            clSVMAlloc(
            context_(),
            SVMTrait::getSVMMemFlags(),
            size*sizeof(T),
            0);
        pointer retValue = reinterpret_cast<pointer>(
            voidPointer);
#if defined(CL_HPP_ENABLE_EXCEPTIONS)
        if (!retValue) {
            std::bad_alloc excep;
            throw excep;
        }
#endif // #if defined(CL_HPP_ENABLE_EXCEPTIONS)

        // If allocation was coarse-grained then map it
        if (map && !(SVMTrait::getSVMMemFlags() & CL_MEM_SVM_FINE_GRAIN_BUFFER)) {
            cl_int err = enqueueMapSVM(retValue, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, size*sizeof(T));
            if (err != CL_SUCCESS) {
                clSVMFree(context_(), retValue);
                retValue = nullptr;
#if defined(CL_HPP_ENABLE_EXCEPTIONS)
                std::bad_alloc excep;
                throw excep;
#endif
            }
        }

        // If exceptions disabled, return null pointer from allocator
        return retValue;
    }

    void deallocate(pointer p, size_type)
    {
        clSVMFree(context_(), p);
    }

    /**
     * Return the maximum possible allocation size.
     * This is the minimum of the maximum sizes of all devices in the context.
     */
    size_type max_size() const noexcept
    {
        size_type maxSize = std::numeric_limits<size_type>::max() / sizeof(T);

        for (const Device &d : context_.getInfo<CL_CONTEXT_DEVICES>()) {
            maxSize = std::min(
                maxSize, 
                static_cast<size_type>(d.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>()));
        }

        return maxSize;
    }

    template< class U, class... Args >
    void construct(U* p, Args&&... args)
    {
        new(p)T(args...);
    }

    template< class U >
    void destroy(U* p)
    {
        p->~U();
    }

    /**
     * Returns true if the contexts match.
     */
    inline bool operator==(SVMAllocator const& rhs)
    {
        return (context_==rhs.context_);
    }

    inline bool operator!=(SVMAllocator const& a)
    {
        return !operator==(a);
    }
}; // class SVMAllocator        return cl::pointer<T>(tmp, detail::Deleter<T, Alloc>{alloc, copies});


template<class SVMTrait>
class SVMAllocator<void, SVMTrait> {
public:
    typedef void value_type;
    typedef value_type* pointer;
    typedef const value_type* const_pointer;

    template<typename U>
    struct rebind
    {
        typedef SVMAllocator<U, SVMTrait> other;
    };

    template<typename U, typename V>
    friend class SVMAllocator;
};

#if !defined(CL_HPP_NO_STD_UNIQUE_PTR)
namespace detail
{
    template<class Alloc>
    class Deleter {
    private:
        Alloc alloc_;
        size_type copies_;

    public:
        typedef typename std::allocator_traits<Alloc>::pointer pointer;

        Deleter(const Alloc &alloc, size_type copies) : alloc_{ alloc }, copies_{ copies }
        {
        }

        void operator()(pointer ptr) const {
            Alloc tmpAlloc{ alloc_ };
            std::allocator_traits<Alloc>::destroy(tmpAlloc, std::addressof(*ptr));
            std::allocator_traits<Alloc>::deallocate(tmpAlloc, ptr, copies_);
        }
    };
} // namespace detail

/**
 * Allocation operation compatible with std::allocate_ptr.
 * Creates a unique_ptr<T> by default.
 * This requirement is to ensure that the control block is not
 * allocated in memory inaccessible to the host.
 */
template <class T, class Alloc, class... Args>
cl::pointer<T, detail::Deleter<Alloc>> allocate_pointer(const Alloc &alloc_, Args&&... args)
{
    Alloc alloc(alloc_);
    static const size_type copies = 1;

    // Ensure that creation of the management block and the
    // object are dealt with separately such that we only provide a deleter

    T* tmp = std::allocator_traits<Alloc>::allocate(alloc, copies);
    if (!tmp) {
#if defined(CL_HPP_ENABLE_EXCEPTIONS)
        std::bad_alloc excep;
        throw excep;
#else
        return nullptr;
#endif
    }

#if defined(CL_HPP_ENABLE_EXCEPTIONS)
    try
#endif
    {
        std::allocator_traits<Alloc>::construct(
            alloc,
            std::addressof(*tmp),
            std::forward<Args>(args)...);

        return cl::pointer<T, detail::Deleter<Alloc>>(tmp, detail::Deleter<Alloc>{alloc, copies});
    }
#if defined(CL_HPP_ENABLE_EXCEPTIONS)
    catch (std::bad_alloc&)
    {
        std::allocator_traits<Alloc>::deallocate(alloc, tmp, copies);
        throw;
    }
#endif
}

template< class T, class SVMTrait, class... Args >
cl::pointer<T, detail::Deleter<SVMAllocator<T, SVMTrait>>> allocate_svm(Args... args)
{
    SVMAllocator<T, SVMTrait> alloc;
    return cl::allocate_pointer<T>(alloc, args...);
}

template< class T, class SVMTrait, class... Args >
cl::pointer<T, detail::Deleter<SVMAllocator<T, SVMTrait>>> allocate_svm(const cl::Context &c, Args... args)
{
    SVMAllocator<T, SVMTrait> alloc(c);
    return cl::allocate_pointer<T>(alloc, args...);
}
#endif // #if !defined(CL_HPP_NO_STD_UNIQUE_PTR)

/*! \brief Vector alias to simplify contruction of coarse-grained SVM containers.
 * 
 */
template < class T >
using coarse_svm_vector = vector<T, cl::SVMAllocator<int, cl::SVMTraitCoarse<>>>;

/*! \brief Vector alias to simplify contruction of fine-grained SVM containers.
*
*/
template < class T >
using fine_svm_vector = vector<T, cl::SVMAllocator<int, cl::SVMTraitFine<>>>;

/*! \brief Vector alias to simplify contruction of fine-grained SVM containers that support platform atomics.
*
*/
template < class T >
using atomic_svm_vector = vector<T, cl::SVMAllocator<int, cl::SVMTraitAtomic<>>>;

#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200


/*! \brief Class interface for Buffer Memory Objects.
 * 
 *  See Memory for details about copy semantics, etc.
 *
 *  \see Memory
 */
class Buffer : public Memory
{
public:

    /*! \brief Constructs a Buffer in a specified context.
     *
     *  Wraps clCreateBuffer().
     *
     *  \param host_ptr Storage to be used if the CL_MEM_USE_HOST_PTR flag was
     *                  specified.  Note alignment & exclusivity requirements.
     */
    Buffer(
        const Context& context,
        cl_mem_flags flags,
        size_type size,
        void* host_ptr = nullptr,
        cl_int* err = nullptr)
    {
        cl_int error;
        object_ = ::clCreateBuffer(context(), flags, size, host_ptr, &error);

        detail::errHandler(error, __CREATE_BUFFER_ERR);
        if (err != nullptr) {
            *err = error;
        }
    }

#if CL_HPP_TARGET_OPENCL_VERSION >= 300
    /*! \brief Constructs a Buffer in a specified context and with specified properties.
     *
     *  Wraps clCreateBufferWithProperties().
     *
     *  \param properties Optional list of properties for the buffer object and
     *                    their corresponding values. The non-empty list must
     *                    end with 0. 
     *  \param host_ptr Storage to be used if the CL_MEM_USE_HOST_PTR flag was
     *                  specified. Note alignment & exclusivity requirements.
     */
    Buffer(
        const Context& context,
        const vector<cl_mem_properties>& properties,
        cl_mem_flags flags,
        size_type size,
        void* host_ptr = nullptr,
        cl_int* err = nullptr)
    {
        cl_int error;

        if (properties.empty()) {
            object_ = ::clCreateBufferWithProperties(context(), nullptr, flags,
                                                     size, host_ptr, &error);
        }
        else {
            object_ = ::clCreateBufferWithProperties(
                context(), properties.data(), flags, size, host_ptr, &error);
        }

        detail::errHandler(error, __CREATE_BUFFER_ERR);
        if (err != nullptr) {
            *err = error;
        }
    }
#endif

    /*! \brief Constructs a Buffer in the default context.
     *
     *  Wraps clCreateBuffer().
     *
     *  \param host_ptr Storage to be used if the CL_MEM_USE_HOST_PTR flag was
     *                  specified.  Note alignment & exclusivity requirements.
     *
     *  \see Context::getDefault()
     */
    Buffer(
        cl_mem_flags flags,
        size_type size,
        void* host_ptr = nullptr,
        cl_int* err = nullptr) : Buffer(Context::getDefault(err), flags, size, host_ptr, err) { }

#if CL_HPP_TARGET_OPENCL_VERSION >= 300
    /*! \brief Constructs a Buffer in the default context and with specified properties.
     *
     *  Wraps clCreateBufferWithProperties().
     *
     *  \param properties Optional list of properties for the buffer object and
     *                    their corresponding values. The non-empty list must
     *                    end with 0. 
     *  \param host_ptr Storage to be used if the CL_MEM_USE_HOST_PTR flag was
     *                  specified. Note alignment & exclusivity requirements.
     * 
     *  \see Context::getDefault()
     */
    Buffer(
        const vector<cl_mem_properties>& properties,
        cl_mem_flags flags,
        size_type size,
        void* host_ptr = nullptr,
        cl_int* err = nullptr) : Buffer(Context::getDefault(err), properties, flags, size, host_ptr, err) { }
#endif

    /*!
     * \brief Construct a Buffer from a host container via iterators.
     * IteratorType must be random access.
     * If useHostPtr is specified iterators must represent contiguous data.
     */
    template< typename IteratorType >
    Buffer(
        IteratorType startIterator,
        IteratorType endIterator,
        bool readOnly,
        bool useHostPtr = false,
        cl_int* err = nullptr)
    {
        typedef typename std::iterator_traits<IteratorType>::value_type DataType;
        cl_int error;

        cl_mem_flags flags = 0;
        if( readOnly ) {
            flags |= CL_MEM_READ_ONLY;
        }
        else {
            flags |= CL_MEM_READ_WRITE;
        }
        if( useHostPtr ) {
            flags |= CL_MEM_USE_HOST_PTR;
        }
        
        size_type size = sizeof(DataType)*(endIterator - startIterator);

        Context context = Context::getDefault(err);

        if( useHostPtr ) {
            object_ = ::clCreateBuffer(context(), flags, size, const_cast<DataType*>(&*startIterator), &error);
        } else {
            object_ = ::clCreateBuffer(context(), flags, size, 0, &error);
        }

        detail::errHandler(error, __CREATE_BUFFER_ERR);
        if (err != nullptr) {
            *err = error;
        }

        if( !useHostPtr ) {
            error = cl::copy(startIterator, endIterator, *this);
            detail::errHandler(error, __CREATE_BUFFER_ERR);
            if (err != nullptr) {
                *err = error;
            }
        }
    }

    /*!
     * \brief Construct a Buffer from a host container via iterators using a specified context.
     * IteratorType must be random access.
     * If useHostPtr is specified iterators must represent contiguous data.
     */
    template< typename IteratorType >
    Buffer(const Context &context, IteratorType startIterator, IteratorType endIterator,
        bool readOnly, bool useHostPtr = false, cl_int* err = nullptr);
    
    /*!
    * \brief Construct a Buffer from a host container via iterators using a specified queue.
    * If useHostPtr is specified iterators must be random access.
    */
    template< typename IteratorType >
    Buffer(const CommandQueue &queue, IteratorType startIterator, IteratorType endIterator,
        bool readOnly, bool useHostPtr = false, cl_int* err = nullptr);

    //! \brief Default constructor - initializes to nullptr.
    Buffer() : Memory() { }

    /*! \brief Constructor from cl_mem - takes ownership.
     *
     * \param retainObject will cause the constructor to retain its cl object.
     *                     Defaults to false to maintain compatibility with earlier versions.
     *
     *  See Memory for further details.
     */
    explicit Buffer(const cl_mem& buffer, bool retainObject = false) :
        Memory(buffer, retainObject) { }

    /*! \brief Assignment from cl_mem - performs shallow copy.
    *
    *  See Memory for further details.
    */
    Buffer& operator = (const cl_mem& rhs)
    {
        Memory::operator=(rhs);
        return *this;
    }


#if CL_HPP_TARGET_OPENCL_VERSION >= 110
    /*! \brief Creates a new buffer object from this.
     *
     *  Wraps clCreateSubBuffer().
     */
    Buffer createSubBuffer(
        cl_mem_flags flags,
        cl_buffer_create_type buffer_create_type,
        const void * buffer_create_info,
        cl_int * err = nullptr)
    {
        Buffer result;
        cl_int error;
        result.object_ = call_clCreateSubBuffer(
            object_, 
            flags, 
            buffer_create_type, 
            buffer_create_info, 
            &error);

        detail::errHandler(error, __CREATE_SUBBUFFER_ERR);
        if (err != nullptr) {
            *err = error;
        }

        return result;
    }
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
};

#if defined (CL_HPP_USE_DX_INTEROP)
/*! \brief Class interface for creating OpenCL buffers from ID3D10Buffer's.
 *
 *  This is provided to facilitate interoperability with Direct3D.
 * 
 *  See Memory for details about copy semantics, etc.
 *
 *  \see Memory
 */
class BufferD3D10 : public Buffer
{
public:
   

    /*! \brief Constructs a BufferD3D10, in a specified context, from a
     *         given ID3D10Buffer.
     *
     *  Wraps clCreateFromD3D10BufferKHR().
     */
    BufferD3D10(
        const Context& context,
        cl_mem_flags flags,
        ID3D10Buffer* bufobj,
        cl_int * err = nullptr) : pfn_clCreateFromD3D10BufferKHR(nullptr)
    {
        typedef CL_API_ENTRY cl_mem (CL_API_CALL *PFN_clCreateFromD3D10BufferKHR)(
            cl_context context, cl_mem_flags flags, ID3D10Buffer*  buffer,
            cl_int* errcode_ret);
        PFN_clCreateFromD3D10BufferKHR pfn_clCreateFromD3D10BufferKHR;
#if CL_HPP_TARGET_OPENCL_VERSION >= 120
        vector<cl_context_properties> props = context.getInfo<CL_CONTEXT_PROPERTIES>();
        cl_platform platform = nullptr;
        for( int i = 0; i < props.size(); ++i ) {
            if( props[i] == CL_CONTEXT_PLATFORM ) {
                platform = props[i+1];
            }
        }
        CL_HPP_INIT_CL_EXT_FCN_PTR_PLATFORM_(platform, clCreateFromD3D10BufferKHR);
#endif
#if CL_HPP_MINIMUM_OPENCL_VERSION < 120
        CL_HPP_INIT_CL_EXT_FCN_PTR_(clCreateFromD3D10BufferKHR);
#endif

        cl_int error;
        object_ = pfn_clCreateFromD3D10BufferKHR(
            context(),
            flags,
            bufobj,
            &error);

        // TODO: This should really have a D3D10 rerror code!
        detail::errHandler(error, __CREATE_GL_BUFFER_ERR);
        if (err != nullptr) {
            *err = error;
        }
    }

    //! \brief Default constructor - initializes to nullptr.
    BufferD3D10() : Buffer() { }

    /*! \brief Constructor from cl_mem - takes ownership.
     *
     * \param retainObject will cause the constructor to retain its cl object.
     *                     Defaults to false to maintain compatibility with 
     *                     earlier versions.
     *  See Memory for further details.
     */
    explicit BufferD3D10(const cl_mem& buffer, bool retainObject = false) : 
        Buffer(buffer, retainObject) { }

    /*! \brief Assignment from cl_mem - performs shallow copy.
     *
     *  See Memory for further details.
     */
    BufferD3D10& operator = (const cl_mem& rhs)
    {
        Buffer::operator=(rhs);
        return *this;
    }
};
#endif

/*! \brief Class interface for GL Buffer Memory Objects.
 *
 *  This is provided to facilitate interoperability with OpenGL.
 * 
 *  See Memory for details about copy semantics, etc.
 * 
 *  \see Memory
 */
class BufferGL : public Buffer
{
public:
    /*! \brief Constructs a BufferGL in a specified context, from a given
     *         GL buffer.
     *
     *  Wraps clCreateFromGLBuffer().
     */
    BufferGL(
        const Context& context,
        cl_mem_flags flags,
        cl_GLuint bufobj,
        cl_int * err = nullptr)
    {
        cl_int error;
        object_ = ::clCreateFromGLBuffer(
            context(),
            flags,
            bufobj,
            &error);

        detail::errHandler(error, __CREATE_GL_BUFFER_ERR);
        if (err != nullptr) {
            *err = error;
        }
    }

    //! \brief Default constructor - initializes to nullptr.
    BufferGL() : Buffer() { }

    /*! \brief Constructor from cl_mem - takes ownership.
     *
     * \param retainObject will cause the constructor to retain its cl object.
     *                     Defaults to false to maintain compatibility with
     *                     earlier versions.
     *  See Memory for further details.
     */
    explicit BufferGL(const cl_mem& buffer, bool retainObject = false) :
        Buffer(buffer, retainObject) { }

    /*! \brief Assignment from cl_mem - performs shallow copy.
     *
     *  See Memory for further details.
     */
    BufferGL& operator = (const cl_mem& rhs)
    {
        Buffer::operator=(rhs);
        return *this;
    }


    //! \brief Wrapper for clGetGLObjectInfo().
    cl_int getObjectInfo(
        cl_gl_object_type *type,
        cl_GLuint * gl_object_name)
    {
        return detail::errHandler(
            ::clGetGLObjectInfo(object_,type,gl_object_name),
            __GET_GL_OBJECT_INFO_ERR);
    }
};

/*! \brief Class interface for GL Render Buffer Memory Objects.
 *
 *  This is provided to facilitate interoperability with OpenGL.
 * 
 *  See Memory for details about copy semantics, etc.
 * 
 *  \see Memory
 */
class BufferRenderGL : public Buffer
{
public:
    /*! \brief Constructs a BufferRenderGL in a specified context, from a given
     *         GL Renderbuffer.
     *
     *  Wraps clCreateFromGLRenderbuffer().
     */
    BufferRenderGL(
        const Context& context,
        cl_mem_flags flags,
        cl_GLuint bufobj,
        cl_int * err = nullptr)
    {
        cl_int error;
        object_ = ::clCreateFromGLRenderbuffer(
            context(),
            flags,
            bufobj,
            &error);

        detail::errHandler(error, __CREATE_GL_RENDER_BUFFER_ERR);
        if (err != nullptr) {
            *err = error;
        }
    }

    //! \brief Default constructor - initializes to nullptr.
    BufferRenderGL() : Buffer() { }

    /*! \brief Constructor from cl_mem - takes ownership.
     *
     * \param retainObject will cause the constructor to retain its cl object.
     *                     Defaults to false to maintain compatibility with 
     *                     earlier versions.
     *  See Memory for further details.
     */
    explicit BufferRenderGL(const cl_mem& buffer, bool retainObject = false) :
        Buffer(buffer, retainObject) { }

    /*! \brief Assignment from cl_mem - performs shallow copy.
     *
     *  See Memory for further details.
     */
    BufferRenderGL& operator = (const cl_mem& rhs)
    {
        Buffer::operator=(rhs);
        return *this;
    }


    //! \brief Wrapper for clGetGLObjectInfo().
    cl_int getObjectInfo(
        cl_gl_object_type *type,
        cl_GLuint * gl_object_name)
    {
        return detail::errHandler(
            ::clGetGLObjectInfo(object_,type,gl_object_name),
            __GET_GL_OBJECT_INFO_ERR);
    }
};

/*! \brief C++ base class for Image Memory objects.
 *
 *  See Memory for details about copy semantics, etc.
 * 
 *  \see Memory
 */
class Image : public Memory
{
protected:
    //! \brief Default constructor - initializes to nullptr.
    Image() : Memory() { }

    /*! \brief Constructor from cl_mem - takes ownership.
     *
     * \param retainObject will cause the constructor to retain its cl object.
     *                     Defaults to false to maintain compatibility with
     *                     earlier versions.
     *  See Memory for further details.
     */
    explicit Image(const cl_mem& image, bool retainObject = false) :
        Memory(image, retainObject) { }

    /*! \brief Assignment from cl_mem - performs shallow copy.
     *
     *  See Memory for further details.
     */
    Image& operator = (const cl_mem& rhs)
    {
        Memory::operator=(rhs);
        return *this;
    }


public:
    //! \brief Wrapper for clGetImageInfo().
    template <typename T>
    cl_int getImageInfo(cl_image_info name, T* param) const
    {
        return detail::errHandler(
            detail::getInfo(&::call_clGetImageInfo, object_, name, param),
            __GET_IMAGE_INFO_ERR);
    }
    
    //! \brief Wrapper for clGetImageInfo() that returns by value.
    template <cl_image_info name> typename
    detail::param_traits<detail::cl_image_info, name>::param_type
    getImageInfo(cl_int* err = nullptr) const
    {
        typename detail::param_traits<
            detail::cl_image_info, name>::param_type param;
        cl_int result = getImageInfo(name, &param);
        if (err != nullptr) {
            *err = result;
        }
        return param;
    }
};

#if CL_HPP_TARGET_OPENCL_VERSION >= 120
/*! \brief Class interface for 1D Image Memory objects.
 *
 *  See Memory for details about copy semantics, etc.
 * 
 *  \see Memory
 */
class Image1D : public Image
{
public:
    /*! \brief Constructs a 1D Image in a specified context.
     *
     *  Wraps clCreateImage().
     */
    Image1D(
        const Context& context,
        cl_mem_flags flags,
        ImageFormat format,
        size_type width,
        void* host_ptr = nullptr,
        cl_int* err = nullptr)
    {
        cl_int error;

        cl_image_desc desc = {};
        desc.image_type = CL_MEM_OBJECT_IMAGE1D;
        desc.image_width = width;

        object_ = ::clCreateImage(
            context(), 
            flags, 
            &format, 
            &desc, 
            host_ptr, 
            &error);

        detail::errHandler(error, __CREATE_IMAGE_ERR);
        if (err != nullptr) {
            *err = error;
        }
    }

    //! \brief Default constructor - initializes to nullptr.
    Image1D() { }

#if CL_HPP_TARGET_OPENCL_VERSION >= 300
    /*! \brief Constructs a Image1D with specified properties.
     *
     *  Wraps clCreateImageWithProperties().
     *
     *  \param properties Optional list of properties for the image object and
     *                    their corresponding values. The non-empty list must
     *                    end with 0.
     *  \param host_ptr Storage to be used if the CL_MEM_USE_HOST_PTR flag was
     *                  specified. Note alignment & exclusivity requirements.
     */
    Image1D(const Context &context, const vector<cl_mem_properties> &properties,
            cl_mem_flags flags, ImageFormat format, size_type width,
            void *host_ptr = nullptr, cl_int *err = nullptr) {
      cl_int error;

      cl_image_desc desc = {};
      desc.image_type = CL_MEM_OBJECT_IMAGE1D;
      desc.image_width = width;

      if (properties.empty()) {
        object_ = ::clCreateImageWithProperties(
            context(), nullptr, flags, &format, &desc, host_ptr, &error);
      } else {
        object_ =
            ::clCreateImageWithProperties(context(), properties.data(), flags,
                                          &format, &desc, host_ptr, &error);
      }

      detail::errHandler(error, __CREATE_IMAGE_ERR);
      if (err != nullptr) {
        *err = error;
      }
    }
#endif //#if CL_HPP_TARGET_OPENCL_VERSION >= 300

    /*! \brief Constructor from cl_mem - takes ownership.
     *
     * \param retainObject will cause the constructor to retain its cl object.
     *                     Defaults to false to maintain compatibility with
     *                     earlier versions.
     *  See Memory for further details.
     */
    explicit Image1D(const cl_mem& image1D, bool retainObject = false) :
        Image(image1D, retainObject) { }

    /*! \brief Assignment from cl_mem - performs shallow copy.
     *
     *  See Memory for further details.
     */
    Image1D& operator = (const cl_mem& rhs)
    {
        Image::operator=(rhs);
        return *this;
    }


};

/*! \class Image1DBuffer
 * \brief Image interface for 1D buffer images.
 */
class Image1DBuffer : public Image
{
public:
    Image1DBuffer(
        const Context& context,
        cl_mem_flags flags,
        ImageFormat format,
        size_type width,
        const Buffer &buffer,
        cl_int* err = nullptr)
    {
        cl_int error;

        cl_image_desc desc = {};
        desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        desc.image_width = width;
        desc.buffer = buffer();

        object_ = ::clCreateImage(
            context(), 
            flags, 
            &format, 
            &desc, 
            nullptr, 
            &error);

        detail::errHandler(error, __CREATE_IMAGE_ERR);
        if (err != nullptr) {
            *err = error;
        }
    }

    Image1DBuffer() { }

#if CL_HPP_TARGET_OPENCL_VERSION >= 300
    /*! \brief Constructs a Image1DBuffer with specified properties.
     *
     *  Wraps clCreateImageWithProperties().
     *
     *  \param properties Optional list of properties for the image object and
     *                    their corresponding values. The non-empty list must
     *                    end with 0.
     *  \param buffer Refer to a valid buffer or image memory object.
     */
    Image1DBuffer(const Context &context,
                  const vector<cl_mem_properties> &properties,
                  cl_mem_flags flags, ImageFormat format, size_type width,
                  const Buffer &buffer, cl_int *err = nullptr) {
      cl_int error;

      cl_image_desc desc = {};
      desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
      desc.image_width = width;
      desc.buffer = buffer();

      if (properties.empty()) {
        object_ = ::clCreateImageWithProperties(
            context(), nullptr, flags, &format, &desc, nullptr, &error);
      } else {
        object_ =
            ::clCreateImageWithProperties(context(), properties.data(), flags,
                                          &format, &desc, nullptr, &error);
      }

      detail::errHandler(error, __CREATE_IMAGE_ERR);
      if (err != nullptr) {
        *err = error;
      }
    }
#endif //#if CL_HPP_TARGET_OPENCL_VERSION >= 300

    /*! \brief Constructor from cl_mem - takes ownership.
     *
     * \param retainObject will cause the constructor to retain its cl object.
     *                     Defaults to false to maintain compatibility with
     *                     earlier versions.
     *  See Memory for further details.
     */
    explicit Image1DBuffer(const cl_mem& image1D, bool retainObject = false) :
        Image(image1D, retainObject) { }

    Image1DBuffer& operator = (const cl_mem& rhs)
    {
        Image::operator=(rhs);
        return *this;
    }
};

/*! \class Image1DArray
 * \brief Image interface for arrays of 1D images.
 */
class Image1DArray : public Image
{
public:
    Image1DArray(
        const Context& context,
        cl_mem_flags flags,
        ImageFormat format,
        size_type arraySize,
        size_type width,
        size_type rowPitch,
        void* host_ptr = nullptr,
        cl_int* err = nullptr)
    {
        cl_int error;

        cl_image_desc desc = {};
        desc.image_type = CL_MEM_OBJECT_IMAGE1D_ARRAY;
        desc.image_width = width;
        desc.image_array_size = arraySize;
        desc.image_row_pitch = rowPitch;

        object_ = ::clCreateImage(
            context(), 
            flags, 
            &format, 
            &desc, 
            host_ptr, 
            &error);

        detail::errHandler(error, __CREATE_IMAGE_ERR);
        if (err != nullptr) {
            *err = error;
        }
    }

    Image1DArray() { }

#if CL_HPP_TARGET_OPENCL_VERSION >= 300
    /*! \brief Constructs a Image1DArray with specified properties.
     *
     *  Wraps clCreateImageWithProperties().
     *
     *  \param properties Optional list of properties for the image object and
     *                    their corresponding values. The non-empty list must
     *                    end with 0.
     *  \param host_ptr Storage to be used if the CL_MEM_USE_HOST_PTR flag was
     *                  specified. Note alignment & exclusivity requirements.
     */
    Image1DArray(const Context &context,
                 const vector<cl_mem_properties> &properties,
                 cl_mem_flags flags, ImageFormat format, size_type arraySize,
                 size_type width, size_type rowPitch = 0,
                 void *host_ptr = nullptr, cl_int *err = nullptr) {
      cl_int error;

      cl_image_desc desc = {};
      desc.image_type = CL_MEM_OBJECT_IMAGE1D_ARRAY;
      desc.image_width = width;
      desc.image_array_size = arraySize;
      desc.image_row_pitch = rowPitch;

      if (properties.empty()) {
        object_ = ::clCreateImageWithProperties(
            context(), nullptr, flags, &format, &desc, host_ptr, &error);
      } else {
        object_ =
            ::clCreateImageWithProperties(context(), properties.data(), flags,
                                          &format, &desc, host_ptr, &error);
      }

      detail::errHandler(error, __CREATE_IMAGE_ERR);
      if (err != nullptr) {
        *err = error;
      }
    }
#endif //#if CL_HPP_TARGET_OPENCL_VERSION >= 300

    /*! \brief Constructor from cl_mem - takes ownership.
     *
     * \param retainObject will cause the constructor to retain its cl object.
     *                     Defaults to false to maintain compatibility with
     *                     earlier versions.
     *  See Memory for further details.
     */
    explicit Image1DArray(const cl_mem& imageArray, bool retainObject = false) :
        Image(imageArray, retainObject) { }


    Image1DArray& operator = (const cl_mem& rhs)
    {
        Image::operator=(rhs);
        return *this;
    }


};
#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 120


/*! \brief Class interface for 2D Image Memory objects.
 *
 *  See Memory for details about copy semantics, etc.
 * 
 *  \see Memory
 */
class Image2D : public Image
{
public:
    /*! \brief Constructs a 2D Image in a specified context.
     *
     *  Wraps clCreateImage().
     */
    Image2D(
        const Context& context,
        cl_mem_flags flags,
        ImageFormat format,
        size_type width,
        size_type height,
        size_type row_pitch = 0,
        void* host_ptr = nullptr,
        cl_int* err = nullptr)
    {
        cl_int error;
        bool useCreateImage;

#if CL_HPP_TARGET_OPENCL_VERSION >= 120 && CL_HPP_MINIMUM_OPENCL_VERSION < 120
        // Run-time decision based on the actual platform
        {
            cl_uint version = detail::getContextPlatformVersion(context());
            useCreateImage = (version >= 0x10002); // OpenCL 1.2 or above
        }
#elif CL_HPP_TARGET_OPENCL_VERSION >= 120
        useCreateImage = true;
#else
        useCreateImage = false;
#endif

#if CL_HPP_TARGET_OPENCL_VERSION >= 120
        if (useCreateImage)
        {
            cl_image_desc desc = {};
            desc.image_type = CL_MEM_OBJECT_IMAGE2D;
            desc.image_width = width;
            desc.image_height = height;
            desc.image_row_pitch = row_pitch;

            object_ = ::clCreateImage(
                context(),
                flags,
                &format,
                &desc,
                host_ptr,
                &error);

            detail::errHandler(error, __CREATE_IMAGE_ERR);
            if (err != nullptr) {
                *err = error;
            }
        }
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
#if CL_HPP_MINIMUM_OPENCL_VERSION < 120
        if (!useCreateImage)
        {
            object_ = ::clCreateImage2D(
                context(), flags,&format, width, height, row_pitch, host_ptr, &error);

            detail::errHandler(error, __CREATE_IMAGE2D_ERR);
            if (err != nullptr) {
                *err = error;
            }
        }
#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 120
    }

#if CL_HPP_TARGET_OPENCL_VERSION >= 120
    /*! \brief Constructs a 2D Image from a buffer.
    * \note This will share storage with the underlying buffer.
    *
    *  Requires OpenCL 2.0 or newer or OpenCL 1.2 and the 
    *  cl_khr_image2d_from_buffer extension.
    *
    *  Wraps clCreateImage().
    */
    Image2D(
        const Context& context,
        ImageFormat format,
        const Buffer &sourceBuffer,
        size_type width,
        size_type height,
        size_type row_pitch = 0,
        cl_int* err = nullptr)
    {
        cl_int error;

        cl_image_desc desc = {};
        desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        desc.image_width = width;
        desc.image_height = height;
        desc.image_row_pitch = row_pitch;
        desc.buffer = sourceBuffer();

        object_ = ::clCreateImage(
            context(),
            0, // flags inherited from buffer
            &format,
            &desc,
            nullptr,
            &error);

        detail::errHandler(error, __CREATE_IMAGE_ERR);
        if (err != nullptr) {
            *err = error;
        }
    }
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120

#if CL_HPP_TARGET_OPENCL_VERSION >= 200
    /*! \brief Constructs a 2D Image from an image.
    * \note This will share storage with the underlying image but may
    *       reinterpret the channel order and type.
    *
    * The image will be created matching with a descriptor matching the source. 
    *
    * \param order is the channel order to reinterpret the image data as.
    *              The channel order may differ as described in the OpenCL 
    *              2.0 API specification.
    *
    * Wraps clCreateImage().
    */
    Image2D(
        const Context& context,
        cl_channel_order order,
        const Image &sourceImage,
        cl_int* err = nullptr)
    {
        cl_int error;

        // Descriptor fields have to match source image
        size_type sourceWidth = 
            sourceImage.getImageInfo<CL_IMAGE_WIDTH>();
        size_type sourceHeight = 
            sourceImage.getImageInfo<CL_IMAGE_HEIGHT>();
        size_type sourceRowPitch =
            sourceImage.getImageInfo<CL_IMAGE_ROW_PITCH>();
        cl_uint sourceNumMIPLevels =
            sourceImage.getImageInfo<CL_IMAGE_NUM_MIP_LEVELS>();
        cl_uint sourceNumSamples =
            sourceImage.getImageInfo<CL_IMAGE_NUM_SAMPLES>();
        cl_image_format sourceFormat =
            sourceImage.getImageInfo<CL_IMAGE_FORMAT>();

        // Update only the channel order. 
        // Channel format inherited from source.
        sourceFormat.image_channel_order = order;

        cl_image_desc desc = {};
        desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        desc.image_width = sourceWidth;
        desc.image_height = sourceHeight;
        desc.image_row_pitch = sourceRowPitch;
        desc.num_mip_levels = sourceNumMIPLevels;
        desc.num_samples = sourceNumSamples;
        desc.buffer = sourceImage();

        object_ = ::clCreateImage(
            context(),
            0, // flags should be inherited from mem_object
            &sourceFormat,
            &desc,
            nullptr,
            &error);

        detail::errHandler(error, __CREATE_IMAGE_ERR);
        if (err != nullptr) {
            *err = error;
        }
    }
#endif //#if CL_HPP_TARGET_OPENCL_VERSION >= 200

#if CL_HPP_TARGET_OPENCL_VERSION >= 300
    /*! \brief Constructs a Image2D with specified properties.
     *
     *  Wraps clCreateImageWithProperties().
     *
     *  \param properties Optional list of properties for the image object and
     *                    their corresponding values. The non-empty list must
     *                    end with 0.
     *  \param host_ptr Storage to be used if the CL_MEM_USE_HOST_PTR flag was
     *                  specified. Note alignment & exclusivity requirements.
     */
    Image2D(const Context &context, const vector<cl_mem_properties> &properties,
            cl_mem_flags flags, ImageFormat format, size_type width,
            size_type height, size_type row_pitch = 0, void *host_ptr = nullptr,
            cl_int *err = nullptr) {
      cl_int error;

      cl_image_desc desc = {};
      desc.image_type = CL_MEM_OBJECT_IMAGE2D;
      desc.image_width = width;
      desc.image_height = height;
      desc.image_row_pitch = row_pitch;

      if (properties.empty()) {
        object_ = ::clCreateImageWithProperties(
            context(), nullptr, flags, &format, &desc, host_ptr, &error);
      } else {
        object_ =
            ::clCreateImageWithProperties(context(), properties.data(), flags,
                                          &format, &desc, host_ptr, &error);
      }

      detail::errHandler(error, __CREATE_IMAGE_ERR);
      if (err != nullptr) {
        *err = error;
      }
    }

    /*! \brief Constructs a Image2D with specified properties.
     *
     *  Wraps clCreateImageWithProperties().
     *
     *  \param properties Optional list of properties for the image object and
     *                    their corresponding values. The non-empty list must
     *                    end with 0.
     *  \param buffer Refer to a valid buffer or image memory object.
     */
    Image2D(const Context &context, const vector<cl_mem_properties> &properties,
            cl_mem_flags flags, ImageFormat format, const Buffer &buffer,
            size_type width, size_type height, size_type row_pitch = 0,
            cl_int *err = nullptr) {
      cl_int error;

      cl_image_desc desc = {};
      desc.image_type = CL_MEM_OBJECT_IMAGE2D;
      desc.image_width = width;
      desc.image_height = height;
      desc.image_row_pitch = row_pitch;
      desc.buffer = buffer();

      if (properties.empty()) {
        object_ = ::clCreateImageWithProperties(
            context(), nullptr, flags, &format, &desc, nullptr, &error);
      } else {
        object_ =
            ::clCreateImageWithProperties(context(), properties.data(), flags,
                                          &format, &desc, nullptr, &error);
      }

      detail::errHandler(error, __CREATE_IMAGE_ERR);
      if (err != nullptr) {
        *err = error;
      }
    }

#endif //#if CL_HPP_TARGET_OPENCL_VERSION >= 300

    //! \brief Default constructor - initializes to nullptr.
    Image2D() { }

    /*! \brief Constructor from cl_mem - takes ownership.
     *
     * \param retainObject will cause the constructor to retain its cl object.
     *                     Defaults to false to maintain compatibility with
     *                     earlier versions.
     *  See Memory for further details.
     */
    explicit Image2D(const cl_mem& image2D, bool retainObject = false) :
        Image(image2D, retainObject) { }

    /*! \brief Assignment from cl_mem - performs shallow copy.
     *
     *  See Memory for further details.
     */
    Image2D& operator = (const cl_mem& rhs)
    {
        Image::operator=(rhs);
        return *this;
    }
};


#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
/*! \brief Class interface for GL 2D Image Memory objects.
 *
 *  This is provided to facilitate interoperability with OpenGL.
 * 
 *  See Memory for details about copy semantics, etc.
 * 
 *  \see Memory
 *  \note Deprecated for OpenCL 1.2. Please use ImageGL instead.
 */
class CL_API_PREFIX__VERSION_1_1_DEPRECATED Image2DGL : public Image2D 
{
public:
    /*! \brief Constructs an Image2DGL in a specified context, from a given
     *         GL Texture.
     *
     *  Wraps clCreateFromGLTexture2D().
     */
    Image2DGL(
        const Context& context,
        cl_mem_flags flags,
        cl_GLenum target,
        cl_GLint  miplevel,
        cl_GLuint texobj,
        cl_int * err = nullptr)
    {
        cl_int error;
        object_ = ::clCreateFromGLTexture2D(
            context(),
            flags,
            target,
            miplevel,
            texobj,
            &error);

        detail::errHandler(error, __CREATE_GL_TEXTURE_2D_ERR);
        if (err != nullptr) {
            *err = error;
        }

    }
    
    //! \brief Default constructor - initializes to nullptr.
    Image2DGL() : Image2D() { }

    /*! \brief Constructor from cl_mem - takes ownership.
     *
     * \param retainObject will cause the constructor to retain its cl object.
     *                     Defaults to false to maintain compatibility with
     *                     earlier versions.
     *  See Memory for further details.
     */
    explicit Image2DGL(const cl_mem& image, bool retainObject = false) : 
        Image2D(image, retainObject) { }

    /*! \brief Assignment from cl_mem - performs shallow copy.
     *c
     *  See Memory for further details.
     */
    Image2DGL& operator = (const cl_mem& rhs)
    {
        Image2D::operator=(rhs);
        return *this;
    }



} CL_API_SUFFIX__VERSION_1_1_DEPRECATED;
#endif // CL_USE_DEPRECATED_OPENCL_1_1_APIS

#if CL_HPP_TARGET_OPENCL_VERSION >= 120
/*! \class Image2DArray
 * \brief Image interface for arrays of 2D images.
 */
class Image2DArray : public Image
{
public:
    Image2DArray(
        const Context& context,
        cl_mem_flags flags,
        ImageFormat format,
        size_type arraySize,
        size_type width,
        size_type height,
        size_type rowPitch,
        size_type slicePitch,
        void* host_ptr = nullptr,
        cl_int* err = nullptr)
    {
        cl_int error;

        cl_image_desc desc = {};
        desc.image_type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
        desc.image_width = width;
        desc.image_height = height;
        desc.image_array_size = arraySize;
        desc.image_row_pitch = rowPitch;
        desc.image_slice_pitch = slicePitch;

        object_ = ::clCreateImage(
            context(), 
            flags, 
            &format, 
            &desc, 
            host_ptr, 
            &error);

        detail::errHandler(error, __CREATE_IMAGE_ERR);
        if (err != nullptr) {
            *err = error;
        }
    }

#if CL_HPP_TARGET_OPENCL_VERSION >= 300
    /*! \brief Constructs a Image2DArray with specified properties.
     *
     *  Wraps clCreateImageWithProperties().
     *
     *  \param properties Optional list of properties for the image object and
     *                    their corresponding values. The non-empty list must
     *                    end with 0.
     *  \param host_ptr Storage to be used if the CL_MEM_USE_HOST_PTR flag was
     *                  specified. Note alignment & exclusivity requirements.
     */
    Image2DArray(const Context &context,
                 const vector<cl_mem_properties> &properties,
                 cl_mem_flags flags, ImageFormat format, size_type arraySize,
                 size_type width, size_type height, size_type rowPitch = 0,
                 size_type slicePitch = 0, void *host_ptr = nullptr,
                 cl_int *err = nullptr) {
      cl_int error;

      cl_image_desc desc = {};
      desc.image_type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
      desc.image_width = width;
      desc.image_height = height;
      desc.image_array_size = arraySize;
      desc.image_row_pitch = rowPitch;
      desc.image_slice_pitch = slicePitch;

      if (properties.empty()) {
        object_ = ::clCreateImageWithProperties(
            context(), nullptr, flags, &format, &desc, host_ptr, &error);
      } else {
        object_ =
            ::clCreateImageWithProperties(context(), properties.data(), flags,
                                          &format, &desc, host_ptr, &error);
      }

      detail::errHandler(error, __CREATE_IMAGE_ERR);
      if (err != nullptr) {
        *err = error;
      }
    }
#endif //#if CL_HPP_TARGET_OPENCL_VERSION >= 300

    Image2DArray() { }
    
    /*! \brief Constructor from cl_mem - takes ownership.
     *
     * \param retainObject will cause the constructor to retain its cl object.
     *                     Defaults to false to maintain compatibility with
     *                     earlier versions.
     *  See Memory for further details.
     */
    explicit Image2DArray(const cl_mem& imageArray, bool retainObject = false) : Image(imageArray, retainObject) { }

    Image2DArray& operator = (const cl_mem& rhs)
    {
        Image::operator=(rhs);
        return *this;
    }

};
#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 120

/*! \brief Class interface for 3D Image Memory objects.
 *
 *  See Memory for details about copy semantics, etc.
 * 
 *  \see Memory
 */
class Image3D : public Image
{
public:
    /*! \brief Constructs a 3D Image in a specified context.
     *
     *  Wraps clCreateImage().
     */
    Image3D(
        const Context& context,
        cl_mem_flags flags,
        ImageFormat format,
        size_type width,
        size_type height,
        size_type depth,
        size_type row_pitch = 0,
        size_type slice_pitch = 0,
        void* host_ptr = nullptr,
        cl_int* err = nullptr)
    {
        cl_int error;
        bool useCreateImage;

#if CL_HPP_TARGET_OPENCL_VERSION >= 120 && CL_HPP_MINIMUM_OPENCL_VERSION < 120
        // Run-time decision based on the actual platform
        {
            cl_uint version = detail::getContextPlatformVersion(context());
            useCreateImage = (version >= 0x10002); // OpenCL 1.2 or above
        }
#elif CL_HPP_TARGET_OPENCL_VERSION >= 120
        useCreateImage = true;
#else
        useCreateImage = false;
#endif

#if CL_HPP_TARGET_OPENCL_VERSION >= 120
        if (useCreateImage)
        {
            cl_image_desc desc = {};
            desc.image_type = CL_MEM_OBJECT_IMAGE3D;
            desc.image_width = width;
            desc.image_height = height;
            desc.image_depth = depth;
            desc.image_row_pitch = row_pitch;
            desc.image_slice_pitch = slice_pitch;

            object_ = ::clCreateImage(
                context(), 
                flags, 
                &format, 
                &desc, 
                host_ptr, 
                &error);

            detail::errHandler(error, __CREATE_IMAGE_ERR);
            if (err != nullptr) {
                *err = error;
            }
        }
#endif  // CL_HPP_TARGET_OPENCL_VERSION >= 120
#if CL_HPP_MINIMUM_OPENCL_VERSION < 120
        if (!useCreateImage)
        {
            object_ = ::clCreateImage3D(
                context(), flags, &format, width, height, depth, row_pitch,
                slice_pitch, host_ptr, &error);

            detail::errHandler(error, __CREATE_IMAGE3D_ERR);
            if (err != nullptr) {
                *err = error;
            }
        }
#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 120
    }

#if CL_HPP_TARGET_OPENCL_VERSION >= 300
    /*! \brief Constructs a Image3D with specified properties.
     *
     *  Wraps clCreateImageWithProperties().
     *
     *  \param properties Optional list of properties for the image object and
     *                    their corresponding values. The non-empty list must
     *                    end with 0.
     *  \param host_ptr Storage to be used if the CL_MEM_USE_HOST_PTR flag was
     *                  specified. Note alignment & exclusivity requirements.
     */
    Image3D(const Context &context, const vector<cl_mem_properties> &properties,
            cl_mem_flags flags, ImageFormat format, size_type width,
            size_type height, size_type depth, size_type row_pitch = 0,
            size_type slice_pitch = 0, void *host_ptr = nullptr,
            cl_int *err = nullptr) {
      cl_int error;

      cl_image_desc desc = {};
      desc.image_type = CL_MEM_OBJECT_IMAGE3D;
      desc.image_width = width;
      desc.image_height = height;
      desc.image_depth = depth;
      desc.image_row_pitch = row_pitch;
      desc.image_slice_pitch = slice_pitch;

      if (properties.empty()) {
        object_ = ::clCreateImageWithProperties(
            context(), nullptr, flags, &format, &desc, host_ptr, &error);
      } else {
        object_ =
            ::clCreateImageWithProperties(context(), properties.data(), flags,
                                          &format, &desc, host_ptr, &error);
      }

      detail::errHandler(error, __CREATE_IMAGE_ERR);
      if (err != nullptr) {
        *err = error;
      }
    }
#endif //#if CL_HPP_TARGET_OPENCL_VERSION >= 300

    //! \brief Default constructor - initializes to nullptr.
    Image3D() : Image() { }

    /*! \brief Constructor from cl_mem - takes ownership.
     *
     * \param retainObject will cause the constructor to retain its cl object.
     *                     Defaults to false to maintain compatibility with
     *                     earlier versions.
     *  See Memory for further details.
     */
    explicit Image3D(const cl_mem& image3D, bool retainObject = false) : 
        Image(image3D, retainObject) { }

    /*! \brief Assignment from cl_mem - performs shallow copy.
     *
     *  See Memory for further details.
     */
    Image3D& operator = (const cl_mem& rhs)
    {
        Image::operator=(rhs);
        return *this;
    }

};

#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
/*! \brief Class interface for GL 3D Image Memory objects.
 *
 *  This is provided to facilitate interoperability with OpenGL.
 * 
 *  See Memory for details about copy semantics, etc.
 * 
 *  \see Memory
 */
class Image3DGL : public Image3D
{
public:
    /*! \brief Constructs an Image3DGL in a specified context, from a given
     *         GL Texture.
     *
     *  Wraps clCreateFromGLTexture3D().
     */
    Image3DGL(
        const Context& context,
        cl_mem_flags flags,
        cl_GLenum target,
        cl_GLint  miplevel,
        cl_GLuint texobj,
        cl_int * err = nullptr)
    {
        cl_int error;
        object_ = ::clCreateFromGLTexture3D(
            context(),
            flags,
            target,
            miplevel,
            texobj,
            &error);

        detail::errHandler(error, __CREATE_GL_TEXTURE_3D_ERR);
        if (err != nullptr) {
            *err = error;
        }
    }

    //! \brief Default constructor - initializes to nullptr.
    Image3DGL() : Image3D() { }

    /*! \brief Constructor from cl_mem - takes ownership.
     *
     * \param retainObject will cause the constructor to retain its cl object.
     *                     Defaults to false to maintain compatibility with
     *                     earlier versions.
     *  See Memory for further details.
     */
    explicit Image3DGL(const cl_mem& image, bool retainObject = false) : 
        Image3D(image, retainObject) { }

    /*! \brief Assignment from cl_mem - performs shallow copy.
     *
     *  See Memory for further details.
     */
    Image3DGL& operator = (const cl_mem& rhs)
    {
        Image3D::operator=(rhs);
        return *this;
    }

};
#endif // CL_USE_DEPRECATED_OPENCL_1_1_APIS

#if CL_HPP_TARGET_OPENCL_VERSION >= 120
/*! \class ImageGL
 * \brief general image interface for GL interop.
 * We abstract the 2D and 3D GL images into a single instance here
 * that wraps all GL sourced images on the grounds that setup information
 * was performed by OpenCL anyway.
 */
class ImageGL : public Image
{
public:
    ImageGL(
        const Context& context,
        cl_mem_flags flags,
        cl_GLenum target,
        cl_GLint  miplevel,
        cl_GLuint texobj,
        cl_int * err = nullptr)
    {
        cl_int error;
        object_ = ::clCreateFromGLTexture(
            context(), 
            flags, 
            target,
            miplevel,
            texobj,
            &error);

        detail::errHandler(error, __CREATE_GL_TEXTURE_ERR);
        if (err != nullptr) {
            *err = error;
        }
    }

    ImageGL() : Image() { }
    
    /*! \brief Constructor from cl_mem - takes ownership.
     *
     * \param retainObject will cause the constructor to retain its cl object.
     *                     Defaults to false to maintain compatibility with
     *                     earlier versions.
     *  See Memory for further details.
     */
    explicit ImageGL(const cl_mem& image, bool retainObject = false) : 
        Image(image, retainObject) { }

    ImageGL& operator = (const cl_mem& rhs)
    {
        Image::operator=(rhs);
        return *this;
    }

};
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120



#if CL_HPP_TARGET_OPENCL_VERSION >= 200
/*! \brief Class interface for Pipe Memory Objects.
*
*  See Memory for details about copy semantics, etc.
*
*  \see Memory
*/
class Pipe : public Memory
{
public:

    /*! \brief Constructs a Pipe in a specified context.
     *
     * Wraps clCreatePipe().
     * @param context Context in which to create the pipe.
     * @param flags Bitfield. Only CL_MEM_READ_WRITE and CL_MEM_HOST_NO_ACCESS are valid.
     * @param packet_size Size in bytes of a single packet of the pipe.
     * @param max_packets Number of packets that may be stored in the pipe.
     *
     */
    Pipe(
        const Context& context,
        cl_uint packet_size,
        cl_uint max_packets,
        cl_int* err = nullptr)
    {
        cl_int error;

        cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS;
        object_ = ::clCreatePipe(context(), flags, packet_size, max_packets, nullptr, &error);

        detail::errHandler(error, __CREATE_PIPE_ERR);
        if (err != nullptr) {
            *err = error;
        }
    }

    /*! \brief Constructs a Pipe in a the default context.
     *
     * Wraps clCreatePipe().
     * @param flags Bitfield. Only CL_MEM_READ_WRITE and CL_MEM_HOST_NO_ACCESS are valid.
     * @param packet_size Size in bytes of a single packet of the pipe.
     * @param max_packets Number of packets that may be stored in the pipe.
     *
     */
    Pipe(
        cl_uint packet_size,
        cl_uint max_packets,
        cl_int* err = nullptr)
    {
        cl_int error;

        Context context = Context::getDefault(err);

        cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS;
        object_ = ::clCreatePipe(context(), flags, packet_size, max_packets, nullptr, &error);

        detail::errHandler(error, __CREATE_PIPE_ERR);
        if (err != nullptr) {
            *err = error;
        }
    }

    //! \brief Default constructor - initializes to nullptr.
    Pipe() : Memory() { }

    /*! \brief Constructor from cl_mem - takes ownership.
     *
     * \param retainObject will cause the constructor to retain its cl object.
     *                     Defaults to false to maintain compatibility with earlier versions.
     *
     *  See Memory for further details.
     */
    explicit Pipe(const cl_mem& pipe, bool retainObject = false) :
        Memory(pipe, retainObject) { }

    /*! \brief Assignment from cl_mem - performs shallow copy.
     *
     *  See Memory for further details.
     */
    Pipe& operator = (const cl_mem& rhs)
    {
        Memory::operator=(rhs);
        return *this;
    }



    //! \brief Wrapper for clGetMemObjectInfo().
    template <typename T>
    cl_int getInfo(cl_pipe_info name, T* param) const
    {
        return detail::errHandler(
            detail::getInfo(&::clGetPipeInfo, object_, name, param),
            __GET_PIPE_INFO_ERR);
    }

    //! \brief Wrapper for clGetMemObjectInfo() that returns by value.
    template <cl_pipe_info name> typename
        detail::param_traits<detail::cl_pipe_info, name>::param_type
        getInfo(cl_int* err = nullptr) const
    {
        typename detail::param_traits<
            detail::cl_pipe_info, name>::param_type param;
        cl_int result = getInfo(name, &param);
        if (err != nullptr) {
            *err = result;
        }
        return param;
    }
}; // class Pipe
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200


/*! \brief Class interface for cl_sampler.
 *
 *  \note Copies of these objects are shallow, meaning that the copy will refer
 *        to the same underlying cl_sampler as the original.  For details, see
 *        clRetainSampler() and clReleaseSampler().
 *
 *  \see cl_sampler 
 */
class Sampler : public detail::Wrapper<cl_sampler>
{
public:
    //! \brief Default constructor - initializes to nullptr.
    Sampler() { }

    /*! \brief Constructs a Sampler in a specified context.
     *
     *  Wraps clCreateSampler().
     */
    Sampler(
        const Context& context,
        cl_bool normalized_coords,
        cl_addressing_mode addressing_mode,
        cl_filter_mode filter_mode,
        cl_int* err = nullptr)
    {
        cl_int error;

#if CL_HPP_TARGET_OPENCL_VERSION >= 200
        cl_sampler_properties sampler_properties[] = {
            CL_SAMPLER_NORMALIZED_COORDS, normalized_coords,
            CL_SAMPLER_ADDRESSING_MODE, addressing_mode,
            CL_SAMPLER_FILTER_MODE, filter_mode,
            0 };
        object_ = ::clCreateSamplerWithProperties(
            context(),
            sampler_properties,
            &error);

        detail::errHandler(error, __CREATE_SAMPLER_WITH_PROPERTIES_ERR);
        if (err != nullptr) {
            *err = error;
        }
#else
        object_ = ::clCreateSampler(
            context(),
            normalized_coords,
            addressing_mode,
            filter_mode,
            &error);

        detail::errHandler(error, __CREATE_SAMPLER_ERR);
        if (err != nullptr) {
            *err = error;
        }
#endif        
    }

    /*! \brief Constructor from cl_sampler - takes ownership.
     * 
     * \param retainObject will cause the constructor to retain its cl object.
     *                     Defaults to false to maintain compatibility with
     *                     earlier versions.
     *  This effectively transfers ownership of a refcount on the cl_sampler
     *  into the new Sampler object.
     */
    explicit Sampler(const cl_sampler& sampler, bool retainObject = false) : 
        detail::Wrapper<cl_type>(sampler, retainObject) { }

    /*! \brief Assignment operator from cl_sampler - takes ownership.
     *
     *  This effectively transfers ownership of a refcount on the rhs and calls
     *  clReleaseSampler() on the value previously held by this instance.
     */
    Sampler& operator = (const cl_sampler& rhs)
    {
        detail::Wrapper<cl_type>::operator=(rhs);
        return *this;
    }

  

    //! \brief Wrapper for clGetSamplerInfo().
    template <typename T>
    cl_int getInfo(cl_sampler_info name, T* param) const
    {
        return detail::errHandler(
            detail::getInfo(&::clGetSamplerInfo, object_, name, param),
            __GET_SAMPLER_INFO_ERR);
    }

    //! \brief Wrapper for clGetSamplerInfo() that returns by value.
    template <cl_sampler_info name> typename
    detail::param_traits<detail::cl_sampler_info, name>::param_type
    getInfo(cl_int* err = nullptr) const
    {
        typename detail::param_traits<
            detail::cl_sampler_info, name>::param_type param;
        cl_int result = getInfo(name, &param);
        if (err != nullptr) {
            *err = result;
        }
        return param;
    }
};

class Program;
class CommandQueue;
class DeviceCommandQueue;
class Kernel;

//! \brief Class interface for specifying NDRange values.
class NDRange
{
private:
    size_type sizes_[3];
    cl_uint dimensions_;

public:
    //! \brief Default constructor - resulting range has zero dimensions.
    NDRange()
        : dimensions_(0)
    {
        sizes_[0] = 0;
        sizes_[1] = 0;
        sizes_[2] = 0;
    }

    //! \brief Constructs one-dimensional range.
    NDRange(size_type size0)
        : dimensions_(1)
    {
        sizes_[0] = size0;
        sizes_[1] = 1;
        sizes_[2] = 1;
    }

    //! \brief Constructs two-dimensional range.
    NDRange(size_type size0, size_type size1)
        : dimensions_(2)
    {
        sizes_[0] = size0;
        sizes_[1] = size1;
        sizes_[2] = 1;
    }

    //! \brief Constructs three-dimensional range.
    NDRange(size_type size0, size_type size1, size_type size2)
        : dimensions_(3)
    {
        sizes_[0] = size0;
        sizes_[1] = size1;
        sizes_[2] = size2;
    }

    //! \brief Constructs one-dimensional range.
    NDRange(array<size_type, 1> a) : NDRange(a[0]){}

    //! \brief Constructs two-dimensional range.
    NDRange(array<size_type, 2> a) : NDRange(a[0], a[1]){}

    //! \brief Constructs three-dimensional range.
    NDRange(array<size_type, 3> a) : NDRange(a[0], a[1], a[2]){}

    /*! \brief Conversion operator to const size_type *.
     *  
     *  \returns a pointer to the size of the first dimension.
     */
    operator const size_type*() const { 
        return sizes_; 
    }

    //! \brief Queries the number of dimensions in the range.
    size_type dimensions() const 
    { 
        return dimensions_; 
    }

    //! \brief Returns the size of the object in bytes based on the
    // runtime number of dimensions
    size_type size() const
    {
        return dimensions_*sizeof(size_type);
    }

    size_type* get()
    {
        return sizes_;
    }
    
    const size_type* get() const
    {
        return sizes_;
    }
};

//! \brief A zero-dimensional range.
static const NDRange NullRange;

//! \brief Local address wrapper for use with Kernel::setArg
struct LocalSpaceArg
{
    size_type size_;
};

namespace detail {

template <typename T, class Enable = void>
struct KernelArgumentHandler;

// Enable for objects that are not subclasses of memory
// Pointers, constants etc
template <typename T>
struct KernelArgumentHandler<T, typename std::enable_if<!std::is_base_of<cl::Memory, T>::value>::type>
{
    static size_type size(const T&) { return sizeof(T); }
    static const T* ptr(const T& value) { return &value; }
};

// Enable for subclasses of memory where we want to get a reference to the cl_mem out
// and pass that in for safety
template <typename T>
struct KernelArgumentHandler<T, typename std::enable_if<std::is_base_of<cl::Memory, T>::value>::type>
{
    static size_type size(const T&) { return sizeof(cl_mem); }
    static const cl_mem* ptr(const T& value) { return &(value()); }
};

// Specialization for DeviceCommandQueue defined later

template <>
struct KernelArgumentHandler<LocalSpaceArg, void>
{
    static size_type size(const LocalSpaceArg& value) { return value.size_; }
    static const void* ptr(const LocalSpaceArg&) { return nullptr; }
};

} 
//! \endcond

/*! Local
 * \brief Helper function for generating LocalSpaceArg objects.
 */
inline LocalSpaceArg
Local(size_type size)
{
    LocalSpaceArg ret = { size };
    return ret;
}

/*! \brief Class interface for cl_kernel.
 *
 *  \note Copies of these objects are shallow, meaning that the copy will refer
 *        to the same underlying cl_kernel as the original.  For details, see
 *        clRetainKernel() and clReleaseKernel().
 *
 *  \see cl_kernel
 */
class Kernel : public detail::Wrapper<cl_kernel>
{
public:
    inline Kernel(const Program& program, const string& name, cl_int* err = nullptr);
    inline Kernel(const Program& program, const char* name, cl_int* err = nullptr);

    //! \brief Default constructor - initializes to nullptr.
    Kernel() { }

    /*! \brief Constructor from cl_kernel - takes ownership.
     * 
     * \param retainObject will cause the constructor to retain its cl object.
     *                     Defaults to false to maintain compatibility with
     *                     earlier versions.
     *  This effectively transfers ownership of a refcount on the cl_kernel
     *  into the new Kernel object.
     */
    explicit Kernel(const cl_kernel& kernel, bool retainObject = false) : 
        detail::Wrapper<cl_type>(kernel, retainObject) { }

    /*! \brief Assignment operator from cl_kernel - takes ownership.
     *
     *  This effectively transfers ownership of a refcount on the rhs and calls
     *  clReleaseKernel() on the value previously held by this instance.
     */
    Kernel& operator = (const cl_kernel& rhs)
    {
        detail::Wrapper<cl_type>::operator=(rhs);
        return *this;
    }




    template <typename T>
    cl_int getInfo(cl_kernel_info name, T* param) const
    {
        return detail::errHandler(
            detail::getInfo(&::clGetKernelInfo, object_, name, param),
            __GET_KERNEL_INFO_ERR);
    }

    template <cl_kernel_info name> typename
    detail::param_traits<detail::cl_kernel_info, name>::param_type
    getInfo(cl_int* err = nullptr) const
    {
        typename detail::param_traits<
            detail::cl_kernel_info, name>::param_type param;
        cl_int result = getInfo(name, &param);
        if (err != nullptr) {
            *err = result;
        }
        return param;
    }

#if CL_HPP_TARGET_OPENCL_VERSION >= 120
    template <typename T>
    cl_int getArgInfo(cl_uint argIndex, cl_kernel_arg_info name, T* param) const
    {
        return detail::errHandler(
            detail::getInfo(&::clGetKernelArgInfo, object_, argIndex, name, param),
            __GET_KERNEL_ARG_INFO_ERR);
    }

    template <cl_kernel_arg_info name> typename
    detail::param_traits<detail::cl_kernel_arg_info, name>::param_type
    getArgInfo(cl_uint argIndex, cl_int* err = nullptr) const
    {
        typename detail::param_traits<
            detail::cl_kernel_arg_info, name>::param_type param;
        cl_int result = getArgInfo(argIndex, name, &param);
        if (err != nullptr) {
            *err = result;
        }
        return param;
    }
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120

    template <typename T>
    cl_int getWorkGroupInfo(
        const Device& device, cl_kernel_work_group_info name, T* param) const
    {
        return detail::errHandler(
            detail::getInfo(
                &::clGetKernelWorkGroupInfo, object_, device(), name, param),
                __GET_KERNEL_WORK_GROUP_INFO_ERR);
    }

    template <cl_kernel_work_group_info name> typename
    detail::param_traits<detail::cl_kernel_work_group_info, name>::param_type
        getWorkGroupInfo(const Device& device, cl_int* err = nullptr) const
    {
        typename detail::param_traits<
        detail::cl_kernel_work_group_info, name>::param_type param;
        cl_int result = getWorkGroupInfo(device, name, &param);
        if (err != nullptr) {
            *err = result;
        }
        return param;
    }
    
#if defined(CL_HPP_USE_CL_SUB_GROUPS_KHR) || CL_HPP_TARGET_OPENCL_VERSION >= 210
    cl_int getSubGroupInfo(const cl::Device &dev, cl_kernel_sub_group_info name, const cl::NDRange &range, size_type* param) const
    {
#if CL_HPP_TARGET_OPENCL_VERSION >= 210

        return detail::errHandler(
            clGetKernelSubGroupInfo(object_, dev(), name, range.size(), range.get(), sizeof(size_type), param, nullptr),
            __GET_KERNEL_SUB_GROUP_INFO_ERR);

#else // #if CL_HPP_TARGET_OPENCL_VERSION >= 210

        typedef clGetKernelSubGroupInfoKHR_fn PFN_clGetKernelSubGroupInfoKHR;
        static PFN_clGetKernelSubGroupInfoKHR pfn_clGetKernelSubGroupInfoKHR = nullptr;
        CL_HPP_INIT_CL_EXT_FCN_PTR_(clGetKernelSubGroupInfoKHR);

        return detail::errHandler(
            pfn_clGetKernelSubGroupInfoKHR(object_, dev(), name, range.size(), range.get(), sizeof(size_type), param, nullptr),
            __GET_KERNEL_SUB_GROUP_INFO_ERR);

#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 210
    }

    template <cl_kernel_sub_group_info name>
        size_type getSubGroupInfo(const cl::Device &dev, const cl::NDRange &range, cl_int* err = nullptr) const
    {
        size_type param;
        cl_int result = getSubGroupInfo(dev, name, range, &param);
        if (err != nullptr) {
            *err = result;
        }
        return param;
    }
#endif // defined(CL_HPP_USE_CL_SUB_GROUPS_KHR) || CL_HPP_TARGET_OPENCL_VERSION >= 210

#if CL_HPP_TARGET_OPENCL_VERSION >= 200
    /*! \brief setArg overload taking a shared_ptr type
     */
    template<typename T, class D>
    cl_int setArg(cl_uint index, const cl::pointer<T, D> &argPtr)
    {
        return detail::errHandler(
            ::clSetKernelArgSVMPointer(object_, index, argPtr.get()),
            __SET_KERNEL_ARGS_ERR);
    }

    /*! \brief setArg overload taking a vector type.
     */
    template<typename T, class Alloc>
    cl_int setArg(cl_uint index, const cl::vector<T, Alloc> &argPtr)
    {
        return detail::errHandler(
            ::clSetKernelArgSVMPointer(object_, index, argPtr.data()),
            __SET_KERNEL_ARGS_ERR);
    }

    /*! \brief setArg overload taking a pointer type
     */
    template<typename T>
    typename std::enable_if<std::is_pointer<T>::value, cl_int>::type
        setArg(cl_uint index, const T argPtr)
    {
        return detail::errHandler(
            ::clSetKernelArgSVMPointer(object_, index, argPtr),
            __SET_KERNEL_ARGS_ERR);
    }
#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200

    /*! \brief setArg overload taking a POD type
     */
    template <typename T>
    typename std::enable_if<!std::is_pointer<T>::value, cl_int>::type
        setArg(cl_uint index, const T &value)
    {
        return detail::errHandler(
            ::clSetKernelArg(
                object_,
                index,
                detail::KernelArgumentHandler<T>::size(value),
                detail::KernelArgumentHandler<T>::ptr(value)),
            __SET_KERNEL_ARGS_ERR);
    }

    cl_int setArg(cl_uint index, size_type size, const void* argPtr)
    {
        return detail::errHandler(
            ::clSetKernelArg(object_, index, size, argPtr),
            __SET_KERNEL_ARGS_ERR);
    }

#if CL_HPP_TARGET_OPENCL_VERSION >= 200
    /*!
     * Specify a vector of SVM pointers that the kernel may access in 
     * addition to its arguments.
     */
    cl_int setSVMPointers(const vector<void*> &pointerList)
    {
        return detail::errHandler(
            ::clSetKernelExecInfo(
                object_,
                CL_KERNEL_EXEC_INFO_SVM_PTRS,
                sizeof(void*)*pointerList.size(),
                pointerList.data()));
    }

    /*!
     * Specify a std::array of SVM pointers that the kernel may access in
     * addition to its arguments.
     */
    template<int ArrayLength>
    cl_int setSVMPointers(const std::array<void*, ArrayLength> &pointerList)
    {
        return detail::errHandler(
            ::clSetKernelExecInfo(
                object_,
                CL_KERNEL_EXEC_INFO_SVM_PTRS,
                sizeof(void*)*pointerList.size(),
                pointerList.data()));
    }

    /*! \brief Enable fine-grained system SVM.
     *
     * \note It is only possible to enable fine-grained system SVM if all devices
     *       in the context associated with kernel support it.
     * 
     * \param svmEnabled True if fine-grained system SVM is requested. False otherwise.
     * \return CL_SUCCESS if the function was executed succesfully. CL_INVALID_OPERATION
     *         if no devices in the context support fine-grained system SVM.
     *
     * \see clSetKernelExecInfo
     */
    cl_int enableFineGrainedSystemSVM(bool svmEnabled)
    {
        cl_bool svmEnabled_ = svmEnabled ? CL_TRUE : CL_FALSE;
        return detail::errHandler(
            ::clSetKernelExecInfo(
                object_,
                CL_KERNEL_EXEC_INFO_SVM_FINE_GRAIN_SYSTEM,
                sizeof(cl_bool),
                &svmEnabled_
                )
            );
    }
    
    template<int index, int ArrayLength, class D, typename T0, typename T1, typename... Ts>
    void setSVMPointersHelper(std::array<void*, ArrayLength> &pointerList, const pointer<T0, D> &t0, const pointer<T1, D> &t1, Ts & ... ts)
    {
        pointerList[index] = static_cast<void*>(t0.get());
        setSVMPointersHelper<index + 1, ArrayLength>(pointerList, t1, ts...);
    }

    template<int index, int ArrayLength, typename T0, typename T1, typename... Ts>
    typename std::enable_if<std::is_pointer<T0>::value, void>::type
    setSVMPointersHelper(std::array<void*, ArrayLength> &pointerList, T0 t0, T1 t1, Ts... ts)
    {
        pointerList[index] = static_cast<void*>(t0);
        setSVMPointersHelper<index + 1, ArrayLength>(pointerList, t1, ts...);
    }

    template<int index, int ArrayLength, typename T0, class D>
    void setSVMPointersHelper(std::array<void*, ArrayLength> &pointerList, const pointer<T0, D> &t0)
    {
        pointerList[index] = static_cast<void*>(t0.get());
    }


    template<int index, int ArrayLength, typename T0>
    typename std::enable_if<std::is_pointer<T0>::value, void>::type
    setSVMPointersHelper(std::array<void*, ArrayLength> &pointerList, T0 t0)
    {
        pointerList[index] = static_cast<void*>(t0);
    }

    template<typename T0, typename... Ts>
    cl_int setSVMPointers(const T0 &t0, Ts & ... ts)
    {
        std::array<void*, 1 + sizeof...(Ts)> pointerList;

        setSVMPointersHelper<0, 1 + sizeof...(Ts)>(pointerList, t0, ts...);
        return detail::errHandler(
            ::clSetKernelExecInfo(
            object_,
            CL_KERNEL_EXEC_INFO_SVM_PTRS,
            sizeof(void*)*(1 + sizeof...(Ts)),
            pointerList.data()));
    }

    template<typename T>
    cl_int setExecInfo(cl_kernel_exec_info param_name, const T& val)
    {
        return detail::errHandler(
            ::clSetKernelExecInfo(
            object_,
            param_name,
            sizeof(T),
            &val));
    }

    template<cl_kernel_exec_info name>
    cl_int setExecInfo(typename detail::param_traits<detail::cl_kernel_exec_info, name>::param_type& val)
    {
        return setExecInfo(name, val);
    }
#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200

#if CL_HPP_TARGET_OPENCL_VERSION >= 210
    /**
     * Make a deep copy of the kernel object including its arguments.
     * @return A new kernel object with internal state entirely separate from that
     *         of the original but with any arguments set on the original intact.
     */
    Kernel clone()
    {
        cl_int error;
        Kernel retValue(clCloneKernel(this->get(), &error));

        detail::errHandler(error, __CLONE_KERNEL_ERR);
        return retValue;
    }
#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 210
};

/*! \class Program
 * \brief Program interface that implements cl_program.
 */
class Program : public detail::Wrapper<cl_program>
{
public:
#if !defined(CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY)
    typedef vector<vector<unsigned char>> Binaries;
    typedef vector<string> Sources;
#else // #if !defined(CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY)
    typedef vector<std::pair<const void*, size_type> > Binaries;
    typedef vector<std::pair<const char*, size_type> > Sources;
#endif // #if !defined(CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY)
    
    Program(
        const string& source,
        bool build = false,
        cl_int* err = nullptr)
    {
        cl_int error;

        const char * strings = source.c_str();
        const size_type length  = source.size();

        Context context = Context::getDefault(err);

        object_ = call_clCreateProgramWithSource(
            context(), (cl_uint)1, &strings, &length, &error);

        detail::errHandler(error, __CREATE_PROGRAM_WITH_SOURCE_ERR);

        if (error == CL_SUCCESS && build) {

            error = ::clBuildProgram(
                object_,
                0,
                nullptr,
#if !defined(CL_HPP_CL_1_2_DEFAULT_BUILD)
                "-cl-std=CL2.0",
#else
                "",
#endif // #if !defined(CL_HPP_CL_1_2_DEFAULT_BUILD)
                nullptr,
                nullptr);

            detail::buildErrHandler(error, __BUILD_PROGRAM_ERR, getBuildInfo<CL_PROGRAM_BUILD_LOG>());
        }

        if (err != nullptr) {
            *err = error;
        }
    }

    Program(
        const Context& context,
        const string& source,
        bool build = false,
        cl_int* err = nullptr)
    {
        cl_int error;

        const char * strings = source.c_str();
        const size_type length  = source.size();

        object_ = call_clCreateProgramWithSource(
            context(), (cl_uint)1, &strings, &length, &error);

        detail::errHandler(error, __CREATE_PROGRAM_WITH_SOURCE_ERR);

        if (error == CL_SUCCESS && build) {
            error = ::clBuildProgram(
                object_,
                0,
                nullptr,
#if !defined(CL_HPP_CL_1_2_DEFAULT_BUILD)
                "-cl-std=CL2.0",
#else
                "",
#endif // #if !defined(CL_HPP_CL_1_2_DEFAULT_BUILD)
                nullptr,
                nullptr);
            
            detail::buildErrHandler(error, __BUILD_PROGRAM_ERR, getBuildInfo<CL_PROGRAM_BUILD_LOG>());
        }

        if (err != nullptr) {
            *err = error;
        }
    }

    /**
     * Create a program from a vector of source strings and the default context.
     * Does not compile or link the program.
     */
    Program(
        const Sources& sources,
        cl_int* err = nullptr)
    {
        cl_int error;
        Context context = Context::getDefault(err);

        const size_type n = (size_type)sources.size();

        vector<size_type> lengths(n);
        vector<const char*> strings(n);

        for (size_type i = 0; i < n; ++i) {
#if !defined(CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY)
            strings[i] = sources[(int)i].data();
            lengths[i] = sources[(int)i].length();
#else // #if !defined(CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY)
            strings[i] = sources[(int)i].first;
            lengths[i] = sources[(int)i].second;
#endif // #if !defined(CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY)
        }

        object_ = call_clCreateProgramWithSource(
            context(), (cl_uint)n, strings.data(), lengths.data(), &error);

        detail::errHandler(error, __CREATE_PROGRAM_WITH_SOURCE_ERR);
        if (err != nullptr) {
            *err = error;
        }
    }

    /**
     * Create a program from a vector of source strings and a provided context.
     * Does not compile or link the program.
     */
    Program(
        const Context& context,
        const Sources& sources,
        cl_int* err = nullptr)
    {
        cl_int error;

        const size_type n = (size_type)sources.size();

        vector<size_type> lengths(n);
        vector<const char*> strings(n);

        for (size_type i = 0; i < n; ++i) {
#if !defined(CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY)
            strings[i] = sources[(int)i].data();
            lengths[i] = sources[(int)i].length();
#else // #if !defined(CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY)
            strings[i] = sources[(int)i].first;
            lengths[i] = sources[(int)i].second;
#endif // #if !defined(CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY)
        }

        object_ = call_clCreateProgramWithSource(
            context(), (cl_uint)n, strings.data(), lengths.data(), &error);

        detail::errHandler(error, __CREATE_PROGRAM_WITH_SOURCE_ERR);
        if (err != nullptr) {
            *err = error;
        }
    }

#if defined(CL_HPP_USE_IL_KHR) || CL_HPP_TARGET_OPENCL_VERSION >= 210
    /**
     * Program constructor to allow construction of program from SPIR-V or another IL.
     *
     * Requires OpenCL 2.1 or newer or the cl_khr_il_program extension.
     */
    Program(
        const vector<char>& IL,
        bool build = false,
        cl_int* err = nullptr)
    {
        cl_int error;

        Context context = Context::getDefault(err);

#if CL_HPP_TARGET_OPENCL_VERSION >= 210

        object_ = ::clCreateProgramWithIL(
            context(), static_cast<const void*>(IL.data()), IL.size(), &error);

#else // #if CL_HPP_TARGET_OPENCL_VERSION >= 210

        typedef clCreateProgramWithILKHR_fn PFN_clCreateProgramWithILKHR;
        static PFN_clCreateProgramWithILKHR pfn_clCreateProgramWithILKHR = nullptr;
        CL_HPP_INIT_CL_EXT_FCN_PTR_(clCreateProgramWithILKHR);

        object_ = pfn_clCreateProgramWithILKHR(
                context(), static_cast<const void*>(IL.data()), IL.size(), &error);

#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 210

        detail::errHandler(error, __CREATE_PROGRAM_WITH_IL_ERR);

        if (error == CL_SUCCESS && build) {

            error = ::clBuildProgram(
                object_,
                0,
                nullptr,
#if !defined(CL_HPP_CL_1_2_DEFAULT_BUILD)
                "-cl-std=CL2.0",
#else
                "",
#endif // #if !defined(CL_HPP_CL_1_2_DEFAULT_BUILD)
                nullptr,
                nullptr);

            detail::buildErrHandler(error, __BUILD_PROGRAM_ERR, getBuildInfo<CL_PROGRAM_BUILD_LOG>());
        }

        if (err != nullptr) {
            *err = error;
        }
    }

    /**
     * Program constructor to allow construction of program from SPIR-V or another IL
     * for a specific context.
     *
     * Requires OpenCL 2.1 or newer or the cl_khr_il_program extension.
     */
    Program(
        const Context& context,
        const vector<char>& IL,
        bool build = false,
        cl_int* err = nullptr)
    {
        cl_int error;

#if CL_HPP_TARGET_OPENCL_VERSION >= 210

        object_ = ::clCreateProgramWithIL(
            context(), static_cast<const void*>(IL.data()), IL.size(), &error);

#else // #if CL_HPP_TARGET_OPENCL_VERSION >= 210

        typedef clCreateProgramWithILKHR_fn PFN_clCreateProgramWithILKHR;
        static PFN_clCreateProgramWithILKHR pfn_clCreateProgramWithILKHR = nullptr;
        CL_HPP_INIT_CL_EXT_FCN_PTR_(clCreateProgramWithILKHR);

        object_ = pfn_clCreateProgramWithILKHR(
            context(), static_cast<const void*>(IL.data()), IL.size(), &error);

#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 210

        detail::errHandler(error, __CREATE_PROGRAM_WITH_IL_ERR);

        if (error == CL_SUCCESS && build) {
            error = ::clBuildProgram(
                object_,
                0,
                nullptr,
#if !defined(CL_HPP_CL_1_2_DEFAULT_BUILD)
                "-cl-std=CL2.0",
#else
                "",
#endif // #if !defined(CL_HPP_CL_1_2_DEFAULT_BUILD)
                nullptr,
                nullptr);

            detail::buildErrHandler(error, __BUILD_PROGRAM_ERR, getBuildInfo<CL_PROGRAM_BUILD_LOG>());
        }

        if (err != nullptr) {
            *err = error;
        }
    }
#endif // defined(CL_HPP_USE_IL_KHR) || CL_HPP_TARGET_OPENCL_VERSION >= 210

    /**
     * Construct a program object from a list of devices and a per-device list of binaries.
     * \param context A valid OpenCL context in which to construct the program.
     * \param devices A vector of OpenCL device objects for which the program will be created.
     * \param binaries A vector of pairs of a pointer to a binary object and its length.
     * \param binaryStatus An optional vector that on completion will be resized to
     *   match the size of binaries and filled with values to specify if each binary
     *   was successfully loaded.
     *   Set to CL_SUCCESS if the binary was successfully loaded.
     *   Set to CL_INVALID_VALUE if the length is 0 or the binary pointer is nullptr.
     *   Set to CL_INVALID_BINARY if the binary provided is not valid for the matching device.
     * \param err if non-nullptr will be set to CL_SUCCESS on successful operation or one of the following errors:
     *   CL_INVALID_CONTEXT if context is not a valid context.
     *   CL_INVALID_VALUE if the length of devices is zero; or if the length of binaries does not match the length of devices; 
     *     or if any entry in binaries is nullptr or has length 0.
     *   CL_INVALID_DEVICE if OpenCL devices listed in devices are not in the list of devices associated with context.
     *   CL_INVALID_BINARY if an invalid program binary was encountered for any device. binaryStatus will return specific status for each device.
     *   CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required by the OpenCL implementation on the host.
     */
    Program(
        const Context& context,
        const vector<Device>& devices,
        const Binaries& binaries,
        vector<cl_int>* binaryStatus = nullptr,
        cl_int* err = nullptr)
    {
        cl_int error;
        
        const size_type numDevices = devices.size();
        
        // Catch size mismatch early and return
        if(binaries.size() != numDevices) {
            error = CL_INVALID_VALUE;
            detail::errHandler(error, __CREATE_PROGRAM_WITH_BINARY_ERR);
            if (err != nullptr) {
                *err = error;
            }
            return;
        }

        vector<size_type> lengths(numDevices);
        vector<const unsigned char*> images(numDevices);
#if !defined(CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY)
        for (size_type i = 0; i < numDevices; ++i) {
            images[i] = binaries[i].data();
            lengths[i] = binaries[(int)i].size();
        }
#else // #if !defined(CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY)
        for (size_type i = 0; i < numDevices; ++i) {
            images[i] = (const unsigned char*)binaries[i].first;
            lengths[i] = binaries[(int)i].second;
        }
#endif // #if !defined(CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY)

        vector<cl_device_id> deviceIDs(numDevices);
        for( size_type deviceIndex = 0; deviceIndex < numDevices; ++deviceIndex ) {
            deviceIDs[deviceIndex] = (devices[deviceIndex])();
        }

        if(binaryStatus) {
            binaryStatus->resize(numDevices);
        }

        object_ = call_clCreateProgramWithBinary(
            context(), (cl_uint) devices.size(),
            deviceIDs.data(),
            lengths.data(), images.data(), (binaryStatus != nullptr && numDevices > 0)
               ? &binaryStatus->front()
               : nullptr, &error);

        detail::errHandler(error, __CREATE_PROGRAM_WITH_BINARY_ERR);
        if (err != nullptr) {
            *err = error;
        }
    }

    
#if CL_HPP_TARGET_OPENCL_VERSION >= 120
    /**
     * Create program using builtin kernels.
     * \param kernelNames Semi-colon separated list of builtin kernel names
     */
    Program(
        const Context& context,
        const vector<Device>& devices,
        const string& kernelNames,
        cl_int* err = nullptr)
    {
        cl_int error;


        size_type numDevices = devices.size();
        vector<cl_device_id> deviceIDs(numDevices);
        for( size_type deviceIndex = 0; deviceIndex < numDevices; ++deviceIndex ) {
            deviceIDs[deviceIndex] = (devices[deviceIndex])();
        }
        
        object_ = ::clCreateProgramWithBuiltInKernels(
            context(), 
            (cl_uint) devices.size(),
            deviceIDs.data(),
            kernelNames.c_str(), 
            &error);

        detail::errHandler(error, __CREATE_PROGRAM_WITH_BUILT_IN_KERNELS_ERR);
        if (err != nullptr) {
            *err = error;
        }
    }
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120

    Program() { }
    

    /*! \brief Constructor from cl_program - takes ownership.
     *
     * \param retainObject will cause the constructor to retain its cl object.
     *                     Defaults to false to maintain compatibility with
     *                     earlier versions.
     */
    explicit Program(const cl_program& program, bool retainObject = false) : 
        detail::Wrapper<cl_type>(program, retainObject) { }

    Program& operator = (const cl_program& rhs)
    {
        detail::Wrapper<cl_type>::operator=(rhs);
        return *this;
    }

    cl_int build(
        const vector<Device>& devices,
        const string& options,
        void (CL_CALLBACK * notifyFptr)(cl_program, void *) = nullptr,
        void* data = nullptr) const
    {
        return build(devices, options.c_str(), notifyFptr, data);
    }

    cl_int build(
        const vector<Device>& devices,
        const char* options = nullptr,
        void (CL_CALLBACK * notifyFptr)(cl_program, void *) = nullptr,
        void* data = nullptr) const
    {
        size_type numDevices = devices.size();
        vector<cl_device_id> deviceIDs(numDevices);

        for( size_type deviceIndex = 0; deviceIndex < numDevices; ++deviceIndex ) {
            deviceIDs[deviceIndex] = (devices[deviceIndex])();
        }

        cl_int buildError = ::clBuildProgram(
            object_,
            (cl_uint)
            devices.size(),
            deviceIDs.data(),
            options,
            notifyFptr,
            data);

        return detail::buildErrHandler(buildError, __BUILD_PROGRAM_ERR, getBuildInfo<CL_PROGRAM_BUILD_LOG>());
    }

    cl_int build(
        const Device& device,
        const string& options,
        void (CL_CALLBACK * notifyFptr)(cl_program, void *) = nullptr,
        void* data = nullptr) const
    {
        return build(device, options.c_str(), notifyFptr, data);
    }

    cl_int build(
        const Device& device,
        const char* options = nullptr,
        void (CL_CALLBACK * notifyFptr)(cl_program, void *) = nullptr,
        void* data = nullptr) const
    {
        cl_device_id deviceID = device();

        cl_int buildError = ::clBuildProgram(
            object_,
            1,
            &deviceID,
            options,
            notifyFptr,
            data);

        BuildLogType buildLog(0);
        buildLog.push_back(std::make_pair(device, getBuildInfo<CL_PROGRAM_BUILD_LOG>(device)));
        return detail::buildErrHandler(buildError, __BUILD_PROGRAM_ERR, buildLog);
    }

    cl_int build(
        const string& options,
        void (CL_CALLBACK * notifyFptr)(cl_program, void *) = nullptr,
        void* data = nullptr) const
    {
        return build(options.c_str(), notifyFptr, data);
    }

    cl_int build(
        const char* options = nullptr,
        void (CL_CALLBACK * notifyFptr)(cl_program, void *) = nullptr,
        void* data = nullptr) const
    {
        cl_int buildError = ::clBuildProgram(
            object_,
            0,
            nullptr,
            options,
            notifyFptr,
            data);

        return detail::buildErrHandler(buildError, __BUILD_PROGRAM_ERR, getBuildInfo<CL_PROGRAM_BUILD_LOG>());
    }

#if CL_HPP_TARGET_OPENCL_VERSION >= 120
    cl_int compile(
        const string& options,
        void (CL_CALLBACK * notifyFptr)(cl_program, void *) = nullptr,
        void* data = nullptr) const
    {
        return compile(options.c_str(), notifyFptr, data);
    }

    cl_int compile(
        const char* options = nullptr,
        void (CL_CALLBACK * notifyFptr)(cl_program, void *) = nullptr,
        void* data = nullptr) const
    {
        cl_int error = ::clCompileProgram(
            object_,
            0,
            nullptr,
            options,
            0,
            nullptr,
            nullptr,
            notifyFptr,
            data);
        return detail::buildErrHandler(error, __COMPILE_PROGRAM_ERR, getBuildInfo<CL_PROGRAM_BUILD_LOG>());
    }

    cl_int compile(
        const string& options,
        const vector<Program>& inputHeaders,
        const vector<string>& headerIncludeNames,
        void (CL_CALLBACK * notifyFptr)(cl_program, void *) = nullptr,
        void* data = nullptr) const
    {
        return compile(options.c_str(), inputHeaders, headerIncludeNames, notifyFptr, data);
    }

    cl_int compile(
        const char* options,
        const vector<Program>& inputHeaders,
        const vector<string>& headerIncludeNames,
        void (CL_CALLBACK * notifyFptr)(cl_program, void *) = nullptr,
        void* data = nullptr) const
    {
        static_assert(sizeof(cl::Program) == sizeof(cl_program),
            "Size of cl::Program must be equal to size of cl_program");
        vector<const char*> headerIncludeNamesCStr;
        for(const string& name: headerIncludeNames) {
            headerIncludeNamesCStr.push_back(name.c_str());
        }
        cl_int error = ::clCompileProgram(
            object_,
            0,
            nullptr,
            options,
            static_cast<cl_uint>(inputHeaders.size()),
            reinterpret_cast<const cl_program*>(inputHeaders.data()),
            reinterpret_cast<const char**>(headerIncludeNamesCStr.data()),
            notifyFptr,
            data);
        return detail::buildErrHandler(error, __COMPILE_PROGRAM_ERR, getBuildInfo<CL_PROGRAM_BUILD_LOG>());
    }

    cl_int compile(
        const string& options,
        const vector<Device>& deviceList,
        const vector<Program>& inputHeaders = vector<Program>(),
        const vector<string>& headerIncludeNames = vector<string>(),
        void (CL_CALLBACK * notifyFptr)(cl_program, void *) = nullptr,
        void* data = nullptr) const
    {
        return compile(options.c_str(), deviceList, inputHeaders, headerIncludeNames, notifyFptr, data);
    }

    cl_int compile(
        const char* options,
        const vector<Device>& deviceList,
        const vector<Program>& inputHeaders = vector<Program>(),
        const vector<string>& headerIncludeNames = vector<string>(),
        void (CL_CALLBACK * notifyFptr)(cl_program, void *) = nullptr,
        void* data = nullptr) const
    {
        static_assert(sizeof(cl::Program) == sizeof(cl_program),
            "Size of cl::Program must be equal to size of cl_program");
        vector<const char*> headerIncludeNamesCStr;
        for(const string& name: headerIncludeNames) {
            headerIncludeNamesCStr.push_back(name.c_str());
        }
        vector<cl_device_id> deviceIDList;
        for(const Device& device: deviceList) {
            deviceIDList.push_back(device());
        }
        cl_int error = ::clCompileProgram(
            object_,
            static_cast<cl_uint>(deviceList.size()),
            reinterpret_cast<const cl_device_id*>(deviceIDList.data()),
            options,
            static_cast<cl_uint>(inputHeaders.size()),
            reinterpret_cast<const cl_program*>(inputHeaders.data()),
            reinterpret_cast<const char**>(headerIncludeNamesCStr.data()),
            notifyFptr,
            data);
        return detail::buildErrHandler(error, __COMPILE_PROGRAM_ERR, getBuildInfo<CL_PROGRAM_BUILD_LOG>());
    }
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120

    template <typename T>
    cl_int getInfo(cl_program_info name, T* param) const
    {
        return detail::errHandler(
            detail::getInfo(&::clGetProgramInfo, object_, name, param),
            __GET_PROGRAM_INFO_ERR);
    }

    template <cl_program_info name> typename
    detail::param_traits<detail::cl_program_info, name>::param_type
    getInfo(cl_int* err = nullptr) const
    {
        typename detail::param_traits<
            detail::cl_program_info, name>::param_type param;
        cl_int result = getInfo(name, &param);
        if (err != nullptr) {
            *err = result;
        }
        return param;
    }

    template <typename T>
    cl_int getBuildInfo(
        const Device& device, cl_program_build_info name, T* param) const
    {
        return detail::errHandler(
            detail::getInfo(
                &::clGetProgramBuildInfo, object_, device(), name, param),
                __GET_PROGRAM_BUILD_INFO_ERR);
    }

    template <cl_program_build_info name> typename
    detail::param_traits<detail::cl_program_build_info, name>::param_type
    getBuildInfo(const Device& device, cl_int* err = nullptr) const
    {
        typename detail::param_traits<
            detail::cl_program_build_info, name>::param_type param;
        cl_int result = getBuildInfo(device, name, &param);
        if (err != nullptr) {
            *err = result;
        }
        return param;
    }
    
    /**
     * Build info function that returns a vector of device/info pairs for the specified 
     * info type and for all devices in the program.
     * On an error reading the info for any device, an empty vector of info will be returned.
     */
    template <cl_program_build_info name>
    vector<std::pair<cl::Device, typename detail::param_traits<detail::cl_program_build_info, name>::param_type>>
        getBuildInfo(cl_int *err = nullptr) const
    {
        cl_int result = CL_SUCCESS;

        auto devs = getInfo<CL_PROGRAM_DEVICES>(&result);
        vector<std::pair<cl::Device, typename detail::param_traits<detail::cl_program_build_info, name>::param_type>>
            devInfo;

        // If there was an initial error from getInfo return the error
        if (result != CL_SUCCESS) {
            if (err != nullptr) {
                *err = result;
            }
            return devInfo;
        }

        for (const cl::Device &d : devs) {
            typename detail::param_traits<
                detail::cl_program_build_info, name>::param_type param;
            result = getBuildInfo(d, name, &param);
            devInfo.push_back(
                std::pair<cl::Device, typename detail::param_traits<detail::cl_program_build_info, name>::param_type>
                (d, param));
            if (result != CL_SUCCESS) {
                // On error, leave the loop and return the error code
                break;
            }
        }
        if (err != nullptr) {
            *err = result;
        }
        if (result != CL_SUCCESS) {
            devInfo.clear();
        }
        return devInfo;
    }

    cl_int createKernels(vector<Kernel>* kernels)
    {
        cl_uint numKernels;
        cl_int err = ::clCreateKernelsInProgram(object_, 0, nullptr, &numKernels);
        if (err != CL_SUCCESS) {
            return detail::errHandler(err, __CREATE_KERNELS_IN_PROGRAM_ERR);
        }

        vector<cl_kernel> value(numKernels);
        
        err = ::clCreateKernelsInProgram(
            object_, numKernels, value.data(), nullptr);
        if (err != CL_SUCCESS) {
            return detail::errHandler(err, __CREATE_KERNELS_IN_PROGRAM_ERR);
        }

        if (kernels) {
            kernels->resize(value.size());

            // Assign to param, constructing with retain behaviour
            // to correctly capture each underlying CL object
            for (size_type i = 0; i < value.size(); i++) {
                // We do not need to retain because this kernel is being created 
                // by the runtime
                (*kernels)[i] = Kernel(value[i], false);
            }
        }
        return CL_SUCCESS;
    }

#if CL_HPP_TARGET_OPENCL_VERSION >= 220
#if defined(CL_USE_DEPRECATED_OPENCL_2_2_APIS)
    /*! \brief Registers a callback function to be called when destructors for
     *         program scope global variables are complete and before the
     *         program is released.
     *
     *  Wraps clSetProgramReleaseCallback().
     *
     *  Each call to this function registers the specified user callback function
     *  on a callback stack associated with program. The registered user callback
     *  functions are called in the reverse order in which they were registered.
     */
    CL_API_PREFIX__VERSION_2_2_DEPRECATED cl_int setReleaseCallback(
        void (CL_CALLBACK * pfn_notify)(cl_program program, void * user_data),
        void * user_data = nullptr) CL_API_SUFFIX__VERSION_2_2_DEPRECATED
    {
        return detail::errHandler(
            ::clSetProgramReleaseCallback(
                object_,
                pfn_notify,
                user_data),
            __SET_PROGRAM_RELEASE_CALLBACK_ERR);
    }
#endif // #if defined(CL_USE_DEPRECATED_OPENCL_2_2_APIS)

    /*! \brief Sets a SPIR-V specialization constant.
     *
     *  Wraps clSetProgramSpecializationConstant().
     */
    template <typename T>
    typename std::enable_if<!std::is_pointer<T>::value, cl_int>::type
        setSpecializationConstant(cl_uint index, const T &value)
    {
        return detail::errHandler(
            ::clSetProgramSpecializationConstant(
                object_,
                index,
                sizeof(value),
                &value),
            __SET_PROGRAM_SPECIALIZATION_CONSTANT_ERR);
    }

    /*! \brief Sets a SPIR-V specialization constant.
     *
     *  Wraps clSetProgramSpecializationConstant().
     */
    cl_int setSpecializationConstant(cl_uint index, size_type size, const void* value)
    {
        return detail::errHandler(
            ::clSetProgramSpecializationConstant(
                object_,
                index,
                size,
                value),
            __SET_PROGRAM_SPECIALIZATION_CONSTANT_ERR);
    }
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 220
};

#if CL_HPP_TARGET_OPENCL_VERSION >= 120
inline Program linkProgram(
    const Program& input1,
    const Program& input2,
    const char* options = nullptr,
    void (CL_CALLBACK * notifyFptr)(cl_program, void *) = nullptr,
    void* data = nullptr,
    cl_int* err = nullptr)
{
    cl_int error_local = CL_SUCCESS;
    cl_program programs[2] = { input1(), input2() };

    Context ctx = input1.getInfo<CL_PROGRAM_CONTEXT>(&error_local);
    if(error_local!=CL_SUCCESS) {
        detail::errHandler(error_local, __LINK_PROGRAM_ERR);
    }

    cl_program prog = ::clLinkProgram(
        ctx(),
        0,
        nullptr,
        options,
        2,
        programs,
        notifyFptr,
        data,
        &error_local);

    detail::errHandler(error_local,__COMPILE_PROGRAM_ERR);
    if (err != nullptr) {
        *err = error_local;
    }

    return Program(prog);
}

inline Program linkProgram(
    const Program& input1,
    const Program& input2,
    const string& options,
    void (CL_CALLBACK * notifyFptr)(cl_program, void *) = nullptr,
    void* data = nullptr,
    cl_int* err = nullptr)
{
    return linkProgram(input1, input2, options.c_str(), notifyFptr, data, err);
}

inline Program linkProgram(
    const vector<Program>& inputPrograms,
    const char* options = nullptr,
    void (CL_CALLBACK * notifyFptr)(cl_program, void *) = nullptr,
    void* data = nullptr,
    cl_int* err = nullptr)
{
    cl_int error_local = CL_SUCCESS;
    Context ctx;

    static_assert(sizeof(cl::Program) == sizeof(cl_program),
        "Size of cl::Program must be equal to size of cl_program");

    if(inputPrograms.size() > 0) {
        ctx = inputPrograms[0].getInfo<CL_PROGRAM_CONTEXT>(&error_local);
        if(error_local!=CL_SUCCESS) {
            detail::errHandler(error_local, __LINK_PROGRAM_ERR);
        }
    }

    cl_program prog = ::clLinkProgram(
        ctx(),
        0,
        nullptr,
        options,
        static_cast<cl_uint>(inputPrograms.size()),
        reinterpret_cast<const cl_program *>(inputPrograms.data()),
        notifyFptr,
        data,
        &error_local);

    detail::errHandler(error_local,__COMPILE_PROGRAM_ERR);
    if (err != nullptr) {
        *err = error_local;
    }

    return Program(prog);
}

inline Program linkProgram(
    const vector<Program>& inputPrograms,
    const string& options,
    void (CL_CALLBACK * notifyFptr)(cl_program, void *) = nullptr,
    void* data = nullptr,
    cl_int* err = nullptr)
{
    return linkProgram(inputPrograms, options.c_str(), notifyFptr, data, err);
}
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120

// Template specialization for CL_PROGRAM_BINARIES
template <>
inline cl_int cl::Program::getInfo(cl_program_info name, vector<vector<unsigned char>>* param) const
{
    if (name != CL_PROGRAM_BINARIES) {
        return CL_INVALID_VALUE;
    }
    if (param) {
        // Resize the parameter array appropriately for each allocation
        // and pass down to the helper

        vector<size_type> sizes = getInfo<CL_PROGRAM_BINARY_SIZES>();
        size_type numBinaries = sizes.size();

        // Resize the parameter array and constituent arrays
        param->resize(numBinaries);
        for (size_type i = 0; i < numBinaries; ++i) {
            (*param)[i].resize(sizes[i]);
        }

        return detail::errHandler(
            detail::getInfo(&::clGetProgramInfo, object_, name, param),
            __GET_PROGRAM_INFO_ERR);
    }

    return CL_SUCCESS;
}

template<>
inline vector<vector<unsigned char>> cl::Program::getInfo<CL_PROGRAM_BINARIES>(cl_int* err) const
{
    vector<vector<unsigned char>> binariesVectors;

    cl_int result = getInfo(CL_PROGRAM_BINARIES, &binariesVectors);
    if (err != nullptr) {
        *err = result;
    }
    return binariesVectors;
}

#if CL_HPP_TARGET_OPENCL_VERSION >= 220
// Template specialization for clSetProgramSpecializationConstant
template <>
inline cl_int cl::Program::setSpecializationConstant(cl_uint index, const bool &value)
{
    cl_uchar ucValue = value ? CL_UCHAR_MAX : 0;
    return detail::errHandler(
        ::clSetProgramSpecializationConstant(
            object_,
            index,
            sizeof(ucValue),
            &ucValue),
        __SET_PROGRAM_SPECIALIZATION_CONSTANT_ERR);
}
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 220

inline Kernel::Kernel(const Program& program, const string& name, cl_int* err)
{
    cl_int error;

    object_ = ::clCreateKernel(program(), name.c_str(), &error);
    detail::errHandler(error, __CREATE_KERNEL_ERR);

    if (err != nullptr) {
        *err = error;
    }
}

inline Kernel::Kernel(const Program& program, const char* name, cl_int* err)
{
    cl_int error;

    object_ = ::clCreateKernel(program(), name, &error);
    detail::errHandler(error, __CREATE_KERNEL_ERR);

    if (err != nullptr) {
        *err = error;
    }
}

#ifdef cl_khr_external_memory
enum class ExternalMemoryType : cl_external_memory_handle_type_khr
{
    None = 0,
#ifdef cl_khr_external_memory_opaque_fd
    OpaqueFd = CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR,
#endif // cl_khr_external_memory_opaque_fd
#ifdef cl_khr_external_memory_win32
    OpaqueWin32 = CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KHR,
    OpaqueWin32Kmt = CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KMT_KHR,
#endif // cl_khr_external_memory_win32
#ifdef cl_khr_external_memory_dma_buf
    DmaBuf = CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR,
#endif // cl_khr_external_memory_dma_buf
};
#endif // cl_khr_external_memory

enum class QueueProperties : cl_command_queue_properties
{
    None = 0,
    Profiling = CL_QUEUE_PROFILING_ENABLE,
    OutOfOrder = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
};

inline QueueProperties operator|(QueueProperties lhs, QueueProperties rhs)
{
    return static_cast<QueueProperties>(static_cast<cl_command_queue_properties>(lhs) | static_cast<cl_command_queue_properties>(rhs));
}

inline QueueProperties operator&(QueueProperties lhs, QueueProperties rhs)
{
    return static_cast<QueueProperties>(static_cast<cl_command_queue_properties>(lhs) & static_cast<cl_command_queue_properties>(rhs));
}

/*! \class CommandQueue
 * \brief CommandQueue interface for cl_command_queue.
 */
class CommandQueue : public detail::Wrapper<cl_command_queue>
{
private:
    static std::once_flag default_initialized_;
    static CommandQueue default_;
    static cl_int default_error_;

    /*! \brief Create the default command queue returned by @ref getDefault.
     *
     * It sets default_error_ to indicate success or failure. It does not throw
     * @c cl::Error.
     */
    static void makeDefault()
    {
        /* We don't want to throw an error from this function, so we have to
         * catch and set the error flag.
         */
#if defined(CL_HPP_ENABLE_EXCEPTIONS)
        try
#endif
        {
            int error;
            Context context = Context::getDefault(&error);

            if (error != CL_SUCCESS) {
                default_error_ = error;
            }
            else {
                Device device = Device::getDefault();
                default_ = CommandQueue(context, device, 0, &default_error_);
            }
        }
#if defined(CL_HPP_ENABLE_EXCEPTIONS)
        catch (cl::Error &e) {
            default_error_ = e.err();
        }
#endif
    }

    /*! \brief Create the default command queue.
     *
     * This sets @c default_. It does not throw
     * @c cl::Error.
     */
    static void makeDefaultProvided(const CommandQueue &c) {
        default_ = c;
    }

#ifdef cl_khr_external_memory
    static std::once_flag ext_memory_initialized_;

    static void initMemoryExtension(const cl::Device& device) 
    {
        auto platform = device.getInfo<CL_DEVICE_PLATFORM>()();

        CL_HPP_INIT_CL_EXT_FCN_PTR_PLATFORM_(platform, clEnqueueAcquireExternalMemObjectsKHR);
        CL_HPP_INIT_CL_EXT_FCN_PTR_PLATFORM_(platform, clEnqueueReleaseExternalMemObjectsKHR);

        if ((pfn_clEnqueueAcquireExternalMemObjectsKHR == nullptr)
            && (pfn_clEnqueueReleaseExternalMemObjectsKHR == nullptr))
        {
            detail::errHandler(CL_INVALID_VALUE, __ENQUEUE_ACQUIRE_EXTERNAL_MEMORY_ERR);
        }
    }
#endif // cl_khr_external_memory

public:
#ifdef CL_HPP_UNIT_TEST_ENABLE
    /*! \brief Reset the default.
    *
    * This sets @c default_ to an empty value to support cleanup in
    * the unit test framework.
    * This function is not thread safe.
    */
    static void unitTestClearDefault() {
        default_ = CommandQueue();
    }
#endif // #ifdef CL_HPP_UNIT_TEST_ENABLE
        

    /*!
     * \brief Constructs a CommandQueue based on passed properties.
     * Will return an CL_INVALID_QUEUE_PROPERTIES error if CL_QUEUE_ON_DEVICE is specified.
     */
   CommandQueue(
        cl_command_queue_properties properties,
        cl_int* err = nullptr)
    {
        cl_int error;

        Context context = Context::getDefault(&error);
        detail::errHandler(error, __CREATE_CONTEXT_ERR);

        if (error != CL_SUCCESS) {
            if (err != nullptr) {
                *err = error;
            }
        }
        else {
            Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
            bool useWithProperties;

#if CL_HPP_TARGET_OPENCL_VERSION >= 200 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
            // Run-time decision based on the actual platform
            {
                cl_uint version = detail::getContextPlatformVersion(context());
                useWithProperties = (version >= 0x20000); // OpenCL 2.0 or above
            }
#elif CL_HPP_TARGET_OPENCL_VERSION >= 200
            useWithProperties = true;
#else
            useWithProperties = false;
#endif

#if CL_HPP_TARGET_OPENCL_VERSION >= 200
            if (useWithProperties) {
                cl_queue_properties queue_properties[] = {
                    CL_QUEUE_PROPERTIES, properties, 0 };
                if ((properties & CL_QUEUE_ON_DEVICE) == 0) {
                    object_ = call_clCreateCommandQueueWithProperties(
                        context(), device(), queue_properties, &error);
                }
                else {
                    error = CL_INVALID_QUEUE_PROPERTIES;
                }

                detail::errHandler(error, __CREATE_COMMAND_QUEUE_WITH_PROPERTIES_ERR);
                if (err != nullptr) {
                    *err = error;
                }
            }
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
#if CL_HPP_MINIMUM_OPENCL_VERSION < 200
            if (!useWithProperties) {
                object_ = ::clCreateCommandQueue(
                    context(), device(), properties, &error);

                detail::errHandler(error, __CREATE_COMMAND_QUEUE_ERR);
                if (err != nullptr) {
                    *err = error;
                }
            }
#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 200
        }
    }

   /*!
    * \brief Constructs a CommandQueue based on passed properties.
    * Will return an CL_INVALID_QUEUE_PROPERTIES error if CL_QUEUE_ON_DEVICE is specified.
    */
   CommandQueue(
       QueueProperties properties,
       cl_int* err = nullptr)
   {
       cl_int error;

       Context context = Context::getDefault(&error);
       detail::errHandler(error, __CREATE_CONTEXT_ERR);

       if (error != CL_SUCCESS) {
           if (err != nullptr) {
               *err = error;
           }
       }
       else {
           Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
           bool useWithProperties;

#if CL_HPP_TARGET_OPENCL_VERSION >= 200 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
           // Run-time decision based on the actual platform
           {
               cl_uint version = detail::getContextPlatformVersion(context());
               useWithProperties = (version >= 0x20000); // OpenCL 2.0 or above
           }
#elif CL_HPP_TARGET_OPENCL_VERSION >= 200
           useWithProperties = true;
#else
           useWithProperties = false;
#endif

#if CL_HPP_TARGET_OPENCL_VERSION >= 200
           if (useWithProperties) {
               cl_queue_properties queue_properties[] = {
                   CL_QUEUE_PROPERTIES, static_cast<cl_queue_properties>(properties), 0 };

               object_ = call_clCreateCommandQueueWithProperties(
                   context(), device(), queue_properties, &error);

               detail::errHandler(error, __CREATE_COMMAND_QUEUE_WITH_PROPERTIES_ERR);
               if (err != nullptr) {
                   *err = error;
               }
           }
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
#if CL_HPP_MINIMUM_OPENCL_VERSION < 200
           if (!useWithProperties) {
               object_ = ::clCreateCommandQueue(
                   context(), device(), static_cast<cl_command_queue_properties>(properties), &error);

               detail::errHandler(error, __CREATE_COMMAND_QUEUE_ERR);
               if (err != nullptr) {
                   *err = error;
               }
           }
#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 200

       }
   }

    /*!
     * \brief Constructs a CommandQueue for an implementation defined device in the given context
     * Will return an CL_INVALID_QUEUE_PROPERTIES error if CL_QUEUE_ON_DEVICE is specified.
     */
    explicit CommandQueue(
        const Context& context,
        cl_command_queue_properties properties = 0,
        cl_int* err = nullptr)
    {
        cl_int error;
        bool useWithProperties;
        vector<cl::Device> devices;
        error = context.getInfo(CL_CONTEXT_DEVICES, &devices);

        detail::errHandler(error, __CREATE_CONTEXT_ERR);

        if (error != CL_SUCCESS)
        {
            if (err != nullptr) {
                *err = error;
            }
            return;
        }

#if CL_HPP_TARGET_OPENCL_VERSION >= 200 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
        // Run-time decision based on the actual platform
        {
            cl_uint version = detail::getContextPlatformVersion(context());
            useWithProperties = (version >= 0x20000); // OpenCL 2.0 or above
        }
#elif CL_HPP_TARGET_OPENCL_VERSION >= 200
        useWithProperties = true;
#else
        useWithProperties = false;
#endif

#if CL_HPP_TARGET_OPENCL_VERSION >= 200
        if (useWithProperties) {
            cl_queue_properties queue_properties[] = {
                CL_QUEUE_PROPERTIES, properties, 0 };
            if ((properties & CL_QUEUE_ON_DEVICE) == 0) {
                object_ = call_clCreateCommandQueueWithProperties(
                    context(), devices[0](), queue_properties, &error);
            }
            else {
                error = CL_INVALID_QUEUE_PROPERTIES;
            }

            detail::errHandler(error, __CREATE_COMMAND_QUEUE_WITH_PROPERTIES_ERR);
            if (err != nullptr) {
                *err = error;
            }
        }
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
#if CL_HPP_MINIMUM_OPENCL_VERSION < 200
        if (!useWithProperties) {
            object_ = ::clCreateCommandQueue(
                context(), devices[0](), properties, &error);

            detail::errHandler(error, __CREATE_COMMAND_QUEUE_ERR);
            if (err != nullptr) {
                *err = error;
            }
        }
#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 200
    }

    /*!
    * \brief Constructs a CommandQueue for an implementation defined device in the given context
    * Will return an CL_INVALID_QUEUE_PROPERTIES error if CL_QUEUE_ON_DEVICE is specified.
    */
    explicit CommandQueue(
        const Context& context,
        QueueProperties properties,
        cl_int* err = nullptr)
    {
        cl_int error;
        bool useWithProperties;
        vector<cl::Device> devices;
        error = context.getInfo(CL_CONTEXT_DEVICES, &devices);

        detail::errHandler(error, __CREATE_CONTEXT_ERR);

        if (error != CL_SUCCESS)
        {
            if (err != nullptr) {
                *err = error;
            }
            return;
        }

#if CL_HPP_TARGET_OPENCL_VERSION >= 200 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
        // Run-time decision based on the actual platform
        {
            cl_uint version = detail::getContextPlatformVersion(context());
            useWithProperties = (version >= 0x20000); // OpenCL 2.0 or above
        }
#elif CL_HPP_TARGET_OPENCL_VERSION >= 200
        useWithProperties = true;
#else
        useWithProperties = false;
#endif

#if CL_HPP_TARGET_OPENCL_VERSION >= 200
        if (useWithProperties) {
            cl_queue_properties queue_properties[] = {
                CL_QUEUE_PROPERTIES, static_cast<cl_queue_properties>(properties), 0 };
            object_ = call_clCreateCommandQueueWithProperties(
                context(), devices[0](), queue_properties, &error);

            detail::errHandler(error, __CREATE_COMMAND_QUEUE_WITH_PROPERTIES_ERR);
            if (err != nullptr) {
                *err = error;
            }
        }
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
#if CL_HPP_MINIMUM_OPENCL_VERSION < 200
        if (!useWithProperties) {
            object_ = ::clCreateCommandQueue(
                context(), devices[0](), static_cast<cl_command_queue_properties>(properties), &error);

            detail::errHandler(error, __CREATE_COMMAND_QUEUE_ERR);
            if (err != nullptr) {
                *err = error;
            }
        }
#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 200
    }

    /*!
     * \brief Constructs a CommandQueue for a passed device and context
     * Will return an CL_INVALID_QUEUE_PROPERTIES error if CL_QUEUE_ON_DEVICE is specified.
     */
    CommandQueue(
        const Context& context,
        const Device& device,
        cl_command_queue_properties properties = 0,
        cl_int* err = nullptr)
    {
        cl_int error;
        bool useWithProperties;

#if CL_HPP_TARGET_OPENCL_VERSION >= 200 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
        // Run-time decision based on the actual platform
        {
            cl_uint version = detail::getContextPlatformVersion(context());
            useWithProperties = (version >= 0x20000); // OpenCL 2.0 or above
        }
#elif CL_HPP_TARGET_OPENCL_VERSION >= 200
        useWithProperties = true;
#else
        useWithProperties = false;
#endif

#if CL_HPP_TARGET_OPENCL_VERSION >= 200
        if (useWithProperties) {
            cl_queue_properties queue_properties[] = {
                CL_QUEUE_PROPERTIES, properties, 0 };
            object_ = call_clCreateCommandQueueWithProperties(
                context(), device(), queue_properties, &error);

            detail::errHandler(error, __CREATE_COMMAND_QUEUE_WITH_PROPERTIES_ERR);
            if (err != nullptr) {
                *err = error;
            }
        }
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
#if CL_HPP_MINIMUM_OPENCL_VERSION < 200
        if (!useWithProperties) {
            object_ = ::clCreateCommandQueue(
                context(), device(), properties, &error);

            detail::errHandler(error, __CREATE_COMMAND_QUEUE_ERR);
            if (err != nullptr) {
                *err = error;
            }
        }
#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 200
    }

    /*!
     * \brief Constructs a CommandQueue for a passed device and context
     * Will return an CL_INVALID_QUEUE_PROPERTIES error if CL_QUEUE_ON_DEVICE is specified.
     */
    CommandQueue(
        const Context& context,
        const Device& device,
        QueueProperties properties,
        cl_int* err = nullptr)
    {
        cl_int error;
        bool useWithProperties;

#if CL_HPP_TARGET_OPENCL_VERSION >= 200 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
        // Run-time decision based on the actual platform
        {
            cl_uint version = detail::getContextPlatformVersion(context());
            useWithProperties = (version >= 0x20000); // OpenCL 2.0 or above
        }
#elif CL_HPP_TARGET_OPENCL_VERSION >= 200
        useWithProperties = true;
#else
        useWithProperties = false;
#endif

#if CL_HPP_TARGET_OPENCL_VERSION >= 200
        if (useWithProperties) {
            cl_queue_properties queue_properties[] = {
                CL_QUEUE_PROPERTIES, static_cast<cl_queue_properties>(properties), 0 };
            object_ = call_clCreateCommandQueueWithProperties(
                context(), device(), queue_properties, &error);

            detail::errHandler(error, __CREATE_COMMAND_QUEUE_WITH_PROPERTIES_ERR);
            if (err != nullptr) {
                *err = error;
            }
        }
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
#if CL_HPP_MINIMUM_OPENCL_VERSION < 200
        if (!useWithProperties) {
            object_ = ::clCreateCommandQueue(
                context(), device(), static_cast<cl_command_queue_properties>(properties), &error);

            detail::errHandler(error, __CREATE_COMMAND_QUEUE_ERR);
            if (err != nullptr) {
                *err = error;
            }
        }
#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 200
    }

    static CommandQueue getDefault(cl_int * err = nullptr) 
    {
        std::call_once(default_initialized_, makeDefault);
#if CL_HPP_TARGET_OPENCL_VERSION >= 200
        detail::errHandler(default_error_, __CREATE_COMMAND_QUEUE_WITH_PROPERTIES_ERR);
#else // CL_HPP_TARGET_OPENCL_VERSION >= 200
        detail::errHandler(default_error_, __CREATE_COMMAND_QUEUE_ERR);
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
        if (err != nullptr) {
            *err = default_error_;
        }
        return default_;
    }

    /**
     * Modify the default command queue to be used by
     * subsequent operations.
     * Will only set the default if no default was previously created.
     * @return updated default command queue.
     *         Should be compared to the passed value to ensure that it was updated.
     */
    static CommandQueue setDefault(const CommandQueue &default_queue)
    {
        std::call_once(default_initialized_, makeDefaultProvided, std::cref(default_queue));
        detail::errHandler(default_error_);
        return default_;
    }

    CommandQueue() { }


    /*! \brief Constructor from cl_command_queue - takes ownership.
     *
     * \param retainObject will cause the constructor to retain its cl object.
     *                     Defaults to false to maintain compatibility with
     *                     earlier versions.
     */
    explicit CommandQueue(const cl_command_queue& commandQueue, bool retainObject = false) : 
        detail::Wrapper<cl_type>(commandQueue, retainObject) { }

    CommandQueue& operator = (const cl_command_queue& rhs)
    {
        detail::Wrapper<cl_type>::operator=(rhs);
        return *this;
    }

    template <typename T>
    cl_int getInfo(cl_command_queue_info name, T* param) const
    {
        return detail::errHandler(
            detail::getInfo(
                &::clGetCommandQueueInfo, object_, name, param),
                __GET_COMMAND_QUEUE_INFO_ERR);
    }

    template <cl_command_queue_info name> typename
    detail::param_traits<detail::cl_command_queue_info, name>::param_type
    getInfo(cl_int* err = nullptr) const
    {
        typename detail::param_traits<
            detail::cl_command_queue_info, name>::param_type param;
        cl_int result = getInfo(name, &param);
        if (err != nullptr) {
            *err = result;
        }
        return param;
    }

    cl_int enqueueReadBuffer(
        const Buffer& buffer,
        cl_bool blocking,
        size_type offset,
        size_type size,
        void* ptr,
        const vector<Event>* events = nullptr,
        Event* event = nullptr) const
    {
        cl_event tmp;
        cl_int err = detail::errHandler(
            call_clEnqueueReadBuffer(
                object_, buffer(), blocking, offset, size,
                ptr,
                (events != nullptr) ? (cl_uint) events->size() : 0,
                (events != nullptr && events->size() > 0) ? (cl_event*) &events->front() : nullptr,
                (event != nullptr) ? &tmp : nullptr),
            __ENQUEUE_READ_BUFFER_ERR);

        if (event != nullptr && err == CL_SUCCESS)
            *event = tmp;

        return err;
    }

    cl_int enqueueWriteBuffer(
        const Buffer& buffer,
        cl_bool blocking,
        size_type offset,
        size_type size,
        const void* ptr,
        const vector<Event>* events = nullptr,
        Event* event = nullptr) const
    {
        cl_event tmp;
        cl_int err = detail::errHandler(
            call_clEnqueueWriteBuffer(
                object_, buffer(), blocking, offset, size,
                ptr,
                (events != nullptr) ? (cl_uint) events->size() : 0,
                (events != nullptr && events->size() > 0) ? (cl_event*) &events->front() : nullptr,
                (event != nullptr) ? &tmp : nullptr),
                __ENQUEUE_WRITE_BUFFER_ERR);

        if (event != nullptr && err == CL_SUCCESS)
            *event = tmp;

        return err;
    }

    cl_int enqueueCopyBuffer(
        const Buffer& src,
        const Buffer& dst,
        size_type src_offset,
        size_type dst_offset,
        size_type size,
        const vector<Event>* events = nullptr,
        Event* event = nullptr) const
    {
        cl_event tmp;
        cl_int err = detail::errHandler(
            call_clEnqueueCopyBuffer(
                object_, src(), dst(), src_offset, dst_offset, size,
                (events != nullptr) ? (cl_uint) events->size() : 0,
                (events != nullptr && events->size() > 0) ? (cl_event*) &events->front() : nullptr,
                (event != nullptr) ? &tmp : nullptr),
            __ENQEUE_COPY_BUFFER_ERR);

        if (event != nullptr && err == CL_SUCCESS)
            *event = tmp;

        return err;
    }
#if CL_HPP_TARGET_OPENCL_VERSION >= 110
    cl_int enqueueReadBufferRect(
        const Buffer& buffer,
        cl_bool blocking,
        const array<size_type, 3>& buffer_offset,
        const array<size_type, 3>& host_offset,
        const array<size_type, 3>& region,
        size_type buffer_row_pitch,
        size_type buffer_slice_pitch,
        size_type host_row_pitch,
        size_type host_slice_pitch,
        void *ptr,
        const vector<Event>* events = nullptr,
        Event* event = nullptr) const
    {
        cl_event tmp;
        cl_int err = detail::errHandler(
            ::clEnqueueReadBufferRect(
                object_, 
                buffer(), 
                blocking,
                buffer_offset.data(),
                host_offset.data(),
                region.data(),
                buffer_row_pitch,
                buffer_slice_pitch,
                host_row_pitch,
                host_slice_pitch,
                ptr,
                (events != nullptr) ? (cl_uint) events->size() : 0,
                (events != nullptr && events->size() > 0) ? (cl_event*) &events->front() : nullptr,
                (event != nullptr) ? &tmp : nullptr),
                __ENQUEUE_READ_BUFFER_RECT_ERR);

        if (event != nullptr && err == CL_SUCCESS)
            *event = tmp;

        return err;
    }

    cl_int enqueueReadBufferRect(
        const Buffer& buffer,
        cl_bool blocking,
        const array<size_type, 2>& buffer_offset,
        const array<size_type, 2>& host_offset,
        const array<size_type, 2>& region,
        size_type buffer_row_pitch,
        size_type buffer_slice_pitch,
        size_type host_row_pitch,
        size_type host_slice_pitch,
        void* ptr,
        const vector<Event>* events = nullptr,
        Event* event = nullptr) const
    { 
        return enqueueReadBufferRect(
            buffer,
            blocking,
            { buffer_offset[0], buffer_offset[1], 0 },
            { host_offset[0], host_offset[1], 0 },
            { region[0], region[1], 1 },
            buffer_row_pitch,
            buffer_slice_pitch,
            host_row_pitch,
            host_slice_pitch,
            ptr,
            events,
            event);
    }

    cl_int enqueueWriteBufferRect(
        const Buffer& buffer,
        cl_bool blocking,
        const array<size_type, 3>& buffer_offset,
        const array<size_type, 3>& host_offset,
        const array<size_type, 3>& region,
        size_type buffer_row_pitch,
        size_type buffer_slice_pitch,
        size_type host_row_pitch,
        size_type host_slice_pitch,
        const void *ptr,
        const vector<Event>* events = nullptr,
        Event* event = nullptr) const
    {
        cl_event tmp;
        cl_int err = detail::errHandler(
            ::clEnqueueWriteBufferRect(
                object_, 
                buffer(), 
                blocking,
                buffer_offset.data(),
                host_offset.data(),
                region.data(),
                buffer_row_pitch,
                buffer_slice_pitch,
                host_row_pitch,
                host_slice_pitch,
                ptr,
                (events != nullptr) ? (cl_uint) events->size() : 0,
                (events != nullptr && events->size() > 0) ? (cl_event*) &events->front() : nullptr,
                (event != nullptr) ? &tmp : nullptr),
                __ENQUEUE_WRITE_BUFFER_RECT_ERR);

        if (event != nullptr && err == CL_SUCCESS)
            *event = tmp;

        return err;
    }

    cl_int enqueueWriteBufferRect(
        const Buffer& buffer,
        cl_bool blocking,
        const array<size_type, 2>& buffer_offset,
        const array<size_type, 2>& host_offset,
        const array<size_type, 2>& region,
        size_type buffer_row_pitch,
        size_type buffer_slice_pitch,
        size_type host_row_pitch,
        size_type host_slice_pitch,
        const void* ptr,
        const vector<Event>* events = nullptr,
        Event* event = nullptr) const
    {
        return enqueueWriteBufferRect(
            buffer, 
            blocking,
            { buffer_offset[0], buffer_offset[1], 0 },
            { host_offset[0], host_offset[1], 0 },
            { region[0], region[1], 1 },
            buffer_row_pitch,
            buffer_slice_pitch,
            host_row_pitch,
            host_slice_pitch,
            ptr,
            events,
            event);
    }

    cl_int enqueueCopyBufferRect(
        const Buffer& src,
        const Buffer& dst,
        const array<size_type, 3>& src_origin,
        const array<size_type, 3>& dst_origin,
        const array<size_type, 3>& region,
        size_type src_row_pitch,
        size_type src_slice_pitch,
        size_type dst_row_pitch,
        size_type dst_slice_pitch,
        const vector<Event>* events = nullptr,
        Event* event = nullptr) const
    {
        cl_event tmp;
        cl_int err = detail::errHandler(
            ::clEnqueueCopyBufferRect(
                object_, 
                src(), 
                dst(), 
                src_origin.data(),
                dst_origin.data(),
                region.data(),
                src_row_pitch,
                src_slice_pitch,
                dst_row_pitch,
                dst_slice_pitch,
                (events != nullptr) ? (cl_uint) events->size() : 0,
                (events != nullptr && events->size() > 0) ? (cl_event*) &events->front() : nullptr,
                (event != nullptr) ? &tmp : nullptr),
            __ENQEUE_COPY_BUFFER_RECT_ERR);

        if (event != nullptr && err == CL_SUCCESS)
            *event = tmp;

        return err;
    }

    cl_int enqueueCopyBufferRect(
        const Buffer& src,
        const Buffer& dst,
        const array<size_type, 2>& src_origin,
        const array<size_type, 2>& dst_origin,
        const array<size_type, 2>& region,
        size_type src_row_pitch,
        size_type src_slice_pitch,
        size_type dst_row_pitch,
        size_type dst_slice_pitch,
        const vector<Event>* events = nullptr,
        Event* event = nullptr) const
    {
        return enqueueCopyBufferRect(
            src,
            dst,
            { src_origin[0], src_origin[1], 0 },
            { dst_origin[0], dst_origin[1], 0 },
            { region[0], region[1], 1 },
            src_row_pitch,
            src_slice_pitch,
            dst_row_pitch,
            dst_slice_pitch,
            events,
            event);
    }

#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
#if CL_HPP_TARGET_OPENCL_VERSION >= 120
    /**
     * Enqueue a command to fill a buffer object with a pattern
     * of a given size. The pattern is specified as a vector type.
     * \tparam PatternType The datatype of the pattern field. 
     *     The pattern type must be an accepted OpenCL data type.
     * \tparam offset Is the offset in bytes into the buffer at 
     *     which to start filling. This must be a multiple of 
     *     the pattern size.
     * \tparam size Is the size in bytes of the region to fill.
     *     This must be a multiple of the pattern size.
     */
    template<typename PatternType>
    cl_int enqueueFillBuffer(
        const Buffer& buffer,
        PatternType pattern,
        size_type offset,
        size_type size,
        const vector<Event>* events = nullptr,
        Event* event = nullptr) const
    {
        cl_event tmp;
        cl_int err = detail::errHandler(
            call_clEnqueueFillBuffer(
                object_, 
                buffer(),
                static_cast<void*>(&pattern),
                sizeof(PatternType), 
                offset, 
                size,
                (events != nullptr) ? (cl_uint) events->size() : 0,
                (events != nullptr && events->size() > 0) ? (cl_event*) &events->front() : nullptr,
                (event != nullptr) ? &tmp : nullptr),
                __ENQUEUE_FILL_BUFFER_ERR);

        if (event != nullptr && err == CL_SUCCESS)
            *event = tmp;

        return err;
    }
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120

    cl_int enqueueReadImage(
        const Image& image,
        cl_bool blocking,
        const array<size_type, 3>& origin,
        const array<size_type, 3>& region,
        size_type row_pitch,
        size_type slice_pitch,
        void* ptr,
        const vector<Event>* events = nullptr,
        Event* event = nullptr) const
    {
        cl_event tmp;
        cl_int err = detail::errHandler(
            ::clEnqueueReadImage(
                object_, 
                image(), 
                blocking, 
                origin.data(),
                region.data(), 
                row_pitch, 
                slice_pitch, 
                ptr,
                (events != nullptr) ? (cl_uint) events->size() : 0,
                (events != nullptr && events->size() > 0) ? (cl_event*) &events->front() : nullptr,
                (event != nullptr) ? &tmp : nullptr),
            __ENQUEUE_READ_IMAGE_ERR);

        if (event != nullptr && err == CL_SUCCESS)
            *event = tmp;

        return err;
    }

    cl_int enqueueReadImage(
        const Image& image,
        cl_bool blocking,
        const array<size_type, 2>& origin,
        const array<size_type, 2>& region,
        size_type row_pitch,
        size_type slice_pitch,
        void* ptr,
        const vector<Event>* events = nullptr,
        Event* event = nullptr) const
    {
        return enqueueReadImage(
            image,
            blocking,
            { origin[0], origin[1], 0 },
            { region[0], region[1], 1 },
            row_pitch,
            slice_pitch,
            ptr,
            events,
            event);
    }

    cl_int enqueueWriteImage(
        const Image& image,
        cl_bool blocking,
        const array<size_type, 3>& origin,
        const array<size_type, 3>& region,
        size_type row_pitch,
        size_type slice_pitch,
        const void* ptr,
        const vector<Event>* events = nullptr,
        Event* event = nullptr) const
    {
        cl_event tmp;
        cl_int err = detail::errHandler(
            call_clEnqueueWriteImage(
                object_, 
                image(), 
                blocking, 
                origin.data(),
                region.data(), 
                row_pitch, 
                slice_pitch, 
                ptr,
                (events != nullptr) ? (cl_uint) events->size() : 0,
                (events != nullptr && events->size() > 0) ? (cl_event*) &events->front() : nullptr,
                (event != nullptr) ? &tmp : nullptr),
            __ENQUEUE_WRITE_IMAGE_ERR);

        if (event != nullptr && err == CL_SUCCESS)
            *event = tmp;

        return err;
    }

    cl_int enqueueWriteImage(
        const Image& image,
        cl_bool blocking,
        const array<size_type, 2>& origin,
        const array<size_type, 2>& region,
        size_type row_pitch,
        size_type slice_pitch,
        const void* ptr,
        const vector<Event>* events = nullptr,
        Event* event = nullptr) const
    {
        return enqueueWriteImage(
            image,
            blocking,
            { origin[0], origin[1], 0 },
            { region[0], region[1], 1 },
            row_pitch,
            slice_pitch,
            ptr,
            events,
            event);
    }

    cl_int enqueueCopyImage(
        const Image& src,
        const Image& dst,
        const array<size_type, 3>& src_origin,
        const array<size_type, 3>& dst_origin,
        const array<size_type, 3>& region,
        const vector<Event>* events = nullptr,
        Event* event = nullptr) const
    {
        cl_event tmp;
        cl_int err = detail::errHandler(
            call_clEnqueueCopyImage(
                object_, 
                src(), 
                dst(), 
                src_origin.data(),
                dst_origin.data(), 
                region.data(),
                (events != nullptr) ? (cl_uint) events->size() : 0,
                (events != nullptr && events->size() > 0) ? (cl_event*) &events->front() : nullptr,
                (event != nullptr) ? &tmp : nullptr),
            __ENQUEUE_COPY_IMAGE_ERR);

        if (event != nullptr && err == CL_SUCCESS)
            *event = tmp;

        return err;
    }

    cl_int enqueueCopyImage(
        const Image& src,
        const Image& dst,
        const array<size_type, 2>& src_origin,
        const array<size_type, 2>& dst_origin,
        const array<size_type, 2>& region,
        const vector<Event>* events = nullptr,
        Event* event = nullptr) const
    {
        return enqueueCopyImage(
            src,
            dst,
            { src_origin[0], src_origin[1], 0 },
            { dst_origin[0], dst_origin[1], 0 },
            { region[0], region[1], 1 },
            events,
            event);
    }

#if CL_HPP_TARGET_OPENCL_VERSION >= 120
    /**
     * Enqueue a command to fill an image object with a specified color.
     * \param fillColor is the color to use to fill the image.
     *     This is a four component RGBA floating-point, signed integer
     *     or unsigned integer color value if  the image channel data
     *     type is an unnormalized signed integer type.   
     */
    template <typename T>
    typename std::enable_if<std::is_same<T, cl_float4>::value ||
                            std::is_same<T, cl_int4  >::value ||
                            std::is_same<T, cl_uint4 >::value,
                            cl_int>::type 
     enqueueFillImage(
         const Image& image, 
         T fillColor,
         const array<size_type, 3>& origin,
         const array<size_type, 3>& region,
         const vector<Event>* events = nullptr,
         Event* event = nullptr) const
    {
        cl_event tmp;
        cl_int err = detail::errHandler(
            call_clEnqueueFillImage(
                object_,
                image(),
                static_cast<void*>(&fillColor),
                origin.data(),
                region.data(),
                (events != nullptr) ? (cl_uint)events->size() : 0,
                (events != nullptr && events->size() > 0) ? (cl_event*)&events->front() : NULL,
                (event != NULL) ? &tmp : nullptr),
            __ENQUEUE_FILL_IMAGE_ERR);

        if (event != nullptr && err == CL_SUCCESS) *event = tmp;

        return err;
    }

   /**
     * Enqueue a command to fill an image object with a specified color.
     * \param fillColor is the color to use to fill the image.
     *     This is a four component RGBA floating-point, signed integer
     *     or unsigned integer color value if  the image channel data
     *     type is an unnormalized signed integer type.
     */
    template <typename T>
    typename std::enable_if<std::is_same<T, cl_float4>::value ||
                            std::is_same<T, cl_int4  >::value ||
                            std::is_same<T, cl_uint4 >::value, cl_int>::type
    enqueueFillImage(
        const Image& image,
        T fillColor,
        const array<size_type, 2>& origin,
        const array<size_type, 2>& region,
        const vector<Event>* events = nullptr,
        Event* event = nullptr) const
    {
        return enqueueFillImage(
            image,
            fillColor,
            { origin[0], origin[1], 0 },
            { region[0], region[1], 1 },
            events,
            event
            );
    }
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120

    cl_int enqueueCopyImageToBuffer(
        const Image& src,
        const Buffer& dst,
        const array<size_type, 3>& src_origin,
        const array<size_type, 3>& region,
        size_type dst_offset,
        const vector<Event>* events = nullptr,
        Event* event = nullptr) const
    {
        cl_event tmp;
        cl_int err = detail::errHandler(
            ::clEnqueueCopyImageToBuffer(
                object_, 
                src(), 
                dst(), 
                src_origin.data(),
                region.data(), 
                dst_offset,
                (events != nullptr) ? (cl_uint) events->size() : 0,
                (events != nullptr && events->size() > 0) ? (cl_event*) &events->front() : nullptr,
                (event != nullptr) ? &tmp : nullptr),
            __ENQUEUE_COPY_IMAGE_TO_BUFFER_ERR);

        if (event != nullptr && err == CL_SUCCESS)
            *event = tmp;

        return err;
    }

    cl_int enqueueCopyImageToBuffer(
        const Image& src,
        const Buffer& dst,
        const array<size_type, 2>& src_origin,
        const array<size_type, 2>& region,
        size_type dst_offset,
        const vector<Event>* events = nullptr,
        Event* event = nullptr) const
    { 
        return enqueueCopyImageToBuffer(
            src,
            dst,
            { src_origin[0], src_origin[1], 0 },
            { region[0], region[1], 1 },
            dst_offset,
            events,
            event);
    }

    cl_int enqueueCopyBufferToImage(
        const Buffer& src,
        const Image& dst,
        size_type src_offset,
        const array<size_type, 3>& dst_origin,
        const array<size_type, 3>& region,
        const vector<Event>* events = nullptr,
        Event* event = nullptr) const
    {
        cl_event tmp;
        cl_int err = detail::errHandler(
            ::clEnqueueCopyBufferToImage(
                object_, 
                src(), 
                dst(), 
                src_offset,
                dst_origin.data(), 
                region.data(),
                (events != nullptr) ? (cl_uint) events->size() : 0,
                (events != nullptr && events->size() > 0) ? (cl_event*) &events->front() : nullptr,
                (event != nullptr) ? &tmp : nullptr),
            __ENQUEUE_COPY_BUFFER_TO_IMAGE_ERR);

        if (event != nullptr && err == CL_SUCCESS)
            *event = tmp;

        return err;
    }

    cl_int enqueueCopyBufferToImage(
        const Buffer& src,
        const Image& dst,
        size_type src_offset,
        const array<size_type, 2>& dst_origin,
        const array<size_type, 2>& region,
        const vector<Event>* events = nullptr,
        Event* event = nullptr) const
    {
        return enqueueCopyBufferToImage(
            src,
            dst, 
            src_offset,
            { dst_origin[0], dst_origin[1], 0 },
            { region[0], region[1], 1 },
            events,
            event);
    }

    void* enqueueMapBuffer(
        const Buffer& buffer,
        cl_bool blocking,
        cl_map_flags flags,
        size_type offset,
        size_type size,
        const vector<Event>* events = nullptr,
        Event* event = nullptr,
        cl_int* err = nullptr) const
    {
        cl_event tmp;
        cl_int error;
        void * result = ::clEnqueueMapBuffer(
            object_, buffer(), blocking, flags, offset, size,
            (events != nullptr) ? (cl_uint) events->size() : 0,
            (events != nullptr && events->size() > 0) ? (cl_event*) &events->front() : nullptr,
            (event != nullptr) ? &tmp : nullptr,
            &error);

        detail::errHandler(error, __ENQUEUE_MAP_BUFFER_ERR);
        if (err != nullptr) {
            *err = error;
        }
        if (event != nullptr && error == CL_SUCCESS)
            *event = tmp;

        return result;
    }

    void* enqueueMapImage(
        const Image& image,
        cl_bool blocking,
        cl_map_flags flags,
        const array<size_type, 3>& origin,
        const array<size_type, 3>& region,
        size_type * row_pitch,
        size_type * slice_pitch,
        const vector<Event>* events = nullptr,
        Event* event = nullptr,
        cl_int* err = nullptr) const
    {
        cl_event tmp;
        cl_int error;
        void * result = call_clEnqueueMapImage(
            object_, image(), blocking, flags,
            origin.data(), 
            region.data(),
            row_pitch, slice_pitch,
            (events != nullptr) ? (cl_uint) events->size() : 0,
            (events != nullptr && events->size() > 0) ? (cl_event*) &events->front() : nullptr,
            (event != nullptr) ? &tmp : nullptr,
            &error);

        detail::errHandler(error, __ENQUEUE_MAP_IMAGE_ERR);
        if (err != nullptr) {
              *err = error;
        }
        if (event != nullptr && error == CL_SUCCESS)
            *event = tmp;
        return result;
    }

    void* enqueueMapImage(
         const Image& image,
         cl_bool blocking,
         cl_map_flags flags,
         const array<size_type, 2>& origin,
         const array<size_type, 2>& region,
         size_type* row_pitch,
         size_type* slice_pitch,
         const vector<Event>* events = nullptr,
         Event* event = nullptr,
         cl_int* err = nullptr) const
    {
        return enqueueMapImage(image, blocking, flags,
                               { origin[0], origin[1], 0 },
                               { region[0], region[1], 1 }, row_pitch,
                               slice_pitch, events, event, err);
    }

#if CL_HPP_TARGET_OPENCL_VERSION >= 200

    /**
    * Enqueues a command that copies a region of memory from the source pointer to the destination pointer.
    * This function is specifically for transferring data between the host and a coarse-grained SVM buffer.
    */
    template<typename T>
    cl_int enqueueMemcpySVM(
            T *dst_ptr,
            const T *src_ptr,
            cl_bool blocking,
            size_type size,
            const vector<Event> *events = nullptr,
            Event *event = nullptr) const {
        cl_event tmp;
        cl_int err = detail::errHandler(::clEnqueueSVMMemcpy(
                object_, blocking, static_cast<void *>(dst_ptr), static_cast<const void *>(src_ptr), size,
                (events != nullptr) ? (cl_uint) events->size() : 0,
                (events != nullptr && events->size() > 0) ? (cl_event *) &events->front() : nullptr,
                (event != nullptr) ? &tmp : nullptr), __ENQUEUE_COPY_SVM_ERR);

        if (event != nullptr && err == CL_SUCCESS)
            *event = tmp;

        return err;
    }

    /**
    *Enqueues a command that will copy data from one coarse-grained SVM buffer to another.
    *This function takes two cl::pointer instances representing the destination and source buffers.
    */
    template<typename T, class D>
    cl_int enqueueMemcpySVM(
            cl::pointer<T, D> &dst_ptr,
            const cl::pointer<T, D> &src_ptr,
            cl_bool blocking,
            size_type size,
            const vector<Event> *events = nullptr,
            Event *event = nullptr) const {
        cl_event tmp;
        cl_int err = detail::errHandler(::clEnqueueSVMMemcpy(
                object_, blocking, static_cast<void *>(dst_ptr.get()), static_cast<const void *>(src_ptr.get()),
                size,
                (events != nullptr) ? (cl_uint) events->size() : 0,
                (events != nullptr && events->size() > 0) ? (cl_event *) &events->front() : nullptr,
                (event != nullptr) ? &tmp : nullptr), __ENQUEUE_COPY_SVM_ERR);

        if (event != nullptr && err == CL_SUCCESS)
            *event = tmp;

        return err;
    }

    /**
    * Enqueues a command that will allow the host to update a region of a coarse-grained SVM buffer.
    * This variant takes a cl::vector instance.
    */
    template<typename T, class Alloc>
    cl_int enqueueMemcpySVM(
            cl::vector<T, Alloc> &dst_container,
            const cl::vector<T, Alloc> &src_container,
            cl_bool blocking,
            const vector<Event> *events = nullptr,
            Event *event = nullptr) const {
        cl_event tmp;
        if(src_container.size() != dst_container.size()){
            return detail::errHandler(CL_INVALID_VALUE,__ENQUEUE_COPY_SVM_ERR);
        }
        cl_int err = detail::errHandler(::clEnqueueSVMMemcpy(
                object_, blocking, static_cast<void *>(dst_container.data()),
                static_cast<const void *>(src_container.data()),
                dst_container.size() * sizeof(T),
                (events != nullptr) ? (cl_uint) events->size() : 0,
                (events != nullptr && events->size() > 0) ? (cl_event *) &events->front() : nullptr,
                (event != NULL) ? &tmp : nullptr), __ENQUEUE_COPY_SVM_ERR);

        if (event != nullptr && err == CL_SUCCESS)
            *event = tmp;

        return err;
    }

    /**
    * Enqueues a command to fill a SVM buffer with a pattern.
    *
    */
    template<typename T, typename PatternType>
    cl_int enqueueMemFillSVM(
            T *ptr,
            PatternType pattern,
            size_type size,
            const vector<Event> *events = nullptr,
            Event *event = nullptr) const {
        cl_event tmp;
        cl_int err = detail::errHandler(::clEnqueueSVMMemFill(
                object_, static_cast<void *>(ptr), static_cast<void *>(&pattern),
                sizeof(PatternType), size,
                (events != nullptr) ? (cl_uint) events->size() : 0,
                (events != nullptr && events->size() > 0) ? (cl_event *) &events->front() : nullptr,
                (event != nullptr) ? &tmp : nullptr), __ENQUEUE_FILL_SVM_ERR);

        if (event != nullptr && err == CL_SUCCESS)
            *event = tmp;

        return err;
    }

    /**
    * Enqueues a command that fills a region of a coarse-grained SVM buffer with a specified pattern.
    * This variant takes a cl::pointer instance.
    */
    template<typename T, class D, typename PatternType>
    cl_int enqueueMemFillSVM(
            cl::pointer<T, D> &ptr,
            PatternType pattern,
            size_type size,
            const vector<Event> *events = nullptr,
            Event *event = nullptr) const {
        cl_event tmp;
        cl_int err = detail::errHandler(::clEnqueueSVMMemFill(
                object_, static_cast<void *>(ptr.get()), static_cast<void *>(&pattern),
                sizeof(PatternType), size,
                (events != nullptr) ? (cl_uint) events->size() : 0,
                (events != nullptr && events->size() > 0) ? (cl_event *) &events->front() : nullptr,
                (event != nullptr) ? &tmp : nullptr), __ENQUEUE_FILL_SVM_ERR);

        if (event != nullptr && err == CL_SUCCESS)
            *event = tmp;

        return err;
    }

    /**
    * Enqueues a command that will allow the host to fill a region of a coarse-grained SVM buffer with a specified pattern.
    * This variant takes a cl::vector instance.
    */
    template<typename T, class Alloc, typename PatternType>
    cl_int enqueueMemFillSVM(
            cl::vector<T, Alloc> &container,
            PatternType pattern,
            const vector<Event> *events = nullptr,
            Event* event = nullptr) const
    {
        cl_event tmp;
        cl_int err = detail::errHandler(::clEnqueueSVMMemFill(
                object_, static_cast<void *>(container.data()), static_cast<void *>(&pattern),
                sizeof(PatternType), container.size() * sizeof(T),
                (events != nullptr) ? (cl_uint) events->size() : 0,
                (events != nullptr && events->size() > 0) ? (cl_event *) &events->front() : nullptr,
                (event != nullptr) ? &tmp : NULL), __ENQUEUE_FILL_SVM_ERR);

        if (event != nullptr && err == CL_SUCCESS)
            *event = tmp;

        return err;
    }

    /**
     * Enqueues a command that will allow the host to update a region of a coarse-grained SVM buffer.
     * This variant takes a raw SVM pointer.
     */
    template<typename T>
    cl_int enqueueMapSVM(
        T* ptr,
        cl_bool blocking,
        cl_map_flags flags,
        size_type size,
        const vector<Event>* events = nullptr,
        Event* event = nullptr) const
    {
        cl_event tmp;
        cl_int err = detail::errHandler(::clEnqueueSVMMap(
            object_, blocking, flags, static_cast<void*>(ptr), size,
            (events != nullptr) ? (cl_uint)events->size() : 0,
            (events != nullptr && events->size() > 0) ? (cl_event*)&events->front() : nullptr,
            (event != nullptr) ? &tmp : nullptr),
            __ENQUEUE_MAP_SVM_ERR);

        if (event != nullptr && err == CL_SUCCESS)
            *event = tmp;

        return err;
    }


    /**
     * Enqueues a command that will allow the host to update a region of a coarse-grained SVM buffer.
     * This variant takes a cl::pointer instance.
     */
    template<typename T, class D>
    cl_int enqueueMapSVM(
        cl::pointer<T, D> &ptr,
        cl_bool blocking,
        cl_map_flags flags,
        size_type size,
        const vector<Event>* events = nullptr,
        Event* event = nullptr) const
    {
        cl_event tmp;
        cl_int err = detail::errHandler(::clEnqueueSVMMap(
            object_, blocking, flags, static_cast<void*>(ptr.get()), size,
            (events != nullptr) ? (cl_uint)events->size() : 0,
            (events != nullptr && events->size() > 0) ? (cl_event*)&events->front() : nullptr,
            (event != nullptr) ? &tmp : nullptr),
            __ENQUEUE_MAP_SVM_ERR);

        if (event != nullptr && err == CL_SUCCESS)
            *event = tmp;

        return err;
    }

    /**
     * Enqueues a command that will allow the host to update a region of a coarse-grained SVM buffer.
     * This variant takes a cl::vector instance.
     */
    template<typename T, class Alloc>
    cl_int enqueueMapSVM(
        cl::vector<T, Alloc> &container,
        cl_bool blocking,
        cl_map_flags flags,
        const vector<Event>* events = nullptr,
        Event* event = nullptr) const
    {
        cl_event tmp;
        cl_int err = detail::errHandler(::clEnqueueSVMMap(
            object_, blocking, flags, static_cast<void*>(container.data()), container.size()*sizeof(T),
            (events != nullptr) ? (cl_uint)events->size() : 0,
            (events != nullptr && events->size() > 0) ? (cl_event*)&events->front() : nullptr,
            (event != nullptr) ? &tmp : nullptr),
            __ENQUEUE_MAP_SVM_ERR);

        if (event != nullptr && err == CL_SUCCESS)
            *event = tmp;

        return err;
    }
#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200

    cl_int enqueueUnmapMemObject(
        const Memory& memory,
        void* mapped_ptr,
        const vector<Event>* events = nullptr,
        Event* event = nullptr) const
    {
        cl_event tmp;
        cl_int err = detail::errHandler(
            call_clEnqueueUnmapMemObject(
                object_, memory(), mapped_ptr,
                (events != nullptr) ? (cl_uint) events->size() : 0,
                (events != nullptr && events->size() > 0) ? (cl_event*) &events->front() : nullptr,
                (event != nullptr) ? &tmp : nullptr),
            __ENQUEUE_UNMAP_MEM_OBJECT_ERR);

        if (event != nullptr && err == CL_SUCCESS)
            *event = tmp;

        return err;
    }


#if CL_HPP_TARGET_OPENCL_VERSION >= 200
    /**
     * Enqueues a command that will release a coarse-grained SVM buffer back to the OpenCL runtime.
     * This variant takes a raw SVM pointer.
     */
    template<typename T>
    cl_int enqueueUnmapSVM(
        T* ptr,
        const vector<Event>* events = nullptr,
        Event* event = nullptr) const
    {
        cl_event tmp;
        cl_int err = detail::errHandler(
            ::clEnqueueSVMUnmap(
            object_, static_cast<void*>(ptr),
            (events != nullptr) ? (cl_uint)events->size() : 0,
            (events != nullptr && events->size() > 0) ? (cl_event*)&events->front() : nullptr,
            (event != nullptr) ? &tmp : nullptr),
            __ENQUEUE_UNMAP_SVM_ERR);

        if (event != nullptr && err == CL_SUCCESS)
            *event = tmp;

        return err;
    }

    /**
     * Enqueues a command that will release a coarse-grained SVM buffer back to the OpenCL runtime.
     * This variant takes a cl::pointer instance.
     */
    template<typename T, class D>
    cl_int enqueueUnmapSVM(
        cl::pointer<T, D> &ptr,
        const vector<Event>* events = nullptr,
        Event* event = nullptr) const
    {
        cl_event tmp;
        cl_int err = detail::errHandler(
            ::clEnqueueSVMUnmap(
            object_, static_cast<void*>(ptr.get()),
            (events != nullptr) ? (cl_uint)events->size() : 0,
            (events != nullptr && events->size() > 0) ? (cl_event*)&events->front() : nullptr,
            (event != nullptr) ? &tmp : nullptr),
            __ENQUEUE_UNMAP_SVM_ERR);

        if (event != nullptr && err == CL_SUCCESS)
            *event = tmp;

        return err;
    }

    /**
     * Enqueues a command that will release a coarse-grained SVM buffer back to the OpenCL runtime.
     * This variant takes a cl::vector instance.
     */
    template<typename T, class Alloc>
    cl_int enqueueUnmapSVM(
        cl::vector<T, Alloc> &container,
        const vector<Event>* events = nullptr,
        Event* event = nullptr) const
    {
        cl_event tmp;
        cl_int err = detail::errHandler(
            ::clEnqueueSVMUnmap(
            object_, static_cast<void*>(container.data()),
            (events != nullptr) ? (cl_uint)events->size() : 0,
            (events != nullptr && events->size() > 0) ? (cl_event*)&events->front() : nullptr,
            (event != nullptr) ? &tmp : nullptr),
            __ENQUEUE_UNMAP_SVM_ERR);

        if (event != nullptr && err == CL_SUCCESS)
            *event = tmp;

        return err;
    }
#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200

#if CL_HPP_TARGET_OPENCL_VERSION >= 120
    /**
     * Enqueues a marker command which waits for either a list of events to complete, 
     * or all previously enqueued commands to complete.
     *
     * Enqueues a marker command which waits for either a list of events to complete, 
     * or if the list is empty it waits for all commands previously enqueued in command_queue 
     * to complete before it completes. This command returns an event which can be waited on, 
     * i.e. this event can be waited on to insure that all events either in the event_wait_list 
     * or all previously enqueued commands, queued before this command to command_queue, 
     * have completed.
     */
    cl_int enqueueMarkerWithWaitList(
        const vector<Event> *events = nullptr,
        Event *event = nullptr) const
    {
        cl_event tmp;
        cl_int err = detail::errHandler(
            call_clEnqueueMarkerWithWaitList(
                object_,
                (events != nullptr) ? (cl_uint) events->size() : 0,
                (events != nullptr && events->size() > 0) ? (cl_event*) &events->front() : nullptr,
                (event != nullptr) ? &tmp : nullptr),
            __ENQUEUE_MARKER_WAIT_LIST_ERR);

        if (event != nullptr && err == CL_SUCCESS)
            *event = tmp;

        return err;
    }

    /**
     * A synchronization point that enqueues a barrier operation.
     *
     * Enqueues a barrier command which waits for either a list of events to complete, 
     * or if the list is empty it waits for all commands previously enqueued in command_queue 
     * to complete before it completes. This command blocks command execution, that is, any 
     * following commands enqueued after it do not execute until it completes. This command 
     * returns an event which can be waited on, i.e. this event can be waited on to insure that 
     * all events either in the event_wait_list or all previously enqueued commands, queued 
     * before this command to command_queue, have completed.
     */
    cl_int enqueueBarrierWithWaitList(
        const vector<Event> *events = nullptr,
        Event *event = nullptr) const
    {
        cl_event tmp;
        cl_int err = detail::errHandler(
            call_clEnqueueBarrierWithWaitList(
                object_,
                (events != nullptr) ? (cl_uint) events->size() : 0,
                (events != nullptr && events->size() > 0) ? (cl_event*) &events->front() : nullptr,
                (event != nullptr) ? &tmp : nullptr),
            __ENQUEUE_BARRIER_WAIT_LIST_ERR);

        if (event != nullptr && err == CL_SUCCESS)
            *event = tmp;

        return err;
    }
    
    /**
     * Enqueues a command to indicate with which device a set of memory objects
     * should be associated.
     */
    cl_int enqueueMigrateMemObjects(
        const vector<Memory> &memObjects,
        cl_mem_migration_flags flags,
        const vector<Event>* events = nullptr,
        Event* event = nullptr
        ) const
    {
        cl_event tmp;
        
        vector<cl_mem> localMemObjects(memObjects.size());

        for( int i = 0; i < (int)memObjects.size(); ++i ) {
            localMemObjects[i] = memObjects[i]();
        }
        
        cl_int err = detail::errHandler(
            ::clEnqueueMigrateMemObjects(
                object_, 
                (cl_uint)memObjects.size(), 
                localMemObjects.data(),
                flags,
                (events != nullptr) ? (cl_uint) events->size() : 0,
                (events != nullptr && events->size() > 0) ? (cl_event*) &events->front() : nullptr,
                (event != nullptr) ? &tmp : nullptr),
            __ENQUEUE_UNMAP_MEM_OBJECT_ERR);

        if (event != nullptr && err == CL_SUCCESS)
            *event = tmp;

        return err;
    }
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120


#if CL_HPP_TARGET_OPENCL_VERSION >= 210
    /**
     * Enqueues a command that will allow the host associate ranges within a set of
     * SVM allocations with a device.
     * @param sizes - The length from each pointer to migrate.
     */
    template<typename T>
    cl_int enqueueMigrateSVM(
        const cl::vector<T*> &svmRawPointers,
        const cl::vector<size_type> &sizes,
        cl_mem_migration_flags flags = 0,
        const vector<Event>* events = nullptr,
        Event* event = nullptr) const
    {
        cl_event tmp;
        cl_int err = detail::errHandler(::clEnqueueSVMMigrateMem(
            object_,
            svmRawPointers.size(), static_cast<void**>(svmRawPointers.data()),
            sizes.data(), // array of sizes not passed
            flags,
            (events != nullptr) ? (cl_uint)events->size() : 0,
            (events != nullptr && events->size() > 0) ? (cl_event*)&events->front() : nullptr,
            (event != nullptr) ? &tmp : nullptr),
            __ENQUEUE_MIGRATE_SVM_ERR);

        if (event != nullptr && err == CL_SUCCESS)
            *event = tmp;

        return err;
    }

    /**
     * Enqueues a command that will allow the host associate a set of SVM allocations with
     * a device.
     */
    template<typename T>
    cl_int enqueueMigrateSVM(
        const cl::vector<T*> &svmRawPointers,
        cl_mem_migration_flags flags = 0,
        const vector<Event>* events = nullptr,
        Event* event = nullptr) const
    {
        return enqueueMigrateSVM(svmRawPointers, cl::vector<size_type>(svmRawPointers.size()), flags, events, event);
    }


    /**
     * Enqueues a command that will allow the host associate ranges within a set of
     * SVM allocations with a device.
     * @param sizes - The length from each pointer to migrate.
     */
    template<typename T, class D>
    cl_int enqueueMigrateSVM(
        const cl::vector<cl::pointer<T, D>> &svmPointers,
        const cl::vector<size_type> &sizes,
        cl_mem_migration_flags flags = 0,
        const vector<Event>* events = nullptr,
        Event* event = nullptr) const
    {
        cl::vector<void*> svmRawPointers;
        svmRawPointers.reserve(svmPointers.size());
        for (auto p : svmPointers) {
            svmRawPointers.push_back(static_cast<void*>(p.get()));
        }

        return enqueueMigrateSVM(svmRawPointers, sizes, flags, events, event);
    }


    /**
     * Enqueues a command that will allow the host associate a set of SVM allocations with
     * a device.
     */
    template<typename T, class D>
    cl_int enqueueMigrateSVM(
        const cl::vector<cl::pointer<T, D>> &svmPointers,
        cl_mem_migration_flags flags = 0,
        const vector<Event>* events = nullptr,
        Event* event = nullptr) const
    {
        return enqueueMigrateSVM(svmPointers, cl::vector<size_type>(svmPointers.size()), flags, events, event);
    }

    /**
     * Enqueues a command that will allow the host associate ranges within a set of
     * SVM allocations with a device.
     * @param sizes - The length from the beginning of each container to migrate.
     */
    template<typename T, class Alloc>
    cl_int enqueueMigrateSVM(
        const cl::vector<cl::vector<T, Alloc>> &svmContainers,
        const cl::vector<size_type> &sizes,
        cl_mem_migration_flags flags = 0,
        const vector<Event>* events = nullptr,
        Event* event = nullptr) const
    {
        cl::vector<void*> svmRawPointers;
        svmRawPointers.reserve(svmContainers.size());
        for (auto p : svmContainers) {
            svmRawPointers.push_back(static_cast<void*>(p.data()));
        }

        return enqueueMigrateSVM(svmRawPointers, sizes, flags, events, event);
    }

    /**
     * Enqueues a command that will allow the host associate a set of SVM allocations with
     * a device.
     */
    template<typename T, class Alloc>
    cl_int enqueueMigrateSVM(
        const cl::vector<cl::vector<T, Alloc>> &svmContainers,
        cl_mem_migration_flags flags = 0,
        const vector<Event>* events = nullptr,
        Event* event = nullptr) const
    {
        return enqueueMigrateSVM(svmContainers, cl::vector<size_type>(svmContainers.size()), flags, events, event);
    }

#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 210
    
    cl_int enqueueNDRangeKernel(
        const Kernel& kernel,
        const NDRange& offset,
        const NDRange& global,
        const NDRange& local = NullRange,
        const vector<Event>* events = nullptr,
        Event* event = nullptr) const
    {
        cl_event tmp;
        cl_int err = detail::errHandler(
            call_clEnqueueNDRangeKernel(
                object_, kernel(), (cl_uint) global.dimensions(),
                offset.dimensions() != 0 ? (const size_type*) offset : nullptr,
                (const size_type*) global,
                local.dimensions() != 0 ? (const size_type*) local : nullptr,
                (events != nullptr) ? (cl_uint) events->size() : 0,
                (events != nullptr && events->size() > 0) ? (cl_event*) &events->front() : nullptr,
                (event != nullptr) ? &tmp : nullptr),
            __ENQUEUE_NDRANGE_KERNEL_ERR);

        if (event != nullptr && err == CL_SUCCESS)
            *event = tmp;

        return err;
    }

#if defined(CL_USE_DEPRECATED_OPENCL_1_2_APIS)
    CL_API_PREFIX__VERSION_1_2_DEPRECATED cl_int enqueueTask(
        const Kernel& kernel,
        const vector<Event>* events = nullptr,
        Event* event = nullptr) const CL_API_SUFFIX__VERSION_1_2_DEPRECATED
    {
        cl_event tmp;
        cl_int err = detail::errHandler(
            ::clEnqueueTask(
                object_, kernel(),
                (events != nullptr) ? (cl_uint) events->size() : 0,
                (events != nullptr && events->size() > 0) ? (cl_event*) &events->front() : nullptr,
                (event != nullptr) ? &tmp : nullptr),
            __ENQUEUE_TASK_ERR);

        if (event != nullptr && err == CL_SUCCESS)
            *event = tmp;

        return err;
    }
#endif // #if defined(CL_USE_DEPRECATED_OPENCL_1_2_APIS)

    cl_int enqueueNativeKernel(
        void (CL_CALLBACK *userFptr)(void *),
        std::pair<void*, size_type> args,
        const vector<Memory>* mem_objects = nullptr,
        const vector<const void*>* mem_locs = nullptr,
        const vector<Event>* events = nullptr,
        Event* event = nullptr) const
    {
        cl_event tmp;
        cl_int err = detail::errHandler(
            ::clEnqueueNativeKernel(
                object_, userFptr, args.first, args.second,
                (mem_objects != nullptr) ? (cl_uint) mem_objects->size() : 0,
                (mem_objects->size() > 0 ) ? reinterpret_cast<const cl_mem *>(mem_objects->data()) : nullptr,
                (mem_locs != nullptr && mem_locs->size() > 0) ? (const void **) &mem_locs->front() : nullptr,
                (events != nullptr) ? (cl_uint) events->size() : 0,
                (events != nullptr && events->size() > 0) ? (cl_event*) &events->front() : nullptr,
                (event != nullptr) ? &tmp : nullptr),
            __ENQUEUE_NATIVE_KERNEL);

        if (event != nullptr && err == CL_SUCCESS)
            *event = tmp;

        return err;
    }

/**
 * Deprecated APIs for 1.2
 */
#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
    CL_API_PREFIX__VERSION_1_1_DEPRECATED 
    cl_int enqueueMarker(Event* event = nullptr) const CL_API_SUFFIX__VERSION_1_1_DEPRECATED
    {
        cl_event tmp;
        cl_int err = detail::errHandler(
            ::clEnqueueMarker(
                object_, 
                (event != nullptr) ? &tmp : nullptr),
            __ENQUEUE_MARKER_ERR);

        if (event != nullptr && err == CL_SUCCESS)
            *event = tmp;

        return err;
    }

    CL_API_PREFIX__VERSION_1_1_DEPRECATED
    cl_int enqueueWaitForEvents(const vector<Event>& events) const CL_API_SUFFIX__VERSION_1_1_DEPRECATED
    {
        return detail::errHandler(
            ::clEnqueueWaitForEvents(
                object_,
                (cl_uint) events.size(),
                events.size() > 0 ? (const cl_event*) &events.front() : nullptr),
            __ENQUEUE_WAIT_FOR_EVENTS_ERR);
    }
#endif // defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)

    cl_int enqueueAcquireGLObjects(
         const vector<Memory>* mem_objects = nullptr,
         const vector<Event>* events = nullptr,
         Event* event = nullptr) const
     {
        cl_event tmp;
        cl_int err = detail::errHandler(
             ::clEnqueueAcquireGLObjects(
                 object_,
                 (mem_objects != nullptr) ? (cl_uint) mem_objects->size() : 0,
                 (mem_objects != nullptr && mem_objects->size() > 0) ? (const cl_mem *) &mem_objects->front(): nullptr,
                 (events != nullptr) ? (cl_uint) events->size() : 0,
                 (events != nullptr && events->size() > 0) ? (cl_event*) &events->front() : nullptr,
                 (event != nullptr) ? &tmp : nullptr),
             __ENQUEUE_ACQUIRE_GL_ERR);

        if (event != nullptr && err == CL_SUCCESS)
            *event = tmp;

        return err;
     }

    cl_int enqueueReleaseGLObjects(
         const vector<Memory>* mem_objects = nullptr,
         const vector<Event>* events = nullptr,
         Event* event = nullptr) const
     {
        cl_event tmp;
        cl_int err = detail::errHandler(
             ::clEnqueueReleaseGLObjects(
                 object_,
                 (mem_objects != nullptr) ? (cl_uint) mem_objects->size() : 0,
                 (mem_objects != nullptr && mem_objects->size() > 0) ? (const cl_mem *) &mem_objects->front(): nullptr,
                 (events != nullptr) ? (cl_uint) events->size() : 0,
                 (events != nullptr && events->size() > 0) ? (cl_event*) &events->front() : nullptr,
                 (event != nullptr) ? &tmp : nullptr),
             __ENQUEUE_RELEASE_GL_ERR);

        if (event != nullptr && err == CL_SUCCESS)
            *event = tmp;

        return err;
     }

#if defined (CL_HPP_USE_DX_INTEROP)
typedef CL_API_ENTRY cl_int (CL_API_CALL *PFN_clEnqueueAcquireD3D10ObjectsKHR)(
    cl_command_queue command_queue, cl_uint num_objects,
    const cl_mem* mem_objects, cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list, cl_event* event);
typedef CL_API_ENTRY cl_int (CL_API_CALL *PFN_clEnqueueReleaseD3D10ObjectsKHR)(
    cl_command_queue command_queue, cl_uint num_objects,
    const cl_mem* mem_objects,  cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list, cl_event* event);

    cl_int enqueueAcquireD3D10Objects(
         const vector<Memory>* mem_objects = nullptr,
         const vector<Event>* events = nullptr,
         Event* event = nullptr) const
    {
        static PFN_clEnqueueAcquireD3D10ObjectsKHR pfn_clEnqueueAcquireD3D10ObjectsKHR = nullptr;
#if CL_HPP_TARGET_OPENCL_VERSION >= 120
        cl_context context = getInfo<CL_QUEUE_CONTEXT>();
        cl::Device device(getInfo<CL_QUEUE_DEVICE>());
        cl_platform_id platform = device.getInfo<CL_DEVICE_PLATFORM>();
        CL_HPP_INIT_CL_EXT_FCN_PTR_PLATFORM_(platform, clEnqueueAcquireD3D10ObjectsKHR);
#endif
#if CL_HPP_MINIMUM_OPENCL_VERSION < 120
        CL_HPP_INIT_CL_EXT_FCN_PTR_(clEnqueueAcquireD3D10ObjectsKHR);
#endif
        
        cl_event tmp;
        cl_int err = detail::errHandler(
             pfn_clEnqueueAcquireD3D10ObjectsKHR(
                 object_,
                 (mem_objects != nullptr) ? (cl_uint) mem_objects->size() : 0,
                 (mem_objects != nullptr && mem_objects->size() > 0) ? (const cl_mem *) &mem_objects->front(): nullptr,
                 (events != nullptr) ? (cl_uint) events->size() : 0,
                 (events != nullptr) ? (cl_event*) &events->front() : nullptr,
                 (event != nullptr) ? &tmp : nullptr),
             __ENQUEUE_ACQUIRE_GL_ERR);

        if (event != nullptr && err == CL_SUCCESS)
            *event = tmp;

        return err;
     }

    cl_int enqueueReleaseD3D10Objects(
         const vector<Memory>* mem_objects = nullptr,
         const vector<Event>* events = nullptr,
         Event* event = nullptr) const
    {
        static PFN_clEnqueueReleaseD3D10ObjectsKHR pfn_clEnqueueReleaseD3D10ObjectsKHR = nullptr;
#if CL_HPP_TARGET_OPENCL_VERSION >= 120
        cl_context context = getInfo<CL_QUEUE_CONTEXT>();
        cl::Device device(getInfo<CL_QUEUE_DEVICE>());
        cl_platform_id platform = device.getInfo<CL_DEVICE_PLATFORM>();
        CL_HPP_INIT_CL_EXT_FCN_PTR_PLATFORM_(platform, clEnqueueReleaseD3D10ObjectsKHR);
#endif
#if CL_HPP_MINIMUM_OPENCL_VERSION < 120
        CL_HPP_INIT_CL_EXT_FCN_PTR_(clEnqueueReleaseD3D10ObjectsKHR);
#endif

        cl_event tmp;
        cl_int err = detail::errHandler(
            pfn_clEnqueueReleaseD3D10ObjectsKHR(
                object_,
                (mem_objects != nullptr) ? (cl_uint) mem_objects->size() : 0,
                (mem_objects != nullptr && mem_objects->size() > 0) ? (const cl_mem *) &mem_objects->front(): nullptr,
                (events != nullptr) ? (cl_uint) events->size() : 0,
                (events != nullptr && events->size() > 0) ? (cl_event*) &events->front() : nullptr,
                (event != nullptr) ? &tmp : nullptr),
            __ENQUEUE_RELEASE_GL_ERR);

        if (event != nullptr && err == CL_SUCCESS)
            *event = tmp;

        return err;
    }
#endif

/**
 * Deprecated APIs for 1.2
 */
#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
    CL_API_PREFIX__VERSION_1_1_DEPRECATED
    cl_int enqueueBarrier() const CL_API_SUFFIX__VERSION_1_1_DEPRECATED
    {
        return detail::errHandler(
            ::clEnqueueBarrier(object_),
            __ENQUEUE_BARRIER_ERR);
    }
#endif // CL_USE_DEPRECATED_OPENCL_1_1_APIS

    cl_int flush() const
    {
        return detail::errHandler(call_clFlush(object_), __FLUSH_ERR);
    }

    cl_int finish() const
    {
        return detail::errHandler(call_clFinish(object_), __FINISH_ERR);
    }

#ifdef cl_khr_external_memory
    cl_int enqueueAcquireExternalMemObjects(
        const vector<Memory>& mem_objects,
        const vector<Event>* events_wait = nullptr,
        Event *event = nullptr)
    {
        cl_int err = CL_INVALID_OPERATION;
        cl_event tmp;

        std::call_once(ext_memory_initialized_, initMemoryExtension, this->getInfo<CL_QUEUE_DEVICE>());

        if (pfn_clEnqueueAcquireExternalMemObjectsKHR)
        {
            err = pfn_clEnqueueAcquireExternalMemObjectsKHR(
                object_,
                static_cast<cl_uint>(mem_objects.size()),
                (mem_objects.size() > 0) ? reinterpret_cast<const cl_mem *>(mem_objects.data()) : nullptr,
                (events_wait != nullptr) ? static_cast<cl_uint>(events_wait->size()) : 0,
                (events_wait != nullptr && events_wait->size() > 0) ? reinterpret_cast<const cl_event*>(events_wait->data()) : nullptr,
                &tmp);
        }

        detail::errHandler(err, __ENQUEUE_ACQUIRE_EXTERNAL_MEMORY_ERR);

        if (event != nullptr && err == CL_SUCCESS)
            *event = tmp;

        return err;
    }

    cl_int enqueueReleaseExternalMemObjects(
        const vector<Memory>& mem_objects,
        const vector<Event>* events_wait = nullptr,
        Event *event = nullptr)
    {
        cl_int err = CL_INVALID_OPERATION;
        cl_event tmp;

        std::call_once(ext_memory_initialized_, initMemoryExtension, this->getInfo<CL_QUEUE_DEVICE>());

        if (pfn_clEnqueueReleaseExternalMemObjectsKHR)
        {
            err = pfn_clEnqueueReleaseExternalMemObjectsKHR(
                object_,
                static_cast<cl_uint>(mem_objects.size()),
                (mem_objects.size() > 0) ? reinterpret_cast<const cl_mem *>(mem_objects.data()) : nullptr,
                (events_wait != nullptr) ? static_cast<cl_uint>(events_wait->size()) : 0,
                (events_wait != nullptr && events_wait->size() > 0) ? reinterpret_cast<const cl_event*>(events_wait->data()) : nullptr,
                &tmp);
        }

        detail::errHandler(err, __ENQUEUE_RELEASE_EXTERNAL_MEMORY_ERR);

        if (event != nullptr && err == CL_SUCCESS)
            *event = tmp;

        return err;
    }
#endif // cl_khr_external_memory && CL_HPP_TARGET_OPENCL_VERSION >= 300

#ifdef cl_khr_semaphore
    cl_int enqueueWaitSemaphores(
        const vector<Semaphore> &sema_objects,
        const vector<cl_semaphore_payload_khr> &sema_payloads = {},
        const vector<Event>* events_wait_list = nullptr,
        Event *event = nullptr) const;

    cl_int enqueueSignalSemaphores(
        const vector<Semaphore> &sema_objects,
        const vector<cl_semaphore_payload_khr>& sema_payloads = {},
        const vector<Event>* events_wait_list = nullptr,
        Event* event = nullptr);
#endif // cl_khr_semaphore
}; // CommandQueue

#ifdef cl_khr_external_memory
CL_HPP_DEFINE_STATIC_MEMBER_ std::once_flag CommandQueue::ext_memory_initialized_;
#endif

CL_HPP_DEFINE_STATIC_MEMBER_ std::once_flag CommandQueue::default_initialized_;
CL_HPP_DEFINE_STATIC_MEMBER_ CommandQueue CommandQueue::default_;
CL_HPP_DEFINE_STATIC_MEMBER_ cl_int CommandQueue::default_error_ = CL_SUCCESS;


#if CL_HPP_TARGET_OPENCL_VERSION >= 200
enum class DeviceQueueProperties : cl_command_queue_properties
{
    None = 0,
    Profiling = CL_QUEUE_PROFILING_ENABLE,
};

inline DeviceQueueProperties operator|(DeviceQueueProperties lhs, DeviceQueueProperties rhs)
{
    return static_cast<DeviceQueueProperties>(static_cast<cl_command_queue_properties>(lhs) | static_cast<cl_command_queue_properties>(rhs));
}

/*! \class DeviceCommandQueue
 * \brief DeviceCommandQueue interface for device cl_command_queues.
 */
class DeviceCommandQueue : public detail::Wrapper<cl_command_queue>
{
public:

    /*!
     * Trivial empty constructor to create a null queue.
     */
    DeviceCommandQueue() { }

    /*!
     * Default construct device command queue on default context and device
     */
    DeviceCommandQueue(DeviceQueueProperties properties, cl_int* err = nullptr)
    {
        cl_int error;
        cl::Context context = cl::Context::getDefault();
        cl::Device device = cl::Device::getDefault();

        cl_command_queue_properties mergedProperties =
            CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE | static_cast<cl_command_queue_properties>(properties);

        cl_queue_properties queue_properties[] = {
            CL_QUEUE_PROPERTIES, mergedProperties, 0 };
        object_ = call_clCreateCommandQueueWithProperties(
            context(), device(), queue_properties, &error);

        detail::errHandler(error, __CREATE_COMMAND_QUEUE_WITH_PROPERTIES_ERR);
        if (err != nullptr) {
            *err = error;
        }
    }

    /*!
     * Create a device command queue for a specified device in the passed context.
     */
    DeviceCommandQueue(
        const Context& context,
        const Device& device,
        DeviceQueueProperties properties = DeviceQueueProperties::None,
        cl_int* err = nullptr)
    {
        cl_int error;

        cl_command_queue_properties mergedProperties =
            CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE | static_cast<cl_command_queue_properties>(properties);
        cl_queue_properties queue_properties[] = {
            CL_QUEUE_PROPERTIES, mergedProperties, 0 };
        object_ = call_clCreateCommandQueueWithProperties(
            context(), device(), queue_properties, &error);

        detail::errHandler(error, __CREATE_COMMAND_QUEUE_WITH_PROPERTIES_ERR);
        if (err != nullptr) {
            *err = error;
        }
    }

    /*!
     * Create a device command queue for a specified device in the passed context.
     */
    DeviceCommandQueue(
        const Context& context,
        const Device& device,
        cl_uint queueSize,
        DeviceQueueProperties properties = DeviceQueueProperties::None,
        cl_int* err = nullptr)
    {
        cl_int error;

        cl_command_queue_properties mergedProperties =
            CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE | static_cast<cl_command_queue_properties>(properties);
        cl_queue_properties queue_properties[] = {
            CL_QUEUE_PROPERTIES, mergedProperties,
            CL_QUEUE_SIZE, queueSize, 
            0 };
        object_ = call_clCreateCommandQueueWithProperties(
            context(), device(), queue_properties, &error);

        detail::errHandler(error, __CREATE_COMMAND_QUEUE_WITH_PROPERTIES_ERR);
        if (err != nullptr) {
            *err = error;
        }
    }

    /*! \brief Constructor from cl_command_queue - takes ownership.
    *
    * \param retainObject will cause the constructor to retain its cl object.
    *                     Defaults to false to maintain compatibility with
    *                     earlier versions.
    */
    explicit DeviceCommandQueue(const cl_command_queue& commandQueue, bool retainObject = false) :
        detail::Wrapper<cl_type>(commandQueue, retainObject) { }

    DeviceCommandQueue& operator = (const cl_command_queue& rhs)
    {
        detail::Wrapper<cl_type>::operator=(rhs);
        return *this;
    }

    template <typename T>
    cl_int getInfo(cl_command_queue_info name, T* param) const
    {
        return detail::errHandler(
            detail::getInfo(
            &::clGetCommandQueueInfo, object_, name, param),
            __GET_COMMAND_QUEUE_INFO_ERR);
    }

    template <cl_command_queue_info name> typename
        detail::param_traits<detail::cl_command_queue_info, name>::param_type
        getInfo(cl_int* err = nullptr) const
    {
        typename detail::param_traits<
            detail::cl_command_queue_info, name>::param_type param;
        cl_int result = getInfo(name, &param);
        if (err != nullptr) {
            *err = result;
        }
        return param;
    }

    /*!
     * Create a new default device command queue for the default device,
     * in the default context and of the default size.
     * If there is already a default queue for the specified device this
     * function will return the pre-existing queue.
     */
    static DeviceCommandQueue makeDefault(
        cl_int *err = nullptr)
    {
        cl_int error;
        cl::Context context = cl::Context::getDefault();
        cl::Device device = cl::Device::getDefault();

        cl_command_queue_properties properties =
            CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE | CL_QUEUE_ON_DEVICE_DEFAULT;
        cl_queue_properties queue_properties[] = {
            CL_QUEUE_PROPERTIES, properties,
            0 };
        DeviceCommandQueue deviceQueue(
            call_clCreateCommandQueueWithProperties(
            context(), device(), queue_properties, &error));

        detail::errHandler(error, __CREATE_COMMAND_QUEUE_WITH_PROPERTIES_ERR);
        if (err != nullptr) {
            *err = error;
        }

        return deviceQueue;
    }

    /*!
     * Create a new default device command queue for the specified device
     * and of the default size.
     * If there is already a default queue for the specified device this
     * function will return the pre-existing queue.
     */
    static DeviceCommandQueue makeDefault(
        const Context &context, const Device &device, cl_int *err = nullptr)
    {
        cl_int error;

        cl_command_queue_properties properties =
            CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE | CL_QUEUE_ON_DEVICE_DEFAULT;
        cl_queue_properties queue_properties[] = {
            CL_QUEUE_PROPERTIES, properties,
            0 };
        DeviceCommandQueue deviceQueue(
            call_clCreateCommandQueueWithProperties(
            context(), device(), queue_properties, &error));

        detail::errHandler(error, __CREATE_COMMAND_QUEUE_WITH_PROPERTIES_ERR);
        if (err != nullptr) {
            *err = error;
        }

        return deviceQueue;
    }

    /*!
     * Create a new default device command queue for the specified device 
     * and of the requested size in bytes.
     * If there is already a default queue for the specified device this
     * function will return the pre-existing queue.
     */
    static DeviceCommandQueue makeDefault(
        const Context &context, const Device &device, cl_uint queueSize, cl_int *err = nullptr)
    {
        cl_int error;

        cl_command_queue_properties properties =
            CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE | CL_QUEUE_ON_DEVICE_DEFAULT;
        cl_queue_properties queue_properties[] = {
            CL_QUEUE_PROPERTIES, properties,
            CL_QUEUE_SIZE, queueSize,
            0 };
        DeviceCommandQueue deviceQueue(
            call_clCreateCommandQueueWithProperties(
                context(), device(), queue_properties, &error));

        detail::errHandler(error, __CREATE_COMMAND_QUEUE_WITH_PROPERTIES_ERR);
        if (err != nullptr) {
            *err = error;
        }

        return deviceQueue;
    }



#if CL_HPP_TARGET_OPENCL_VERSION >= 210
    /*!
     * Modify the default device command queue to be used for subsequent kernels.
     * This can update the default command queue for a device repeatedly to account
     * for kernels that rely on the default.
     * @return updated default device command queue.
     */
    static DeviceCommandQueue updateDefault(const Context &context, const Device &device, const DeviceCommandQueue &default_queue, cl_int *err = nullptr)
    {
        cl_int error;
        error = clSetDefaultDeviceCommandQueue(context.get(), device.get(), default_queue.get());

        detail::errHandler(error, __SET_DEFAULT_DEVICE_COMMAND_QUEUE_ERR);
        if (err != nullptr) {
            *err = error;
        }
        return default_queue;
    }

    /*!
     * Return the current default command queue for the specified command queue
     */
    static DeviceCommandQueue getDefault(const CommandQueue &queue, cl_int * err = nullptr)
    {
        return queue.getInfo<CL_QUEUE_DEVICE_DEFAULT>(err);
    }

#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 210
}; // DeviceCommandQueue

namespace detail
{
    // Specialization for device command queue
    template <>
    struct KernelArgumentHandler<cl::DeviceCommandQueue, void>
    {
        static size_type size(const cl::DeviceCommandQueue&) { return sizeof(cl_command_queue); }
        static const cl_command_queue* ptr(const cl::DeviceCommandQueue& value) { return &(value()); }
    };
} // namespace detail

#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200


template< typename IteratorType >
Buffer::Buffer(
    const Context &context,
    IteratorType startIterator,
    IteratorType endIterator,
    bool readOnly,
    bool useHostPtr,
    cl_int* err)
{
    typedef typename std::iterator_traits<IteratorType>::value_type DataType;
    cl_int error;

    cl_mem_flags flags = 0;
    if( readOnly ) {
        flags |= CL_MEM_READ_ONLY;
    }
    else {
        flags |= CL_MEM_READ_WRITE;
    }
    if( useHostPtr ) {
        flags |= CL_MEM_USE_HOST_PTR;
    }
    
    size_type size = sizeof(DataType)*(endIterator - startIterator);

    if( useHostPtr ) {
        object_ = ::clCreateBuffer(context(), flags, size, const_cast<DataType*>(&*startIterator), &error);
    } else {
        object_ = ::clCreateBuffer(context(), flags, size, 0, &error);
    }

    detail::errHandler(error, __CREATE_BUFFER_ERR);
    if (err != nullptr) {
        *err = error;
    }

    if( !useHostPtr ) {
        CommandQueue queue(context, 0, &error);
        detail::errHandler(error, __CREATE_BUFFER_ERR);
        if (err != nullptr) {
            *err = error;
        }

        error = cl::copy(queue, startIterator, endIterator, *this);
        detail::errHandler(error, __CREATE_BUFFER_ERR);
        if (err != nullptr) {
            *err = error;
        }
    }
}

template< typename IteratorType >
Buffer::Buffer(
    const CommandQueue &queue,
    IteratorType startIterator,
    IteratorType endIterator,
    bool readOnly,
    bool useHostPtr,
    cl_int* err)
{
    typedef typename std::iterator_traits<IteratorType>::value_type DataType;
    cl_int error;

    cl_mem_flags flags = 0;
    if (readOnly) {
        flags |= CL_MEM_READ_ONLY;
    }
    else {
        flags |= CL_MEM_READ_WRITE;
    }
    if (useHostPtr) {
        flags |= CL_MEM_USE_HOST_PTR;
    }

    size_type size = sizeof(DataType)*(endIterator - startIterator);

    Context context = queue.getInfo<CL_QUEUE_CONTEXT>();

    if (useHostPtr) {
        object_ = ::clCreateBuffer(context(), flags, size, const_cast<DataType*>(&*startIterator), &error);
    }
    else {
        object_ = ::clCreateBuffer(context(), flags, size, 0, &error);
    }

    detail::errHandler(error, __CREATE_BUFFER_ERR);
    if (err != nullptr) {
        *err = error;
    }

    if (!useHostPtr) {
        error = cl::copy(queue, startIterator, endIterator, *this);
        detail::errHandler(error, __CREATE_BUFFER_ERR);
        if (err != nullptr) {
            *err = error;
        }
    }
}

inline cl_int enqueueReadBuffer(
    const Buffer& buffer,
    cl_bool blocking,
    size_type offset,
    size_type size,
    void* ptr,
    const vector<Event>* events = nullptr,
    Event* event = nullptr)
{
    cl_int error;
    CommandQueue queue = CommandQueue::getDefault(&error);

    if (error != CL_SUCCESS) {
        return error;
    }

    return queue.enqueueReadBuffer(buffer, blocking, offset, size, ptr, events, event);
}

inline cl_int enqueueWriteBuffer(
        const Buffer& buffer,
        cl_bool blocking,
        size_type offset,
        size_type size,
        const void* ptr,
        const vector<Event>* events = nullptr,
        Event* event = nullptr)
{
    cl_int error;
    CommandQueue queue = CommandQueue::getDefault(&error);

    if (error != CL_SUCCESS) {
        return error;
    }

    return queue.enqueueWriteBuffer(buffer, blocking, offset, size, ptr, events, event);
}

inline void* enqueueMapBuffer(
        const Buffer& buffer,
        cl_bool blocking,
        cl_map_flags flags,
        size_type offset,
        size_type size,
        const vector<Event>* events = nullptr,
        Event* event = nullptr,
        cl_int* err = nullptr)
{
    cl_int error;
    CommandQueue queue = CommandQueue::getDefault(&error);
    detail::errHandler(error, __ENQUEUE_MAP_BUFFER_ERR);
    if (err != nullptr) {
        *err = error;
    }

    void * result = ::clEnqueueMapBuffer(
            queue(), buffer(), blocking, flags, offset, size,
            (events != nullptr) ? (cl_uint) events->size() : 0,
            (events != nullptr && events->size() > 0) ? (cl_event*) &events->front() : nullptr,
            (cl_event*) event,
            &error);

    detail::errHandler(error, __ENQUEUE_MAP_BUFFER_ERR);
    if (err != nullptr) {
        *err = error;
    }
    return result;
}


#if CL_HPP_TARGET_OPENCL_VERSION >= 200
/**
 * Enqueues to the default queue a command that will allow the host to
 * update a region of a coarse-grained SVM buffer.
 * This variant takes a raw SVM pointer.
 */
template<typename T>
inline cl_int enqueueMapSVM(
    T* ptr,
    cl_bool blocking,
    cl_map_flags flags,
    size_type size,
    const vector<Event>* events,
    Event* event)
{
    cl_int error;
    CommandQueue queue = CommandQueue::getDefault(&error);
    if (error != CL_SUCCESS) {
        return detail::errHandler(error, __ENQUEUE_MAP_SVM_ERR);
    }

    return queue.enqueueMapSVM(
        ptr, blocking, flags, size, events, event);
}

/**
 * Enqueues to the default queue a command that will allow the host to 
 * update a region of a coarse-grained SVM buffer.
 * This variant takes a cl::pointer instance.
 */
template<typename T, class D>
inline cl_int enqueueMapSVM(
    cl::pointer<T, D> &ptr,
    cl_bool blocking,
    cl_map_flags flags,
    size_type size,
    const vector<Event>* events = nullptr,
    Event* event = nullptr)
{
    cl_int error;
    CommandQueue queue = CommandQueue::getDefault(&error);
    if (error != CL_SUCCESS) {
        return detail::errHandler(error, __ENQUEUE_MAP_BUFFER_ERR);
    }

    return queue.enqueueMapSVM(
        ptr, blocking, flags, size, events, event);
}

/**
 * Enqueues to the default queue a command that will allow the host to
 * update a region of a coarse-grained SVM buffer.
 * This variant takes a cl::vector instance.
 */
template<typename T, class Alloc>
inline cl_int enqueueMapSVM(
    cl::vector<T, Alloc> &container,
    cl_bool blocking,
    cl_map_flags flags,
    const vector<Event>* events = nullptr,
    Event* event = nullptr)
{
    cl_int error;
    CommandQueue queue = CommandQueue::getDefault(&error);
    if (error != CL_SUCCESS) {
        return detail::errHandler(error, __ENQUEUE_MAP_SVM_ERR);
    }

    return queue.enqueueMapSVM(
        container, blocking, flags, events, event);
}

#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200

inline cl_int enqueueUnmapMemObject(
    const Memory& memory,
    void* mapped_ptr,
    const vector<Event>* events = nullptr,
    Event* event = nullptr)
{
    cl_int error;
    CommandQueue queue = CommandQueue::getDefault(&error);
    detail::errHandler(error, __ENQUEUE_MAP_BUFFER_ERR);
    if (error != CL_SUCCESS) {
        return error;
    }

    cl_event tmp;
    cl_int err = detail::errHandler(
        ::clEnqueueUnmapMemObject(
        queue(), memory(), mapped_ptr,
        (events != nullptr) ? (cl_uint)events->size() : 0,
        (events != nullptr && events->size() > 0) ? (cl_event*)&events->front() : nullptr,
        (event != nullptr) ? &tmp : nullptr),
        __ENQUEUE_UNMAP_MEM_OBJECT_ERR);

    if (event != nullptr && err == CL_SUCCESS)
        *event = tmp;

    return err;
}

#if CL_HPP_TARGET_OPENCL_VERSION >= 200
/**
 * Enqueues to the default queue a command that will release a coarse-grained 
 * SVM buffer back to the OpenCL runtime.
 * This variant takes a raw SVM pointer.
 */
template<typename T>
inline cl_int enqueueUnmapSVM(
    T* ptr,
    const vector<Event>* events = nullptr,
    Event* event = nullptr)
{
    cl_int error;
    CommandQueue queue = CommandQueue::getDefault(&error);
    if (error != CL_SUCCESS) {
        return detail::errHandler(error, __ENQUEUE_UNMAP_SVM_ERR);
    }

    return detail::errHandler(queue.enqueueUnmapSVM(ptr, events, event), 
        __ENQUEUE_UNMAP_SVM_ERR);

}

/**
 * Enqueues to the default queue a command that will release a coarse-grained 
 * SVM buffer back to the OpenCL runtime.
 * This variant takes a cl::pointer instance.
 */
template<typename T, class D>
inline cl_int enqueueUnmapSVM(
    cl::pointer<T, D> &ptr,
    const vector<Event>* events = nullptr,
    Event* event = nullptr)
{
    cl_int error;
    CommandQueue queue = CommandQueue::getDefault(&error);
    if (error != CL_SUCCESS) {
        return detail::errHandler(error, __ENQUEUE_UNMAP_SVM_ERR);
    }

    return detail::errHandler(queue.enqueueUnmapSVM(ptr, events, event),
        __ENQUEUE_UNMAP_SVM_ERR);
}

/**
 * Enqueues to the default queue a command that will release a coarse-grained 
 * SVM buffer back to the OpenCL runtime.
 * This variant takes a cl::vector instance.
 */
template<typename T, class Alloc>
inline cl_int enqueueUnmapSVM(
    cl::vector<T, Alloc> &container,
    const vector<Event>* events = nullptr,
    Event* event = nullptr)
{
    cl_int error;
    CommandQueue queue = CommandQueue::getDefault(&error);
    if (error != CL_SUCCESS) {
        return detail::errHandler(error, __ENQUEUE_UNMAP_SVM_ERR);
    }

    return detail::errHandler(queue.enqueueUnmapSVM(container, events, event),
        __ENQUEUE_UNMAP_SVM_ERR);
}

#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200

inline cl_int enqueueCopyBuffer(
        const Buffer& src,
        const Buffer& dst,
        size_type src_offset,
        size_type dst_offset,
        size_type size,
        const vector<Event>* events = nullptr,
        Event* event = nullptr)
{
    cl_int error;
    CommandQueue queue = CommandQueue::getDefault(&error);

    if (error != CL_SUCCESS) {
        return error;
    }

    return queue.enqueueCopyBuffer(src, dst, src_offset, dst_offset, size, events, event);
}

/**
 * Blocking copy operation between iterators and a buffer.
 * Host to Device.
 * Uses default command queue.
 */
template< typename IteratorType >
inline cl_int copy( IteratorType startIterator, IteratorType endIterator, cl::Buffer &buffer )
{
    cl_int error;
    CommandQueue queue = CommandQueue::getDefault(&error);
    if (error != CL_SUCCESS)
        return error;

    return cl::copy(queue, startIterator, endIterator, buffer);
}

/**
 * Blocking copy operation between iterators and a buffer.
 * Device to Host.
 * Uses default command queue.
 */
template< typename IteratorType >
inline cl_int copy( const cl::Buffer &buffer, IteratorType startIterator, IteratorType endIterator )
{
    cl_int error;
    CommandQueue queue = CommandQueue::getDefault(&error);
    if (error != CL_SUCCESS)
        return error;

    return cl::copy(queue, buffer, startIterator, endIterator);
}

/**
 * Blocking copy operation between iterators and a buffer.
 * Host to Device.
 * Uses specified queue.
 */
template< typename IteratorType >
inline cl_int copy( const CommandQueue &queue, IteratorType startIterator, IteratorType endIterator, cl::Buffer &buffer )
{
    typedef typename std::iterator_traits<IteratorType>::value_type DataType;
    cl_int error;
    
    size_type length = endIterator-startIterator;
    size_type byteLength = length*sizeof(DataType);

    DataType *pointer = 
        static_cast<DataType*>(queue.enqueueMapBuffer(buffer, CL_TRUE, CL_MAP_WRITE, 0, byteLength, 0, 0, &error));
    // if exceptions enabled, enqueueMapBuffer will throw
    if( error != CL_SUCCESS ) {
        return error;
    }
#if defined(_MSC_VER) && _MSC_VER < 1920
    std::copy(
        startIterator,
        endIterator,
        stdext::checked_array_iterator<DataType*>(
            pointer, length));
#else
    std::copy(startIterator, endIterator, pointer);
#endif // defined(_MSC_VER) && _MSC_VER < 1920
    Event endEvent;
    error = queue.enqueueUnmapMemObject(buffer, pointer, 0, &endEvent);
    // if exceptions enabled, enqueueUnmapMemObject will throw
    if( error != CL_SUCCESS ) { 
        return error;
    }
    endEvent.wait();
    return CL_SUCCESS;
}

/**
 * Blocking copy operation between iterators and a buffer.
 * Device to Host.
 * Uses specified queue.
 */
template< typename IteratorType >
inline cl_int copy( const CommandQueue &queue, const cl::Buffer &buffer, IteratorType startIterator, IteratorType endIterator )
{
    typedef typename std::iterator_traits<IteratorType>::value_type DataType;
    cl_int error;
        
    size_type length = endIterator-startIterator;
    size_type byteLength = length*sizeof(DataType);

    DataType *pointer = 
        static_cast<DataType*>(queue.enqueueMapBuffer(buffer, CL_TRUE, CL_MAP_READ, 0, byteLength, 0, 0, &error));
    // if exceptions enabled, enqueueMapBuffer will throw
    if( error != CL_SUCCESS ) {
        return error;
    }
    std::copy(pointer, pointer + length, startIterator);
    Event endEvent;
    error = queue.enqueueUnmapMemObject(buffer, pointer, 0, &endEvent);
    // if exceptions enabled, enqueueUnmapMemObject will throw
    if( error != CL_SUCCESS ) { 
        return error;
    }
    endEvent.wait();
    return CL_SUCCESS;
}


#if CL_HPP_TARGET_OPENCL_VERSION >= 200
/**
 * Blocking SVM map operation - performs a blocking map underneath.
 */
template<typename T, class Alloc>
inline cl_int mapSVM(cl::vector<T, Alloc> &container)
{
    return enqueueMapSVM(container, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE);
}

/**
* Blocking SVM map operation - performs a blocking map underneath.
*/
template<typename T, class Alloc>
inline cl_int unmapSVM(cl::vector<T, Alloc> &container)
{
    return enqueueUnmapSVM(container);
}

#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200

#if CL_HPP_TARGET_OPENCL_VERSION >= 110
inline cl_int enqueueReadBufferRect(
    const Buffer& buffer,
    cl_bool blocking,
    const array<size_type, 3>& buffer_offset,
    const array<size_type, 3>& host_offset,
    const array<size_type, 3>& region,
    size_type buffer_row_pitch,
    size_type buffer_slice_pitch,
    size_type host_row_pitch,
    size_type host_slice_pitch,
    void *ptr,
    const vector<Event>* events = nullptr,
    Event* event = nullptr)
{
    cl_int error;
    CommandQueue queue = CommandQueue::getDefault(&error);

    if (error != CL_SUCCESS) {
        return error;
    }

    return queue.enqueueReadBufferRect(
        buffer, 
        blocking, 
        buffer_offset, 
        host_offset,
        region,
        buffer_row_pitch,
        buffer_slice_pitch,
        host_row_pitch,
        host_slice_pitch,
        ptr, 
        events, 
        event);
}

inline cl_int enqueueReadBufferRect(
    const Buffer& buffer, 
    cl_bool blocking,
    const array<size_type, 2>& buffer_offset,
    const array<size_type, 2>& host_offset, 
    const array<size_type, 2>& region,
    size_type buffer_row_pitch,
    size_type buffer_slice_pitch,
    size_type host_row_pitch,
    size_type host_slice_pitch,
    void* ptr,
    const vector<Event>* events = nullptr,
    Event* event = nullptr)
{
    return enqueueReadBufferRect(
        buffer,
        blocking,
        { buffer_offset[0], buffer_offset[1], 0 },
        { host_offset[0], host_offset[1], 0 },
        { region[0], region[1], 1 },
        buffer_row_pitch,
        buffer_slice_pitch,
        host_row_pitch,
        host_slice_pitch,
        ptr,
        events,
        event);
}

inline cl_int enqueueWriteBufferRect(
    const Buffer& buffer,
    cl_bool blocking,
    const array<size_type, 3>& buffer_offset,
    const array<size_type, 3>& host_offset,
    const array<size_type, 3>& region,
    size_type buffer_row_pitch,
    size_type buffer_slice_pitch,
    size_type host_row_pitch,
    size_type host_slice_pitch,
    const void *ptr,
    const vector<Event>* events = nullptr,
    Event* event = nullptr)
{
    cl_int error;
    CommandQueue queue = CommandQueue::getDefault(&error);

    if (error != CL_SUCCESS) {
        return error;
    }

    return queue.enqueueWriteBufferRect(
        buffer, 
        blocking, 
        buffer_offset, 
        host_offset,
        region,
        buffer_row_pitch,
        buffer_slice_pitch,
        host_row_pitch,
        host_slice_pitch,
        ptr, 
        events, 
        event);
}

inline cl_int enqueueWriteBufferRect(
    const Buffer& buffer,
    cl_bool blocking,
    const array<size_type, 2>& buffer_offset,
    const array<size_type, 2>& host_offset,
    const array<size_type, 2>& region,
    size_type buffer_row_pitch,
    size_type buffer_slice_pitch,
    size_type host_row_pitch,
    size_type host_slice_pitch,
    const void* ptr,
    const vector<Event>* events = nullptr,
    Event* event = nullptr)
{
    return enqueueWriteBufferRect(
        buffer, 
        blocking,
        { buffer_offset[0], buffer_offset[1], 0 },
        { host_offset[0], host_offset[1], 0 },
        { region[0], region[1], 1 }, 
        buffer_row_pitch,
        buffer_slice_pitch,
        host_row_pitch,
        host_slice_pitch,
        ptr,
        events,
        event);
}

inline cl_int enqueueCopyBufferRect(
    const Buffer& src,
    const Buffer& dst,
    const array<size_type, 3>& src_origin,
    const array<size_type, 3>& dst_origin,
    const array<size_type, 3>& region,
    size_type src_row_pitch,
    size_type src_slice_pitch,
    size_type dst_row_pitch,
    size_type dst_slice_pitch,
    const vector<Event>* events = nullptr,
    Event* event = nullptr)
{
    cl_int error;
    CommandQueue queue = CommandQueue::getDefault(&error);

    if (error != CL_SUCCESS) {
        return error;
    }

    return queue.enqueueCopyBufferRect(
        src,
        dst,
        src_origin,
        dst_origin,
        region,
        src_row_pitch,
        src_slice_pitch,
        dst_row_pitch,
        dst_slice_pitch,
        events, 
        event);
}

inline cl_int enqueueCopyBufferRect(
    const Buffer& src,
    const Buffer& dst,
    const array<size_type, 2>& src_origin,
    const array<size_type, 2>& dst_origin,
    const array<size_type, 2>& region,
    size_type src_row_pitch,
    size_type src_slice_pitch,
    size_type dst_row_pitch,
    size_type dst_slice_pitch,
    const vector<Event>* events = nullptr,
    Event* event = nullptr)
{
    return enqueueCopyBufferRect(
        src,
        dst, 
        { src_origin[0], src_origin[1], 0 },
        { dst_origin[0], dst_origin[1], 0 },
        { region[0], region[1], 1 }, 
        src_row_pitch,
        src_slice_pitch,
        dst_row_pitch,
        dst_slice_pitch,
        events,
        event);
}
#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110

inline cl_int enqueueReadImage(
    const Image& image,
    cl_bool blocking,
    const array<size_type, 3>& origin,
    const array<size_type, 3>& region,
    size_type row_pitch,
    size_type slice_pitch,
    void* ptr,
    const vector<Event>* events = nullptr,
    Event* event = nullptr) 
{
    cl_int error;
    CommandQueue queue = CommandQueue::getDefault(&error);

    if (error != CL_SUCCESS) {
        return error;
    }

    return queue.enqueueReadImage(
        image,
        blocking,
        origin,
        region,
        row_pitch,
        slice_pitch,
        ptr,
        events, 
        event);
}

inline cl_int enqueueReadImage(
    const Image& image, 
    cl_bool blocking,
    const array<size_type, 2>& origin,
    const array<size_type, 2>& region,
    size_type row_pitch,
    size_type slice_pitch,
    void* ptr, 
    const vector<Event>* events = nullptr,
    Event* event = nullptr)
{
    return enqueueReadImage(
        image,
        blocking, 
        { origin[0], origin[1], 0 },
        { region[0], region[1], 1 },
        row_pitch,
        slice_pitch,
        ptr,
        events,
        event);
}

inline cl_int enqueueWriteImage(
    const Image& image,
    cl_bool blocking,
    const array<size_type, 3>& origin,
    const array<size_type, 3>& region,
    size_type row_pitch,
    size_type slice_pitch,
    const void* ptr,
    const vector<Event>* events = nullptr,
    Event* event = nullptr)
{
    cl_int error;
    CommandQueue queue = CommandQueue::getDefault(&error);

    if (error != CL_SUCCESS) {
        return error;
    }

    return queue.enqueueWriteImage(
        image,
        blocking,
        origin,
        region,
        row_pitch,
        slice_pitch,
        ptr,
        events, 
        event);
}

inline cl_int enqueueWriteImage(
    const Image& image, 
    cl_bool blocking,
    const array<size_type, 2>& origin,
    const array<size_type, 2>& region,
    size_type row_pitch, 
    size_type slice_pitch,
    const void* ptr,
    const vector<Event>* events = nullptr,
    Event* event = nullptr)
{
    return enqueueWriteImage(
        image, 
        blocking, 
        { origin[0], origin[1], 0 },
        { region[0], region[1], 1 }, 
        row_pitch,
        slice_pitch,
        ptr,
        events,
        event);    
}

inline cl_int enqueueCopyImage(
    const Image& src,
    const Image& dst,
    const array<size_type, 3>& src_origin,
    const array<size_type, 3>& dst_origin,
    const array<size_type, 3>& region,
    const vector<Event>* events = nullptr,
    Event* event = nullptr)
{
    cl_int error;
    CommandQueue queue = CommandQueue::getDefault(&error);

    if (error != CL_SUCCESS) {
        return error;
    }

    return queue.enqueueCopyImage(
        src,
        dst,
        src_origin,
        dst_origin,
        region,
        events,
        event);
}

inline cl_int enqueueCopyImage(
    const Image& src, 
    const Image& dst,
    const array<size_type, 2>& src_origin,
    const array<size_type, 2>& dst_origin,
    const array<size_type, 2>& region,
    const vector<Event>* events = nullptr,
    Event* event = nullptr)
{
    return enqueueCopyImage(
        src, 
        dst,
        { src_origin[0], src_origin[1], 0 },
        { dst_origin[0], dst_origin[1], 0 },
        { region[0], region[1], 1 },
        events,
        event);
}

inline cl_int enqueueCopyImageToBuffer(
    const Image& src,
    const Buffer& dst,
    const array<size_type, 3>& src_origin,
    const array<size_type, 3>& region,
    size_type dst_offset,
    const vector<Event>* events = nullptr,
    Event* event = nullptr)
{
    cl_int error;
    CommandQueue queue = CommandQueue::getDefault(&error);

    if (error != CL_SUCCESS) {
        return error;
    }

    return queue.enqueueCopyImageToBuffer(
        src,
        dst,
        src_origin,
        region,
        dst_offset,
        events,
        event);
}

inline cl_int enqueueCopyImageToBuffer(
    const Image& src, 
    const Buffer& dst,
    const array<size_type, 2>& src_origin,
    const array<size_type, 2>& region,
    size_type dst_offset,
    const vector<Event>* events = nullptr,
    Event* event = nullptr)
{
    return enqueueCopyImageToBuffer(
        src,
        dst,
        { src_origin[0], src_origin[1], 0 },
        { region[0], region[1], 1 },
        dst_offset,
        events,
        event);
}

inline cl_int enqueueCopyBufferToImage(
    const Buffer& src,
    const Image& dst,
    size_type src_offset,
    const array<size_type, 3>& dst_origin,
    const array<size_type, 3>& region,
    const vector<Event>* events = nullptr,
    Event* event = nullptr)
{
    cl_int error;
    CommandQueue queue = CommandQueue::getDefault(&error);

    if (error != CL_SUCCESS) {
        return error;
    }

    return queue.enqueueCopyBufferToImage(
        src,
        dst,
        src_offset,
        dst_origin,
        region,
        events,
        event);
}

inline cl_int enqueueCopyBufferToImage(
    const Buffer& src,
    const Image& dst,
    size_type src_offset,
    const array<size_type, 2>& dst_origin,
    const array<size_type, 2>& region,
    const vector<Event>* events = nullptr,
    Event* event = nullptr)
{
    cl_int error;
    CommandQueue queue = CommandQueue::getDefault(&error);

    if (error != CL_SUCCESS) {
        return error;
    }

    return enqueueCopyBufferToImage(
        src,
        dst,
        src_offset,
        { dst_origin[0], dst_origin[1], 0 },
        { region[0], region[1], 1 },
        events,
        event);
}

inline cl_int flush(void)
{
    cl_int error;
    CommandQueue queue = CommandQueue::getDefault(&error);

    if (error != CL_SUCCESS) {
        return error;
    }

    return queue.flush();
}

inline cl_int finish(void)
{
    cl_int error;
    CommandQueue queue = CommandQueue::getDefault(&error);

    if (error != CL_SUCCESS) {
        return error;
    } 


    return queue.finish();
}

class EnqueueArgs
{
private:
    CommandQueue queue_;
    const NDRange offset_;
    const NDRange global_;
    const NDRange local_;
    vector<Event> events_;

    template<typename... Ts>
    friend class KernelFunctor;

public:
    EnqueueArgs(NDRange global) : 
      queue_(CommandQueue::getDefault()),
      offset_(NullRange), 
      global_(global),
      local_(NullRange)
    {

    }

    EnqueueArgs(NDRange global, NDRange local) : 
      queue_(CommandQueue::getDefault()),
      offset_(NullRange), 
      global_(global),
      local_(local)
    {

    }

    EnqueueArgs(NDRange offset, NDRange global, NDRange local) : 
      queue_(CommandQueue::getDefault()),
      offset_(offset), 
      global_(global),
      local_(local)
    {

    }

    EnqueueArgs(Event e, NDRange global) : 
      queue_(CommandQueue::getDefault()),
      offset_(NullRange), 
      global_(global),
      local_(NullRange)
    {
        events_.push_back(e);
    }

    EnqueueArgs(Event e, NDRange global, NDRange local) : 
      queue_(CommandQueue::getDefault()),
      offset_(NullRange), 
      global_(global),
      local_(local)
    {
        events_.push_back(e);
    }

    EnqueueArgs(Event e, NDRange offset, NDRange global, NDRange local) : 
      queue_(CommandQueue::getDefault()),
      offset_(offset), 
      global_(global),
      local_(local)
    {
        events_.push_back(e);
    }

    EnqueueArgs(const vector<Event> &events, NDRange global) : 
      queue_(CommandQueue::getDefault()),
      offset_(NullRange), 
      global_(global),
      local_(NullRange),
      events_(events)
    {

    }

    EnqueueArgs(const vector<Event> &events, NDRange global, NDRange local) : 
      queue_(CommandQueue::getDefault()),
      offset_(NullRange), 
      global_(global),
      local_(local),
      events_(events)
    {

    }

    EnqueueArgs(const vector<Event> &events, NDRange offset, NDRange global, NDRange local) : 
      queue_(CommandQueue::getDefault()),
      offset_(offset), 
      global_(global),
      local_(local),
      events_(events)
    {

    }

    EnqueueArgs(CommandQueue &queue, NDRange global) : 
      queue_(queue),
      offset_(NullRange), 
      global_(global),
      local_(NullRange)
    {

    }

    EnqueueArgs(CommandQueue &queue, NDRange global, NDRange local) : 
      queue_(queue),
      offset_(NullRange), 
      global_(global),
      local_(local)
    {

    }

    EnqueueArgs(CommandQueue &queue, NDRange offset, NDRange global, NDRange local) : 
      queue_(queue),
      offset_(offset), 
      global_(global),
      local_(local)
    {

    }

    EnqueueArgs(CommandQueue &queue, Event e, NDRange global) : 
      queue_(queue),
      offset_(NullRange), 
      global_(global),
      local_(NullRange)
    {
        events_.push_back(e);
    }

    EnqueueArgs(CommandQueue &queue, Event e, NDRange global, NDRange local) : 
      queue_(queue),
      offset_(NullRange), 
      global_(global),
      local_(local)
    {
        events_.push_back(e);
    }

    EnqueueArgs(CommandQueue &queue, Event e, NDRange offset, NDRange global, NDRange local) : 
      queue_(queue),
      offset_(offset), 
      global_(global),
      local_(local)
    {
        events_.push_back(e);
    }

    EnqueueArgs(CommandQueue &queue, const vector<Event> &events, NDRange global) : 
      queue_(queue),
      offset_(NullRange), 
      global_(global),
      local_(NullRange),
      events_(events)
    {

    }

    EnqueueArgs(CommandQueue &queue, const vector<Event> &events, NDRange global, NDRange local) : 
      queue_(queue),
      offset_(NullRange), 
      global_(global),
      local_(local),
      events_(events)
    {

    }

    EnqueueArgs(CommandQueue &queue, const vector<Event> &events, NDRange offset, NDRange global, NDRange local) : 
      queue_(queue),
      offset_(offset), 
      global_(global),
      local_(local),
      events_(events)
    {

    }
};


//----------------------------------------------------------------------------------------------


/**
 * Type safe kernel functor.
 * 
 */
template<typename... Ts>
class KernelFunctor
{
private:
    Kernel kernel_;

    template<int index, typename T0, typename... T1s>
    void setArgs(T0&& t0, T1s&&... t1s)
    {
        kernel_.setArg(index, t0);
        setArgs<index + 1, T1s...>(std::forward<T1s>(t1s)...);
    }

    template<int index, typename T0>
    void setArgs(T0&& t0)
    {
        kernel_.setArg(index, t0);
    }

    template<int index>
    void setArgs()
    {
    }


public:
    KernelFunctor(Kernel kernel) : kernel_(kernel)
    {}

    KernelFunctor(
        const Program& program,
        const string name,
        cl_int * err = nullptr) :
        kernel_(program, name.c_str(), err)
    {}

    //! \brief Return type of the functor
    typedef Event result_type;

    /**
     * Enqueue kernel.
     * @param args Launch parameters of the kernel.
     * @param t0... List of kernel arguments based on the template type of the functor.
     */
    Event operator() (
        const EnqueueArgs& args,
        Ts... ts)
    {
        Event event;
        setArgs<0>(std::forward<Ts>(ts)...);
        
        args.queue_.enqueueNDRangeKernel(
            kernel_,
            args.offset_,
            args.global_,
            args.local_,
            &args.events_,
            &event);

        return event;
    }

    /**
    * Enqueue kernel with support for error code.
    * @param args Launch parameters of the kernel.
    * @param t0... List of kernel arguments based on the template type of the functor.
    * @param error Out parameter returning the error code from the execution.
    */
    Event operator() (
        const EnqueueArgs& args,
        Ts... ts,
        cl_int &error)
    {
        Event event;
        setArgs<0>(std::forward<Ts>(ts)...);

        error = args.queue_.enqueueNDRangeKernel(
            kernel_,
            args.offset_,
            args.global_,
            args.local_,
            &args.events_,
            &event);
        
        return event;
    }

#if CL_HPP_TARGET_OPENCL_VERSION >= 200
    cl_int setSVMPointers(const vector<void*> &pointerList)
    {
        return kernel_.setSVMPointers(pointerList);
    }

    template<typename T0, typename... T1s>
    cl_int setSVMPointers(const T0 &t0, T1s &... ts)
    {
        return kernel_.setSVMPointers(t0, ts...);
    }
#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200

    Kernel getKernel()
    {
        return kernel_;
    }
};

namespace compatibility {
    /**
     * Backward compatibility class to ensure that cl.hpp code works with opencl.hpp.
     * Please use KernelFunctor directly.
     */
    template<typename... Ts>
    struct make_kernel
    {
        typedef KernelFunctor<Ts...> FunctorType;

        FunctorType functor_;

        make_kernel(
            const Program& program,
            const string name,
            cl_int * err = nullptr) :
            functor_(FunctorType(program, name, err))
        {}

        make_kernel(
            const Kernel kernel) :
            functor_(FunctorType(kernel))
        {}

        //! \brief Return type of the functor
        typedef Event result_type;

        //! \brief Function signature of kernel functor with no event dependency.
        typedef Event type_(
            const EnqueueArgs&,
            Ts...);

        Event operator()(
            const EnqueueArgs& enqueueArgs,
            Ts... args)
        {
            return functor_(
                enqueueArgs, args...);
        }
    };
} // namespace compatibility

#ifdef cl_khr_semaphore

#ifdef cl_khr_external_semaphore
enum ExternalSemaphoreType : cl_external_semaphore_handle_type_khr
{
    None = 0,
#ifdef cl_khr_external_semaphore_opaque_fd
    OpaqueFd = CL_SEMAPHORE_HANDLE_OPAQUE_FD_KHR,
#endif // cl_khr_external_semaphore_opaque_fd
#ifdef cl_khr_external_semaphore_sync_fd
    SyncFd = CL_SEMAPHORE_HANDLE_SYNC_FD_KHR,
#endif // cl_khr_external_semaphore_sync_fd
#ifdef cl_khr_external_semaphore_win32
    OpaqueWin32 = CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KHR,
    OpaqueWin32Kmt = CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KMT_KHR,
#endif // cl_khr_external_semaphore_win32
};
#endif // cl_khr_external_semaphore

class Semaphore : public detail::Wrapper<cl_semaphore_khr>
{
public:
    Semaphore() : detail::Wrapper<cl_type>() {}
    Semaphore(
        const Context &context,
        const vector<cl_semaphore_properties_khr>& sema_props,
        cl_int *err = nullptr) 
    {
        /* initialization of addresses to extension functions (it is done only once) */
        std::call_once(ext_init_, initExtensions, context);

        cl_int error = CL_INVALID_OPERATION;

        if (pfn_clCreateSemaphoreWithPropertiesKHR)
        {
            object_ = pfn_clCreateSemaphoreWithPropertiesKHR(
                context(),
                sema_props.data(),
                &error);
        }
          
        detail::errHandler(error, __CREATE_SEMAPHORE_KHR_WITH_PROPERTIES_ERR);

        if (err != nullptr) {
            *err = error;
        }
    }
    Semaphore(
        const vector<cl_semaphore_properties_khr>& sema_props,
        cl_int* err = nullptr):Semaphore(Context::getDefault(err), sema_props, err) {}
    
    explicit Semaphore(const cl_semaphore_khr& semaphore, bool retainObject = false) :
        detail::Wrapper<cl_type>(semaphore, retainObject) {}
    Semaphore& operator = (const cl_semaphore_khr& rhs) {
        detail::Wrapper<cl_type>::operator=(rhs);
        return *this;
    }
    template <typename T>
    cl_int getInfo(cl_semaphore_info_khr name, T* param) const
    {
        if (pfn_clGetSemaphoreInfoKHR == nullptr) {
            return detail::errHandler(CL_INVALID_OPERATION,
                                      __GET_SEMAPHORE_KHR_INFO_ERR);
        }

        return detail::errHandler(
            detail::getInfo(pfn_clGetSemaphoreInfoKHR, object_, name, param),
            __GET_SEMAPHORE_KHR_INFO_ERR);
    }
    template <cl_semaphore_info_khr name> typename
    detail::param_traits<detail::cl_semaphore_info_khr, name>::param_type
    getInfo(cl_int* err = nullptr) const
    {
        typename detail::param_traits<
            detail::cl_semaphore_info_khr, name>::param_type param;
        cl_int result = getInfo(name, &param);
        if (err != nullptr) {
            *err = result;        
        }
        return param;      
    }

#ifdef cl_khr_external_semaphore
    template <typename T>
    cl_int getHandleForTypeKHR(
        const Device& device, cl_external_semaphore_handle_type_khr name, T* param) const
    {
        if (pfn_clGetSemaphoreHandleForTypeKHR == nullptr) {
            return detail::errHandler(CL_INVALID_OPERATION,
                                      __GET_SEMAPHORE_HANDLE_FOR_TYPE_KHR_ERR);
        }

        return detail::errHandler(
            detail::getInfo(
                pfn_clGetSemaphoreHandleForTypeKHR, object_, device(), name, param),
                __GET_SEMAPHORE_HANDLE_FOR_TYPE_KHR_ERR);
    }

    template <cl_external_semaphore_handle_type_khr type> typename
    detail::param_traits<detail::cl_external_semaphore_handle_type_khr, type>::param_type
        getHandleForTypeKHR(const Device& device, cl_int* err = nullptr) const
    {
        typename detail::param_traits<
        detail::cl_external_semaphore_handle_type_khr, type>::param_type param;
        cl_int result = getHandleForTypeKHR(device, type, &param);
        if (err != nullptr) {
            *err = result;
        }
        return param;
    }
#endif // cl_khr_external_semaphore

    cl_int retain()
    { 
        if (pfn_clRetainSemaphoreKHR == nullptr) {
            return detail::errHandler(CL_INVALID_OPERATION,
                                      __RETAIN_SEMAPHORE_KHR_ERR);
        }
        return pfn_clRetainSemaphoreKHR(object_);
    }

    cl_int release()
    { 
        if (pfn_clReleaseSemaphoreKHR == nullptr) {
            return detail::errHandler(CL_INVALID_OPERATION,
                                      __RELEASE_SEMAPHORE_KHR_ERR);
        }
        return pfn_clReleaseSemaphoreKHR(object_);
    }

private:
    static std::once_flag ext_init_;

    static void initExtensions(const Context& context)
    {
#if CL_HPP_TARGET_OPENCL_VERSION >= 120
        Device device = context.getInfo<CL_CONTEXT_DEVICES>().at(0);
        cl_platform_id platform = device.getInfo<CL_DEVICE_PLATFORM>()();
        CL_HPP_INIT_CL_EXT_FCN_PTR_PLATFORM_(platform, clCreateSemaphoreWithPropertiesKHR);
        CL_HPP_INIT_CL_EXT_FCN_PTR_PLATFORM_(platform, clReleaseSemaphoreKHR);
        CL_HPP_INIT_CL_EXT_FCN_PTR_PLATFORM_(platform, clRetainSemaphoreKHR);
        CL_HPP_INIT_CL_EXT_FCN_PTR_PLATFORM_(platform, clEnqueueWaitSemaphoresKHR);
        CL_HPP_INIT_CL_EXT_FCN_PTR_PLATFORM_(platform, clEnqueueSignalSemaphoresKHR);
        CL_HPP_INIT_CL_EXT_FCN_PTR_PLATFORM_(platform, clGetSemaphoreInfoKHR);
#ifdef cl_khr_external_semaphore
        CL_HPP_INIT_CL_EXT_FCN_PTR_PLATFORM_(platform, clGetSemaphoreHandleForTypeKHR);
#endif // cl_khr_external_semaphore

#else
        CL_HPP_INIT_CL_EXT_FCN_PTR_(clCreateSemaphoreWithPropertiesKHR);
        CL_HPP_INIT_CL_EXT_FCN_PTR_(clReleaseSemaphoreKHR);
        CL_HPP_INIT_CL_EXT_FCN_PTR_(clRetainSemaphoreKHR);
        CL_HPP_INIT_CL_EXT_FCN_PTR_(clEnqueueWaitSemaphoresKHR);
        CL_HPP_INIT_CL_EXT_FCN_PTR_(clEnqueueSignalSemaphoresKHR);
        CL_HPP_INIT_CL_EXT_FCN_PTR_(clGetSemaphoreInfoKHR);
#ifdef cl_khr_external_semaphore
        CL_HPP_INIT_CL_EXT_FCN_PTR_(clGetSemaphoreHandleForTypeKHR);
#endif // cl_khr_external_semaphore

#endif
        if ((pfn_clCreateSemaphoreWithPropertiesKHR == nullptr) &&
            (pfn_clReleaseSemaphoreKHR              == nullptr) &&
            (pfn_clRetainSemaphoreKHR               == nullptr) &&
            (pfn_clEnqueueWaitSemaphoresKHR         == nullptr) &&
            (pfn_clEnqueueSignalSemaphoresKHR       == nullptr) &&
#ifdef cl_khr_external_semaphore
            (pfn_clGetSemaphoreHandleForTypeKHR     == nullptr) &&
#endif // cl_khr_external_semaphore
            (pfn_clGetSemaphoreInfoKHR              == nullptr))
        {
            detail::errHandler(CL_INVALID_VALUE, __CREATE_SEMAPHORE_KHR_WITH_PROPERTIES_ERR);
        }
    }

};

CL_HPP_DEFINE_STATIC_MEMBER_ std::once_flag Semaphore::ext_init_;

inline cl_int CommandQueue::enqueueWaitSemaphores(
    const vector<Semaphore> &sema_objects,
    const vector<cl_semaphore_payload_khr> &sema_payloads,
    const vector<Event>* events_wait_list,
    Event *event) const
{
    cl_event tmp;
    cl_int err = CL_INVALID_OPERATION;

    if (pfn_clEnqueueWaitSemaphoresKHR != nullptr) {
        err = pfn_clEnqueueWaitSemaphoresKHR(
                object_,
                (cl_uint)sema_objects.size(),
                (const cl_semaphore_khr *) &sema_objects.front(),
                (sema_payloads.size() > 0) ? &sema_payloads.front() : nullptr,
                (events_wait_list != nullptr) ? (cl_uint) events_wait_list->size() : 0,
                (events_wait_list != nullptr && events_wait_list->size() > 0) ? (cl_event*) &events_wait_list->front() : nullptr,
                (event != nullptr) ? &tmp : nullptr);
    }

    detail::errHandler(err, __ENQUEUE_WAIT_SEMAPHORE_KHR_ERR);

    if (event != nullptr && err == CL_SUCCESS)
        *event = tmp;

    return err;
}

inline cl_int CommandQueue::enqueueSignalSemaphores(
    const vector<Semaphore> &sema_objects,
    const vector<cl_semaphore_payload_khr>& sema_payloads,
    const vector<Event>* events_wait_list,
    Event* event)
{
    cl_event tmp;
    cl_int err = CL_INVALID_OPERATION;

    if (pfn_clEnqueueSignalSemaphoresKHR != nullptr) {
        err = pfn_clEnqueueSignalSemaphoresKHR(
                object_,
                (cl_uint)sema_objects.size(),
                (const cl_semaphore_khr*) &sema_objects.front(),
                (sema_payloads.size() > 0) ? &sema_payloads.front() : nullptr,
                (events_wait_list != nullptr) ? (cl_uint) events_wait_list->size() : 0,
                (events_wait_list != nullptr && events_wait_list->size() > 0) ? (cl_event*) &events_wait_list->front() : nullptr,
                (event != nullptr) ? &tmp : nullptr);
    }

    detail::errHandler(err, __ENQUEUE_SIGNAL_SEMAPHORE_KHR_ERR);

    if (event != nullptr && err == CL_SUCCESS)
        *event = tmp;

    return err;
}

#endif // cl_khr_semaphore

#if defined(cl_khr_command_buffer)
/*! \class CommandBufferKhr
 * \brief CommandBufferKhr interface for cl_command_buffer_khr.
 */
class CommandBufferKhr : public detail::Wrapper<cl_command_buffer_khr>
{
public:
    //! \brief Default constructor - initializes to nullptr.
    CommandBufferKhr() : detail::Wrapper<cl_type>() { }

    explicit CommandBufferKhr(const vector<CommandQueue> &queues,
        cl_command_buffer_properties_khr properties = 0,
        cl_int* errcode_ret = nullptr)
    {
        cl_command_buffer_properties_khr command_buffer_properties[] = {
            CL_COMMAND_BUFFER_FLAGS_KHR, properties, 0
        };

        /* initialization of addresses to extension functions (it is done only once) */
        std::call_once(ext_init_, [&] { initExtensions(queues[0].getInfo<CL_QUEUE_DEVICE>()); });
        cl_int error = CL_INVALID_OPERATION;

        static_assert(sizeof(cl::CommandQueue) == sizeof(cl_command_queue),
            "Size of cl::CommandQueue must be equal to size of cl_command_queue");

        if (pfn_clCreateCommandBufferKHR)
        {
            object_ = pfn_clCreateCommandBufferKHR((cl_uint) queues.size(),
                (cl_command_queue *) &queues.front(),
                command_buffer_properties,
                &error);
        }

        detail::errHandler(error, __CREATE_COMMAND_BUFFER_KHR_ERR);
        if (errcode_ret != nullptr) {
            *errcode_ret = error;
        }
    }

    explicit CommandBufferKhr(const cl_command_buffer_khr& commandBufferKhr, bool retainObject = false) :
        detail::Wrapper<cl_type>(commandBufferKhr, retainObject) { }

    CommandBufferKhr& operator=(const cl_command_buffer_khr& rhs)
    {
        detail::Wrapper<cl_type>::operator=(rhs);
        return *this;
    }

    template <typename T>
    cl_int getInfo(cl_command_buffer_info_khr name, T* param) const
    {
        if (pfn_clGetCommandBufferInfoKHR == nullptr) {
            return detail::errHandler(CL_INVALID_OPERATION,
                    __GET_COMMAND_BUFFER_INFO_KHR_ERR);
        }
        return detail::errHandler(
            detail::getInfo(pfn_clGetCommandBufferInfoKHR, object_, name, param),
                __GET_COMMAND_BUFFER_INFO_KHR_ERR);
    }

    template <cl_command_buffer_info_khr name> typename
        detail::param_traits<detail::cl_command_buffer_info_khr, name>::param_type
        getInfo(cl_int* err = nullptr) const
    {
        typename detail::param_traits<
            detail::cl_command_buffer_info_khr, name>::param_type param;
        cl_int result = getInfo(name, &param);
        if (err != nullptr) {
            *err = result;
        }
        return param;
    }

    cl_int finalizeCommandBuffer() const
    {
        return detail::errHandler(::clFinalizeCommandBufferKHR(object_), __FINALIZE_COMMAND_BUFFER_KHR_ERR);
    }

    cl_int enqueueCommandBuffer(vector<CommandQueue> &queues,
        const vector<Event>* events = nullptr,
        Event* event = nullptr)
    {
        if (pfn_clEnqueueCommandBufferKHR == nullptr) {
            return detail::errHandler(CL_INVALID_OPERATION,
                    __ENQUEUE_COMMAND_BUFFER_KHR_ERR);
        }

         static_assert(sizeof(cl::CommandQueue) == sizeof(cl_command_queue),
            "Size of cl::CommandQueue must be equal to size of cl_command_queue");

        return detail::errHandler(pfn_clEnqueueCommandBufferKHR((cl_uint) queues.size(),
                (cl_command_queue *) &queues.front(),
                object_,
                (events != nullptr) ? (cl_uint) events->size() : 0,
                (events != nullptr && events->size() > 0) ? (cl_event*) &events->front() : nullptr,
                (cl_event*) event),
                __ENQUEUE_COMMAND_BUFFER_KHR_ERR);
    }

    cl_int commandBarrierWithWaitList(const vector<cl_sync_point_khr>* sync_points_vec = nullptr,
        cl_sync_point_khr* sync_point = nullptr,
        MutableCommandKhr* mutable_handle = nullptr,
        const CommandQueue* command_queue = nullptr)
    {
        if (pfn_clCommandBarrierWithWaitListKHR == nullptr) {
            return detail::errHandler(CL_INVALID_OPERATION,
                    __COMMAND_BARRIER_WITH_WAIT_LIST_KHR_ERR);
        }

        cl_sync_point_khr tmp_sync_point;
        cl_int error = detail::errHandler(
            pfn_clCommandBarrierWithWaitListKHR(object_,
                (command_queue != nullptr) ? (*command_queue)() : nullptr,
#if CL_KHR_COMMAND_BUFFER_EXTENSION_VERSION > CL_MAKE_VERSION(0, 9, 4)
                nullptr, // Properties
#endif
                (sync_points_vec != nullptr) ? (cl_uint) sync_points_vec->size() : 0,
                (sync_points_vec != nullptr && sync_points_vec->size() > 0) ? &sync_points_vec->front() : nullptr,
                (sync_point != nullptr) ? &tmp_sync_point : nullptr,
                (cl_mutable_command_khr*) mutable_handle),
            __COMMAND_BARRIER_WITH_WAIT_LIST_KHR_ERR);

        if (sync_point != nullptr && error == CL_SUCCESS)
            *sync_point = tmp_sync_point;

        return error;
    }

    cl_int commandCopyBuffer(const Buffer& src,
        const Buffer& dst,
        size_type src_offset,
        size_type dst_offset,
        size_type size,
        const vector<cl_sync_point_khr>* sync_points_vec = nullptr,
        cl_sync_point_khr* sync_point = nullptr,
        MutableCommandKhr* mutable_handle = nullptr,
        const CommandQueue* command_queue = nullptr)
    {
        if (pfn_clCommandCopyBufferKHR == nullptr) {
            return detail::errHandler(CL_INVALID_OPERATION,
                    __COMMAND_COPY_BUFFER_KHR_ERR);
        }

        cl_sync_point_khr tmp_sync_point;
        cl_int error = detail::errHandler(
            pfn_clCommandCopyBufferKHR(object_,
                (command_queue != nullptr) ? (*command_queue)() : nullptr,
#if CL_KHR_COMMAND_BUFFER_EXTENSION_VERSION > CL_MAKE_VERSION(0, 9, 4)
                nullptr, // Properties
#endif
                src(),
                dst(),
                src_offset,
                dst_offset,
                size,
                (sync_points_vec != nullptr) ? (cl_uint) sync_points_vec->size() : 0,
                (sync_points_vec != nullptr && sync_points_vec->size() > 0) ? &sync_points_vec->front() : nullptr,
                (sync_point != nullptr) ? &tmp_sync_point : nullptr,
                (cl_mutable_command_khr*) mutable_handle),
            __COMMAND_COPY_BUFFER_KHR_ERR);

        if (sync_point != nullptr && error == CL_SUCCESS)
            *sync_point = tmp_sync_point;

        return error;
    }

    cl_int commandCopyBufferRect(const Buffer& src,
        const Buffer& dst,
        const array<size_type, 3>& src_origin,
        const array<size_type, 3>& dst_origin,
        const array<size_type, 3>& region,
        size_type src_row_pitch,
        size_type src_slice_pitch,
        size_type dst_row_pitch,
        size_type dst_slice_pitch,
        const vector<cl_sync_point_khr>* sync_points_vec = nullptr,
        cl_sync_point_khr* sync_point = nullptr,
        MutableCommandKhr* mutable_handle = nullptr,
        const CommandQueue* command_queue = nullptr)
    {
        if (pfn_clCommandCopyBufferRectKHR == nullptr) {
            return detail::errHandler(CL_INVALID_OPERATION,
                    __COMMAND_COPY_BUFFER_RECT_KHR_ERR);
        }

        cl_sync_point_khr tmp_sync_point;
        cl_int error = detail::errHandler(
            pfn_clCommandCopyBufferRectKHR(object_,
                (command_queue != nullptr) ? (*command_queue)() : nullptr,
#if CL_KHR_COMMAND_BUFFER_EXTENSION_VERSION > CL_MAKE_VERSION(0, 9, 4)
                nullptr, // Properties
#endif
                src(),
                dst(),
                src_origin.data(),
                dst_origin.data(),
                region.data(),
                src_row_pitch,
                src_slice_pitch,
                dst_row_pitch,
                dst_slice_pitch,
                (sync_points_vec != nullptr) ? (cl_uint) sync_points_vec->size() : 0,
                (sync_points_vec != nullptr && sync_points_vec->size() > 0) ? &sync_points_vec->front() : nullptr,
                (sync_point != nullptr) ? &tmp_sync_point : nullptr,
                (cl_mutable_command_khr*) mutable_handle),
            __COMMAND_COPY_BUFFER_RECT_KHR_ERR);

        if (sync_point != nullptr && error == CL_SUCCESS)
            *sync_point = tmp_sync_point;

        return error;
    }

    cl_int commandCopyBufferToImage(const Buffer& src,
        const Image& dst,
        size_type src_offset,
        const array<size_type, 3>& dst_origin,
        const array<size_type, 3>& region,
        const vector<cl_sync_point_khr>* sync_points_vec = nullptr,
        cl_sync_point_khr* sync_point = nullptr,
        MutableCommandKhr* mutable_handle = nullptr,
        const CommandQueue* command_queue = nullptr)
    {
        if (pfn_clCommandCopyBufferToImageKHR == nullptr) {
            return detail::errHandler(CL_INVALID_OPERATION,
                    __COMMAND_COPY_BUFFER_TO_IMAGE_KHR_ERR);
        }

        cl_sync_point_khr tmp_sync_point;
        cl_int error = detail::errHandler(
            pfn_clCommandCopyBufferToImageKHR(object_,
                (command_queue != nullptr) ? (*command_queue)() : nullptr,
#if CL_KHR_COMMAND_BUFFER_EXTENSION_VERSION > CL_MAKE_VERSION(0, 9, 4)
                nullptr, // Properties
#endif
                src(),
                dst(),
                src_offset,
                dst_origin.data(),
                region.data(),
                (sync_points_vec != nullptr) ? (cl_uint) sync_points_vec->size() : 0,
                (sync_points_vec != nullptr && sync_points_vec->size() > 0) ? &sync_points_vec->front() : nullptr,
                (sync_point != nullptr) ? &tmp_sync_point : nullptr,
                (cl_mutable_command_khr*) mutable_handle),
            __COMMAND_COPY_BUFFER_TO_IMAGE_KHR_ERR);

        if (sync_point != nullptr && error == CL_SUCCESS)
            *sync_point = tmp_sync_point;

        return error;
    }

    cl_int commandCopyImage(const Image& src,
        const Image& dst,
        const array<size_type, 3>& src_origin,
        const array<size_type, 3>& dst_origin,
        const array<size_type, 3>& region,
        const vector<cl_sync_point_khr>* sync_points_vec = nullptr,
        cl_sync_point_khr* sync_point = nullptr,
        MutableCommandKhr* mutable_handle = nullptr,
        const CommandQueue* command_queue = nullptr)
    {
        if (pfn_clCommandCopyImageKHR == nullptr) {
            return detail::errHandler(CL_INVALID_OPERATION,
                    __COMMAND_COPY_IMAGE_KHR_ERR);
        }

        cl_sync_point_khr tmp_sync_point;
        cl_int error = detail::errHandler(
            pfn_clCommandCopyImageKHR(object_,
                (command_queue != nullptr) ? (*command_queue)() : nullptr,
#if CL_KHR_COMMAND_BUFFER_EXTENSION_VERSION > CL_MAKE_VERSION(0, 9, 4)
                nullptr, // Properties
#endif
                src(),
                dst(),
                src_origin.data(),
                dst_origin.data(),
                region.data(),
                (sync_points_vec != nullptr) ? (cl_uint) sync_points_vec->size() : 0,
                (sync_points_vec != nullptr && sync_points_vec->size() > 0) ? &sync_points_vec->front() : nullptr,
                (sync_point != nullptr) ? &tmp_sync_point : nullptr,
                (cl_mutable_command_khr*) mutable_handle),
            __COMMAND_COPY_IMAGE_KHR_ERR);

        if (sync_point != nullptr && error == CL_SUCCESS)
            *sync_point = tmp_sync_point;

        return error;
    }

    cl_int commandCopyImageToBuffer(const Image& src,
        const Buffer& dst,
        const array<size_type, 3>& src_origin,
        const array<size_type, 3>& region,
        size_type dst_offset,
        const vector<cl_sync_point_khr>* sync_points_vec = nullptr,
        cl_sync_point_khr* sync_point = nullptr,
        MutableCommandKhr* mutable_handle = nullptr,
        const CommandQueue* command_queue = nullptr)
    {
        if (pfn_clCommandCopyImageToBufferKHR == nullptr) {
            return detail::errHandler(CL_INVALID_OPERATION,
                    __COMMAND_COPY_IMAGE_TO_BUFFER_KHR_ERR);
        }

        cl_sync_point_khr tmp_sync_point;
        cl_int error = detail::errHandler(
            pfn_clCommandCopyImageToBufferKHR(object_,
                (command_queue != nullptr) ? (*command_queue)() : nullptr,
#if CL_KHR_COMMAND_BUFFER_EXTENSION_VERSION > CL_MAKE_VERSION(0, 9, 4)
                nullptr, // Properties
#endif
                src(),
                dst(),
                src_origin.data(),
                region.data(),
                dst_offset,
                (sync_points_vec != nullptr) ? (cl_uint) sync_points_vec->size() : 0,
                (sync_points_vec != nullptr && sync_points_vec->size() > 0) ? &sync_points_vec->front() : nullptr,
                (sync_point != nullptr) ? &tmp_sync_point : nullptr,
                (cl_mutable_command_khr*) mutable_handle),
            __COMMAND_COPY_IMAGE_TO_BUFFER_KHR_ERR);

        if (sync_point != nullptr && error == CL_SUCCESS)
            *sync_point = tmp_sync_point;

        return error;
    }

    template<typename PatternType>
    cl_int commandFillBuffer(const Buffer& buffer,
        PatternType pattern,
        size_type offset,
        size_type size,
        const vector<cl_sync_point_khr>* sync_points_vec = nullptr,
        cl_sync_point_khr* sync_point = nullptr,
        MutableCommandKhr* mutable_handle = nullptr,
        const CommandQueue* command_queue = nullptr)
    {
        if (pfn_clCommandFillBufferKHR == nullptr) {
            return detail::errHandler(CL_INVALID_OPERATION,
                    __COMMAND_FILL_BUFFER_KHR_ERR);
        }

        cl_sync_point_khr tmp_sync_point;
        cl_int error = detail::errHandler(
            pfn_clCommandFillBufferKHR(object_,
                (command_queue != nullptr) ? (*command_queue)() : nullptr,
#if CL_KHR_COMMAND_BUFFER_EXTENSION_VERSION > CL_MAKE_VERSION(0, 9, 4)
                nullptr, // Properties
#endif
                buffer(),
                static_cast<void*>(&pattern),
                sizeof(PatternType),
                offset,
                size,
                (sync_points_vec != nullptr) ? (cl_uint) sync_points_vec->size() : 0,
                (sync_points_vec != nullptr && sync_points_vec->size() > 0) ? &sync_points_vec->front() : nullptr,
                (sync_point != nullptr) ? &tmp_sync_point : nullptr,
                (cl_mutable_command_khr*) mutable_handle),
            __COMMAND_FILL_BUFFER_KHR_ERR);

        if (sync_point != nullptr && error == CL_SUCCESS)
            *sync_point = tmp_sync_point;

        return error;
    }

    cl_int commandFillImage(const Image& image,
        cl_float4 fillColor,
        const array<size_type, 3>& origin,
        const array<size_type, 3>& region,
        const vector<cl_sync_point_khr>* sync_points_vec = nullptr,
        cl_sync_point_khr* sync_point = nullptr,
        MutableCommandKhr* mutable_handle = nullptr,
        const CommandQueue* command_queue = nullptr)
    {
        if (pfn_clCommandFillImageKHR == nullptr) {
            return detail::errHandler(CL_INVALID_OPERATION,
                    __COMMAND_FILL_IMAGE_KHR_ERR);
        }

        cl_sync_point_khr tmp_sync_point;
        cl_int error = detail::errHandler(
            pfn_clCommandFillImageKHR(object_,
                (command_queue != nullptr) ? (*command_queue)() : nullptr,
#if CL_KHR_COMMAND_BUFFER_EXTENSION_VERSION > CL_MAKE_VERSION(0, 9, 4)
                nullptr, // Properties
#endif
                image(),
                static_cast<void*>(&fillColor),
                origin.data(),
                region.data(),
                (sync_points_vec != nullptr) ? (cl_uint) sync_points_vec->size() : 0,
                (sync_points_vec != nullptr && sync_points_vec->size() > 0) ? &sync_points_vec->front() : nullptr,
                (sync_point != nullptr) ? &tmp_sync_point : nullptr,
                (cl_mutable_command_khr*) mutable_handle),
            __COMMAND_FILL_IMAGE_KHR_ERR);

        if (sync_point != nullptr && error == CL_SUCCESS)
            *sync_point = tmp_sync_point;

        return error;
    }

    cl_int commandNDRangeKernel(
#if CL_KHR_COMMAND_BUFFER_EXTENSION_VERSION > CL_MAKE_VERSION(0, 9, 4)
            const cl::vector<cl_command_properties_khr> &properties,
#else
            const cl::vector<cl_ndrange_kernel_command_properties_khr> &properties,
#endif
        const Kernel& kernel,
        const NDRange& offset,
        const NDRange& global,
        const NDRange& local = NullRange,
        const vector<cl_sync_point_khr>* sync_points_vec = nullptr,
        cl_sync_point_khr* sync_point = nullptr,
        MutableCommandKhr* mutable_handle = nullptr,
        const CommandQueue* command_queue = nullptr)
    {
        if (pfn_clCommandNDRangeKernelKHR == nullptr) {
            return detail::errHandler(CL_INVALID_OPERATION,
                    __COMMAND_NDRANGE_KERNEL_KHR_ERR);
        }

        cl_sync_point_khr tmp_sync_point;
        cl_int error = detail::errHandler(
            pfn_clCommandNDRangeKernelKHR(object_,
                (command_queue != nullptr) ? (*command_queue)() : nullptr,
                &properties[0],
                kernel(),
                (cl_uint) global.dimensions(),
                offset.dimensions() != 0 ? (const size_type*) offset : nullptr,
                (const size_type*) global,
                local.dimensions() != 0 ? (const size_type*) local : nullptr,
                (sync_points_vec != nullptr) ? (cl_uint) sync_points_vec->size() : 0,
                (sync_points_vec != nullptr && sync_points_vec->size() > 0) ? &sync_points_vec->front() : nullptr,
                (sync_point != nullptr) ? &tmp_sync_point : nullptr,
                (cl_mutable_command_khr*) mutable_handle),
            __COMMAND_NDRANGE_KERNEL_KHR_ERR);

        if (sync_point != nullptr && error == CL_SUCCESS)
            *sync_point = tmp_sync_point;

        return error;
    }

#if defined(cl_khr_command_buffer_mutable_dispatch)
#if CL_KHR_COMMAND_BUFFER_MUTABLE_DISPATCH_EXTENSION_VERSION <                 \
    CL_MAKE_VERSION(0, 9, 2)
    cl_int updateMutableCommands(const cl_mutable_base_config_khr* mutable_config)
    {
        if (pfn_clUpdateMutableCommandsKHR == nullptr) {
            return detail::errHandler(CL_INVALID_OPERATION,
                    __UPDATE_MUTABLE_COMMANDS_KHR_ERR);
        }
        return detail::errHandler(pfn_clUpdateMutableCommandsKHR(object_, mutable_config),
                        __UPDATE_MUTABLE_COMMANDS_KHR_ERR);
    }
#else
    template <int ArrayLength>
    cl_int updateMutableCommands(std::array<cl_command_buffer_update_type_khr,
                                            ArrayLength> &config_types,
                                 std::array<const void *, ArrayLength> &configs) {
        if (pfn_clUpdateMutableCommandsKHR == nullptr) {
            return detail::errHandler(CL_INVALID_OPERATION,
                                      __UPDATE_MUTABLE_COMMANDS_KHR_ERR);
        }
        return detail::errHandler(
            pfn_clUpdateMutableCommandsKHR(object_, static_cast<cl_uint>(configs.size()),
                                           config_types.data(), configs.data()),
            __UPDATE_MUTABLE_COMMANDS_KHR_ERR);
    }
#endif /* CL_KHR_COMMAND_BUFFER_MUTABLE_DISPATCH_EXTENSION_VERSION */
#endif /* cl_khr_command_buffer_mutable_dispatch */

private:
    static std::once_flag ext_init_;

    static void initExtensions(const cl::Device& device)
    {
#if CL_HPP_TARGET_OPENCL_VERSION >= 120
        cl_platform_id platform = device.getInfo<CL_DEVICE_PLATFORM>()();
        CL_HPP_INIT_CL_EXT_FCN_PTR_PLATFORM_(platform, clCreateCommandBufferKHR);
        CL_HPP_INIT_CL_EXT_FCN_PTR_PLATFORM_(platform, clFinalizeCommandBufferKHR);
        CL_HPP_INIT_CL_EXT_FCN_PTR_PLATFORM_(platform, clRetainCommandBufferKHR);
        CL_HPP_INIT_CL_EXT_FCN_PTR_PLATFORM_(platform, clReleaseCommandBufferKHR);
        CL_HPP_INIT_CL_EXT_FCN_PTR_PLATFORM_(platform, clGetCommandBufferInfoKHR);
        CL_HPP_INIT_CL_EXT_FCN_PTR_PLATFORM_(platform, clEnqueueCommandBufferKHR);
        CL_HPP_INIT_CL_EXT_FCN_PTR_PLATFORM_(platform, clCommandBarrierWithWaitListKHR);
        CL_HPP_INIT_CL_EXT_FCN_PTR_PLATFORM_(platform, clCommandCopyBufferKHR);
        CL_HPP_INIT_CL_EXT_FCN_PTR_PLATFORM_(platform, clCommandCopyBufferRectKHR);
        CL_HPP_INIT_CL_EXT_FCN_PTR_PLATFORM_(platform, clCommandCopyBufferToImageKHR);
        CL_HPP_INIT_CL_EXT_FCN_PTR_PLATFORM_(platform, clCommandCopyImageKHR);
        CL_HPP_INIT_CL_EXT_FCN_PTR_PLATFORM_(platform, clCommandCopyImageToBufferKHR);
        CL_HPP_INIT_CL_EXT_FCN_PTR_PLATFORM_(platform, clCommandFillBufferKHR);
        CL_HPP_INIT_CL_EXT_FCN_PTR_PLATFORM_(platform, clCommandFillImageKHR);
        CL_HPP_INIT_CL_EXT_FCN_PTR_PLATFORM_(platform, clCommandNDRangeKernelKHR);
#if defined(cl_khr_command_buffer_mutable_dispatch)
        CL_HPP_INIT_CL_EXT_FCN_PTR_PLATFORM_(platform, clUpdateMutableCommandsKHR);
        CL_HPP_INIT_CL_EXT_FCN_PTR_PLATFORM_(platform, clGetMutableCommandInfoKHR);
#endif /* cl_khr_command_buffer_mutable_dispatch */
#elif CL_HPP_TARGET_OPENCL_VERSION >= 110
        CL_HPP_INIT_CL_EXT_FCN_PTR_(clCreateCommandBufferKHR);
        CL_HPP_INIT_CL_EXT_FCN_PTR_(clFinalizeCommandBufferKHR);
        CL_HPP_INIT_CL_EXT_FCN_PTR_(clRetainCommandBufferKHR);
        CL_HPP_INIT_CL_EXT_FCN_PTR_(clReleaseCommandBufferKHR);
        CL_HPP_INIT_CL_EXT_FCN_PTR_(clGetCommandBufferInfoKHR);
        CL_HPP_INIT_CL_EXT_FCN_PTR_(clEnqueueCommandBufferKHR);
        CL_HPP_INIT_CL_EXT_FCN_PTR_(clCommandBarrierWithWaitListKHR);
        CL_HPP_INIT_CL_EXT_FCN_PTR_(clCommandCopyBufferKHR);
        CL_HPP_INIT_CL_EXT_FCN_PTR_(clCommandCopyBufferRectKHR);
        CL_HPP_INIT_CL_EXT_FCN_PTR_(clCommandCopyBufferToImageKHR);
        CL_HPP_INIT_CL_EXT_FCN_PTR_(clCommandCopyImageKHR);
        CL_HPP_INIT_CL_EXT_FCN_PTR_(clCommandCopyImageToBufferKHR);
        CL_HPP_INIT_CL_EXT_FCN_PTR_(clCommandFillBufferKHR);
        CL_HPP_INIT_CL_EXT_FCN_PTR_(clCommandFillImageKHR);
        CL_HPP_INIT_CL_EXT_FCN_PTR_(clCommandNDRangeKernelKHR);
#if defined(cl_khr_command_buffer_mutable_dispatch)
        CL_HPP_INIT_CL_EXT_FCN_PTR_(clUpdateMutableCommandsKHR);
        CL_HPP_INIT_CL_EXT_FCN_PTR_(clGetMutableCommandInfoKHR);
#endif /* cl_khr_command_buffer_mutable_dispatch */
#endif
        if ((pfn_clCreateCommandBufferKHR        == nullptr) &&
            (pfn_clFinalizeCommandBufferKHR      == nullptr) &&
            (pfn_clRetainCommandBufferKHR        == nullptr) &&
            (pfn_clReleaseCommandBufferKHR       == nullptr) &&
            (pfn_clGetCommandBufferInfoKHR       == nullptr) &&
            (pfn_clEnqueueCommandBufferKHR       == nullptr) &&
            (pfn_clCommandBarrierWithWaitListKHR == nullptr) &&
            (pfn_clCommandCopyBufferKHR          == nullptr) &&
            (pfn_clCommandCopyBufferRectKHR      == nullptr) &&
            (pfn_clCommandCopyBufferToImageKHR   == nullptr) &&
            (pfn_clCommandCopyImageKHR           == nullptr) &&
            (pfn_clCommandCopyImageToBufferKHR   == nullptr) &&
            (pfn_clCommandFillBufferKHR          == nullptr) &&
            (pfn_clCommandFillImageKHR           == nullptr) &&
            (pfn_clCommandNDRangeKernelKHR       == nullptr)
#if defined(cl_khr_command_buffer_mutable_dispatch)
            && (pfn_clUpdateMutableCommandsKHR      == nullptr)
            && (pfn_clGetMutableCommandInfoKHR      == nullptr)
#endif /* cl_khr_command_buffer_mutable_dispatch */
            )
        {
            detail::errHandler(CL_INVALID_VALUE, __CREATE_COMMAND_BUFFER_KHR_ERR);
        }
    }
}; // CommandBufferKhr

CL_HPP_DEFINE_STATIC_MEMBER_ std::once_flag CommandBufferKhr::ext_init_;

#if defined(cl_khr_command_buffer_mutable_dispatch)
/*! \class MutableCommandKhr
 * \brief MutableCommandKhr interface for cl_mutable_command_khr.
 */
class MutableCommandKhr : public detail::Wrapper<cl_mutable_command_khr>
{
public:
    //! \brief Default constructor - initializes to nullptr.
    MutableCommandKhr() : detail::Wrapper<cl_type>() { }

    explicit MutableCommandKhr(const cl_mutable_command_khr& mutableCommandKhr, bool retainObject = false) :
        detail::Wrapper<cl_type>(mutableCommandKhr, retainObject) { }

    MutableCommandKhr& operator=(const cl_mutable_command_khr& rhs)
    {
        detail::Wrapper<cl_type>::operator=(rhs);
        return *this;
    }

    template <typename T>
    cl_int getInfo(cl_mutable_command_info_khr name, T* param) const
    {
        if (pfn_clGetMutableCommandInfoKHR == nullptr) {
            return detail::errHandler(CL_INVALID_OPERATION,
                    __GET_MUTABLE_COMMAND_INFO_KHR_ERR);
        }
        return detail::errHandler(
            detail::getInfo(pfn_clGetMutableCommandInfoKHR, object_, name, param),
                __GET_MUTABLE_COMMAND_INFO_KHR_ERR);
    }

    template <cl_mutable_command_info_khr name> typename
        detail::param_traits<detail::cl_mutable_command_info_khr, name>::param_type
        getInfo(cl_int* err = nullptr) const
    {
        typename detail::param_traits<
            detail::cl_mutable_command_info_khr, name>::param_type param;
        cl_int result = getInfo(name, &param);
        if (err != nullptr) {
            *err = result;
        }
        return param;
    }
}; // MutableCommandKhr
#endif /* cl_khr_command_buffer_mutable_dispatch */

#endif // cl_khr_command_buffer
//----------------------------------------------------------------------------------------------------------------------

#undef CL_HPP_ERR_STR_
#if !defined(CL_HPP_USER_OVERRIDE_ERROR_STRINGS)
#undef __GET_DEVICE_INFO_ERR               
#undef __GET_PLATFORM_INFO_ERR             
#undef __GET_DEVICE_IDS_ERR                
#undef __GET_PLATFORM_IDS_ERR              
#undef __GET_CONTEXT_INFO_ERR              
#undef __GET_EVENT_INFO_ERR                
#undef __GET_EVENT_PROFILE_INFO_ERR        
#undef __GET_MEM_OBJECT_INFO_ERR           
#undef __GET_IMAGE_INFO_ERR                
#undef __GET_SAMPLER_INFO_ERR              
#undef __GET_KERNEL_INFO_ERR               
#undef __GET_KERNEL_ARG_INFO_ERR           
#undef __GET_KERNEL_SUB_GROUP_INFO_ERR     
#undef __GET_KERNEL_WORK_GROUP_INFO_ERR    
#undef __GET_PROGRAM_INFO_ERR              
#undef __GET_PROGRAM_BUILD_INFO_ERR        
#undef __GET_COMMAND_QUEUE_INFO_ERR        
#undef __CREATE_CONTEXT_ERR                
#undef __CREATE_CONTEXT_FROM_TYPE_ERR
#undef __CREATE_COMMAND_BUFFER_KHR_ERR
#undef __GET_COMMAND_BUFFER_INFO_KHR_ERR
#undef __FINALIZE_COMMAND_BUFFER_KHR_ERR
#undef __ENQUEUE_COMMAND_BUFFER_KHR_ERR
#undef __COMMAND_BARRIER_WITH_WAIT_LIST_KHR_ERR
#undef __COMMAND_COPY_BUFFER_KHR_ERR
#undef __COMMAND_COPY_BUFFER_RECT_KHR_ERR
#undef __COMMAND_COPY_BUFFER_TO_IMAGE_KHR_ERR
#undef __COMMAND_COPY_IMAGE_KHR_ERR
#undef __COMMAND_COPY_IMAGE_TO_BUFFER_KHR_ERR
#undef __COMMAND_FILL_BUFFER_KHR_ERR
#undef __COMMAND_FILL_IMAGE_KHR_ERR
#undef __COMMAND_NDRANGE_KERNEL_KHR_ERR
#undef __UPDATE_MUTABLE_COMMANDS_KHR_ERR
#undef __GET_MUTABLE_COMMAND_INFO_KHR_ERR
#undef __RETAIN_COMMAND_BUFFER_KHR_ERR
#undef __RELEASE_COMMAND_BUFFER_KHR_ERR
#undef __GET_SUPPORTED_IMAGE_FORMATS_ERR   
#undef __SET_CONTEXT_DESCTRUCTOR_CALLBACK_ERR
#undef __CREATE_BUFFER_ERR                 
#undef __COPY_ERR                          
#undef __CREATE_SUBBUFFER_ERR              
#undef __CREATE_GL_BUFFER_ERR              
#undef __CREATE_GL_RENDER_BUFFER_ERR       
#undef __GET_GL_OBJECT_INFO_ERR            
#undef __CREATE_IMAGE_ERR                  
#undef __CREATE_GL_TEXTURE_ERR             
#undef __IMAGE_DIMENSION_ERR               
#undef __SET_MEM_OBJECT_DESTRUCTOR_CALLBACK_ERR 
#undef __CREATE_USER_EVENT_ERR             
#undef __SET_USER_EVENT_STATUS_ERR         
#undef __SET_EVENT_CALLBACK_ERR            
#undef __WAIT_FOR_EVENTS_ERR               
#undef __CREATE_KERNEL_ERR                 
#undef __SET_KERNEL_ARGS_ERR               
#undef __CREATE_PROGRAM_WITH_SOURCE_ERR    
#undef __CREATE_PROGRAM_WITH_BINARY_ERR    
#undef __CREATE_PROGRAM_WITH_IL_ERR        
#undef __CREATE_PROGRAM_WITH_BUILT_IN_KERNELS_ERR    
#undef __BUILD_PROGRAM_ERR                 
#undef __COMPILE_PROGRAM_ERR               
#undef __LINK_PROGRAM_ERR                  
#undef __CREATE_KERNELS_IN_PROGRAM_ERR     
#undef __CREATE_COMMAND_QUEUE_WITH_PROPERTIES_ERR          
#undef __CREATE_SAMPLER_WITH_PROPERTIES_ERR                
#undef __SET_COMMAND_QUEUE_PROPERTY_ERR    
#undef __ENQUEUE_READ_BUFFER_ERR           
#undef __ENQUEUE_READ_BUFFER_RECT_ERR      
#undef __ENQUEUE_WRITE_BUFFER_ERR          
#undef __ENQUEUE_WRITE_BUFFER_RECT_ERR     
#undef __ENQEUE_COPY_BUFFER_ERR            
#undef __ENQEUE_COPY_BUFFER_RECT_ERR       
#undef __ENQUEUE_FILL_BUFFER_ERR           
#undef __ENQUEUE_READ_IMAGE_ERR            
#undef __ENQUEUE_WRITE_IMAGE_ERR           
#undef __ENQUEUE_COPY_IMAGE_ERR            
#undef __ENQUEUE_FILL_IMAGE_ERR            
#undef __ENQUEUE_COPY_IMAGE_TO_BUFFER_ERR  
#undef __ENQUEUE_COPY_BUFFER_TO_IMAGE_ERR  
#undef __ENQUEUE_MAP_BUFFER_ERR
#undef __ENQUEUE_MAP_IMAGE_ERR
#undef __ENQUEUE_MAP_SVM_ERR
#undef __ENQUEUE_FILL_SVM_ERR
#undef __ENQUEUE_COPY_SVM_ERR
#undef __ENQUEUE_UNMAP_SVM_ERR              
#undef __ENQUEUE_MAP_IMAGE_ERR             
#undef __ENQUEUE_UNMAP_MEM_OBJECT_ERR      
#undef __ENQUEUE_NDRANGE_KERNEL_ERR        
#undef __ENQUEUE_NATIVE_KERNEL             
#undef __ENQUEUE_MIGRATE_MEM_OBJECTS_ERR   
#undef __ENQUEUE_MIGRATE_SVM_ERR
#undef __ENQUEUE_ACQUIRE_GL_ERR            
#undef __ENQUEUE_RELEASE_GL_ERR            
#undef __CREATE_PIPE_ERR             
#undef __GET_PIPE_INFO_ERR           
#undef __RETAIN_ERR                        
#undef __RELEASE_ERR                       
#undef __FLUSH_ERR                         
#undef __FINISH_ERR                        
#undef __VECTOR_CAPACITY_ERR               
#undef __CREATE_SUB_DEVICES_ERR
#undef __ENQUEUE_ACQUIRE_EXTERNAL_MEMORY_ERR
#undef __ENQUEUE_RELEASE_EXTERNAL_MEMORY_ERR
#undef __ENQUEUE_MARKER_ERR                
#undef __ENQUEUE_WAIT_FOR_EVENTS_ERR       
#undef __ENQUEUE_BARRIER_ERR               
#undef __UNLOAD_COMPILER_ERR               
#undef __CREATE_GL_TEXTURE_2D_ERR          
#undef __CREATE_GL_TEXTURE_3D_ERR          
#undef __CREATE_IMAGE2D_ERR                
#undef __CREATE_IMAGE3D_ERR                
#undef __CREATE_COMMAND_QUEUE_ERR          
#undef __ENQUEUE_TASK_ERR                  
#undef __CREATE_SAMPLER_ERR                
#undef __ENQUEUE_MARKER_WAIT_LIST_ERR                
#undef __ENQUEUE_BARRIER_WAIT_LIST_ERR               
#undef __CLONE_KERNEL_ERR     
#undef __GET_HOST_TIMER_ERR
#undef __GET_DEVICE_AND_HOST_TIMER_ERR
#undef __GET_SEMAPHORE_KHR_INFO_ERR
#undef __CREATE_SEMAPHORE_KHR_WITH_PROPERTIES_ERR
#undef __GET_IMAGE_REQUIREMENT_INFO_EXT_ERR
#undef __ENQUEUE_WAIT_SEMAPHORE_KHR_ERR
#undef __ENQUEUE_SIGNAL_SEMAPHORE_KHR_ERR
#undef __RETAIN_SEMAPHORE_KHR_ERR
#undef __RELEASE_SEMAPHORE_KHR_ERR
#undef __GET_SEMAPHORE_HANDLE_FOR_TYPE_KHR_ERR

#endif //CL_HPP_USER_OVERRIDE_ERROR_STRINGS

// Extensions
#undef CL_HPP_CREATE_CL_EXT_FCN_PTR_ALIAS_
#undef CL_HPP_INIT_CL_EXT_FCN_PTR_
#undef CL_HPP_INIT_CL_EXT_FCN_PTR_PLATFORM_

#undef CL_HPP_DEFINE_STATIC_MEMBER_

} // namespace cl

#endif // CL_HPP_
