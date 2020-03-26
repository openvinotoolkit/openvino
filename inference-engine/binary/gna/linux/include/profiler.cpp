/*
 INTEL CONFIDENTIAL
 Copyright 2017 Intel Corporation.

 The source code contained or described herein and all documents related
 to the source code ("Material") are owned by Intel Corporation or its suppliers
 or licensors. Title to the Material remains with Intel Corporation or its suppliers
 and licensors. The Material may contain trade secrets and proprietary
 and confidential information of Intel Corporation and its suppliers and licensors,
 and is protected by worldwide copyright and trade secret laws and treaty provisions.
 No part of the Material may be used, copied, reproduced, modified, published,
 uploaded, posted, transmitted, distributed, or disclosed in any way without Intel's
 prior express written permission.

 No license under any patent, copyright, trade secret or other intellectual
 property right is granted to or conferred upon you by disclosure or delivery
 of the Materials, either expressly, by implication, inducement, estoppel
 or otherwise. Any license under such intellectual property rights must
 be express and approved by Intel in writing.

 Unless otherwise agreed by Intel in writing, you may not remove or alter this notice
 or any other notice embedded in Materials by Intel or Intel's suppliers or licensors
 in any way.
*/

#include <limits.h>
#include <stdlib.h>
#include <stdint.h>

#if defined(_WIN32)
#include <Windows.h>
#endif

#include "profiler.h"

#if defined(_WIN32)

void getTsc(uint64_t * const result)
{
    QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER * const>(result));
}

#else
#if defined(__GNUC__) && !defined(__clang__)
static __inline__ uint64_t __rdtsc(void)
{
    unsigned hi, lo;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return (static_cast<uint64_t>(lo)) | ((static_cast<uint64_t>(hi)) << 32);
}
#endif

void getTsc(uint64_t * const result)
{
    *result = __rdtsc();
}
#endif