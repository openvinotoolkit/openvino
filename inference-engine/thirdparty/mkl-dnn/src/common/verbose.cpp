/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <stdlib.h>
#ifndef _WIN32
#include <sys/time.h>
#endif

#include "mkldnn.h"
#include "mkldnn_version.h"
#include "c_types_map.hpp"
#include "verbose.hpp"
#include "cpu_isa_traits.hpp"

/* Intel MKL-DNN CPU ISA info */
#define ISA_ANY "Intel 64"
#define SSE42 "Intel SSE4.2"
#define AVX "Intel AVX"
#define AVX2 "Intel AVX2"
#define AVX512_COMMON "Intel AVX-512"
#define AVX512_CORE \
    "Intel AVX-512 with AVX512BW, AVX512VL, and AVX512DQ extensions"
#define AVX512_CORE_VNNI "Intel AVX-512 with Intel DL Boost"
#define AVX512_MIC \
    "Intel AVX-512 with AVX512CD, AVX512ER, and AVX512PF extensions"
#define AVX512_MIC_4OPS \
    "Intel AVX-512 with AVX512_4FMAPS and AVX512_4VNNIW extensions"
#define AVX512_CORE_BF16 \
    "Intel AVX-512 with Intel DL Boost and bfloat16 support"

namespace mkldnn {
namespace impl {

static verbose_t verbose;
static bool initialized;
static bool version_printed = false;

const verbose_t *mkldnn_verbose() {
#if !defined(DISABLE_VERBOSE)
    if (!initialized) {
        const int len = 2;
        char val[len] = {0};
        if (mkldnn_getenv("MKLDNN_VERBOSE", val, len) == 1)
            verbose.level = atoi(val);
        initialized = true;
    }
    if (!version_printed && verbose.level > 0) {
        printf("mkldnn_verbose,info,Intel MKL-DNN v%d.%d.%d (commit %s)\n",
                mkldnn_version()->major, mkldnn_version()->minor,
                mkldnn_version()->patch, mkldnn_version()->hash);
        printf("mkldnn_verbose,info,Detected ISA is %s\n", get_isa_info());
        version_printed = true;
    }
#else
    verbose.level = 0;
#endif
    return &verbose;
}

double get_msec() {
#ifdef _WIN32
    static LARGE_INTEGER frequency;
    if (frequency.QuadPart == 0)
        QueryPerformanceFrequency(&frequency);
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return 1e+3 * now.QuadPart / frequency.QuadPart;
#else
    struct timeval time;
    gettimeofday(&time, NULL);
    return 1e+3 * time.tv_sec + 1e-3 * time.tv_usec;
#endif
}

const char *get_isa_info() {
    using namespace mkldnn::impl::cpu;
    if (mayiuse(avx512_core_bf16)) return AVX512_CORE_BF16;
    if (mayiuse(avx512_mic_4ops))  return AVX512_MIC_4OPS;
    if (mayiuse(avx512_mic))       return AVX512_MIC;
    if (mayiuse(avx512_core_vnni)) return AVX512_CORE_VNNI;
    if (mayiuse(avx512_core))      return AVX512_CORE;
    if (mayiuse(avx512_common))    return AVX512_COMMON;
    if (mayiuse(avx2))             return AVX2;
    if (mayiuse(avx))              return AVX;
    if (mayiuse(sse42))            return SSE42;
    return ISA_ANY;
}

}
}

mkldnn_status_t mkldnn_set_verbose(int level) {
    using namespace mkldnn::impl::status;
    if (level < 0 || level > 2) return invalid_arguments;
    mkldnn::impl::verbose.level = level;
    mkldnn::impl::initialized = true;
    return success;
}

const mkldnn_version_t *mkldnn_version() {
    static mkldnn_version_t ver = {
        MKLDNN_VERSION_MAJOR,
        MKLDNN_VERSION_MINOR,
        MKLDNN_VERSION_PATCH,
        MKLDNN_VERSION_HASH};
    return &ver;
}

