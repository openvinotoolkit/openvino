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
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

#include "mkldnn.h"
#include "c_types_map.hpp"
#include "verbose.hpp"

namespace mkldnn {
namespace impl {

static verbose_t verbose;

const verbose_t *mkldnn_verbose() {
#if !defined(DISABLE_VERBOSE)
    static int initialized = 0;
    if (!initialized) {
        const int len = 2;
        char val[len] = {0};
        if (mkldnn_getenv(val, "MKLDNN_VERBOSE", len) == 1)
            verbose.level = atoi(val);
        initialized = 1;
    }
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

}
}

mkldnn_status_t mkldnn_verbose_set(int level) {
    using namespace mkldnn::impl::status;
    if (level < 0 || level > 2) return invalid_arguments;
    mkldnn::impl::verbose.level = level;
    return success;
}
