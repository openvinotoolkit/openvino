// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace CommonTestUtils {

extern const char* DEVICE_AUTO;
extern const char* DEVICE_CPU;
extern const char* DEVICE_GNA;
extern const char* DEVICE_GPU;
extern const char* DEVICE_BATCH;
extern const char* DEVICE_KEEMBAY;
extern const char* DEVICE_MULTI;
extern const char* DEVICE_TEMPLATE;
extern const char* DEVICE_HETERO;

const char OP_REPORT_FILENAME[] = "report_op";
const char API_REPORT_FILENAME[] = "report_api";
const char REPORT_EXTENSION[] = ".xml";

const char DEVICE_SUFFIX_SEPARATOR = '.';

const unsigned int maxFileNameLength = 140;

#ifdef _WIN32
    #ifdef __MINGW32__
        const char pre[] = "lib";
    #else
        const char pre[] = "";
    #endif
    const char ext[] = ".dll";
    const char FileSeparator[] = "\\";
#else
    #if defined __APPLE__
        const char pre[] = "lib";
        const char ext[] = ".so";
    #else
        const char pre[] = "lib";
        const char ext[] = ".so";
    #endif
    const char FileSeparator[] = "/";
#endif

}  // namespace CommonTestUtils
