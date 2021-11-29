// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace CommonTestUtils {

const char DEVICE_AUTO[] = "AUTO";
const char DEVICE_CPU[] = "CPU";
const char DEVICE_GNA[] = "GNA";
const char DEVICE_GPU[] = "GPU";
const char DEVICE_HDDL[] = "HDDL";
const char DEVICE_MYRIAD[] = "MYRIAD";
const char DEVICE_KEEMBAY[] = "VPUX";
const char DEVICE_MULTI[] = "MULTI";
const char DEVICE_TEMPLATE[] = "TEMPLATE";
const char DEVICE_HETERO[] = "HETERO";

const char REPORT_FILENAME[] = "report";
const char REPORT_EXTENSION[] = ".xml";

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