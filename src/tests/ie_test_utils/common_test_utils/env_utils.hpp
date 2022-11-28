// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdlib.h>

namespace CommonTestUtils {

int setEnvironment(const char* name, const char* value, int overwrite = 1) {
#ifdef _WIN32
    return _putenv_s(name, value);
#elif defined(__linux) || defined(__APPLE__)
    return setenv(name, value, overwrite);
#endif
}

int unsetEnvironment(const char* name) {
#ifdef _WIN32
    return _putenv_s(name, "");
#elif defined(__linux) || defined(__APPLE__)
    return unsetenv(name);
#endif
}
} // namespace CommonTestUtils
