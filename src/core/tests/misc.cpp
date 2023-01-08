// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "misc.hpp"

FILE* port_open(const char* command, const char* type) {
#ifdef _WIN32
    return _popen(command, type);
#elif defined(__linux) || defined(__APPLE__)
    return popen(command, type);
#endif
}

int port_close(FILE* stream) {
#ifdef _WIN32
    return _pclose(stream);
#elif defined(__linux) || defined(__APPLE__)
    return pclose(stream);
#endif
}

int set_environment(const char* name, const char* value, int overwrite) {
#ifdef _WIN32
    return _putenv_s(name, value);
#elif defined(__linux) || defined(__APPLE__)
    return setenv(name, value, overwrite);
#endif
}

int unset_environment(const char* name) {
#ifdef _WIN32
    return _putenv_s(name, "");
#elif defined(__linux) || defined(__APPLE__)
    return unsetenv(name);
#endif
}
