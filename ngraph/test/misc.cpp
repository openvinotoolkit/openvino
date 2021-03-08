//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************
#include "misc.hpp"

FILE* port_open(const char* command, const char* type)
{
#ifdef _WIN32
    return _popen(command, type);
#elif defined(__linux) || defined(__APPLE__)
    return popen(command, type);
#endif
}

int port_close(FILE* stream)
{
#ifdef _WIN32
    return _pclose(stream);
#elif defined(__linux) || defined(__APPLE__)
    return pclose(stream);
#endif
}

int set_environment(const char* name, const char* value, int overwrite)
{
#ifdef _WIN32
    return _putenv_s(name, value);
#elif defined(__linux) || defined(__APPLE__)
    return setenv(name, value, overwrite);
#endif
}

int unset_environment(const char* name)
{
#ifdef _WIN32
    return _putenv_s(name, "");
#elif defined(__linux) || defined(__APPLE__)
    return unsetenv(name);
#endif
}
