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

#include <cstddef>

#include "ngraph/util.hpp"
#include "ngraph/version.hpp"

using namespace std;

extern "C" NGRAPH_API const char* get_ngraph_version_string()
{
    return NGRAPH_VERSION_NUMBER;
}

namespace ngraph
{
    NGRAPH_API void get_version(size_t& major, size_t& minor, size_t& patch, std::string& extra)
    {
        string version = NGRAPH_VERSION_NUMBER;
        ngraph::parse_version_string(version, major, minor, patch, extra);
    }
}
