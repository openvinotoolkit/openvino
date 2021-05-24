// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
} // namespace ngraph
