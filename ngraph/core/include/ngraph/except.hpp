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

#pragma once

#include <sstream>
#include <stdexcept>

#include <ngraph/ngraph_visibility.hpp>

namespace ngraph
{
    /// Base error for ngraph runtime errors.
    class NGRAPH_API ngraph_error : public std::runtime_error
    {
    public:
        explicit ngraph_error(const std::string& what_arg)
            : std::runtime_error(what_arg)
        {
        }

        explicit ngraph_error(const char* what_arg)
            : std::runtime_error(what_arg)
        {
        }

        explicit ngraph_error(const std::stringstream& what_arg)
            : std::runtime_error(what_arg.str())
        {
        }
    };

    class NGRAPH_API unsupported_op : public std::runtime_error
    {
    public:
        unsupported_op(const std::string& what_arg)
            : std::runtime_error(what_arg)
        {
        }
    };
}
