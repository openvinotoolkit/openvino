// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
} // namespace ngraph
