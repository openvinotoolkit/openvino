// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/deprecated.hpp>
#include <ngraph/ngraph_visibility.hpp>
#include <sstream>
#include <stdexcept>

#include "openvino/core/except.hpp"

namespace ngraph {
/// Base error for ngraph runtime errors.
using ngraph_error = ov::Exception;

class NGRAPH_DEPRECATED("This class is deprecated and will be removed soon.") NGRAPH_API unsupported_op
    : public std::runtime_error {
public:
    unsupported_op(const std::string& what_arg) : std::runtime_error(what_arg) {}
};
}  // namespace ngraph
