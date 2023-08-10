// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if !defined(IN_OV_COMPONENT) && !defined(NGRAPH_LEGACY_HEADER_INCLUDED)
#    define NGRAPH_LEGACY_HEADER_INCLUDED
#    ifdef _MSC_VER
#        pragma message( \
            "The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    else
#        warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    endif
#endif

#include <ngraph/deprecated.hpp>
#include <ngraph/ngraph_visibility.hpp>
#include <sstream>
#include <stdexcept>

#include "openvino/core/except.hpp"

namespace ngraph {
/// Base error for ngraph runtime errors.
using ngraph_error = ov::Exception;

class NGRAPH_API_DEPRECATED NGRAPH_API unsupported_op : public std::runtime_error {
public:
    unsupported_op(const std::string& what_arg) : std::runtime_error(what_arg) {}
};
}  // namespace ngraph
