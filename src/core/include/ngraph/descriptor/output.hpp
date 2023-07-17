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

#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "ngraph/descriptor/input.hpp"
#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/node_output.hpp"
#include "openvino/core/descriptor/output.hpp"

namespace ngraph {
using ov::Node;
namespace descriptor {
// Describes an output tensor of an op
using ov::descriptor::Output;
}  // namespace descriptor
}  // namespace ngraph
