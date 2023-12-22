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

#include <memory.h>

#include <algorithm>
#include <functional>

#include "ngraph/node.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/pattern/op/any.hpp"
#include "ngraph/pattern/op/any_of.hpp"
#include "ngraph/pattern/op/any_output.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/pattern/op/skip.hpp"
#include "openvino/pass/pattern/matcher.hpp"

namespace ov {
namespace pass {
class GraphRewrite;
}
}  // namespace ov
namespace ngraph {
namespace pass {
using ov::pass::GraphRewrite;
}

namespace pattern {
using ov::pass::pattern::Matcher;
using ov::pass::pattern::MatcherState;
}  // namespace pattern
}  // namespace ngraph
