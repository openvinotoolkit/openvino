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

#include <functional>
#include <memory>
#include <ngraph/log.hpp>
#include <set>

#include "ngraph/pass/pass.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace ngraph {
using ov::graph_rewrite_callback;
using ov::handler_callback;
using ov::matcher_pass_callback;
namespace pass {

using ov::pass::BackwardGraphRewrite;
using ov::pass::GraphRewrite;
using ov::pass::MatcherPass;

}  // namespace pass
}  // namespace ngraph
