// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <set>

#include "ngraph/pass/pass.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace ngraph {
using ov::graph_rewrite_callback;
using ov::handler_callback;
using ov::matcher_pass_callback;
using ov::recurrent_graph_rewrite_callback;
namespace pass {
using ov::pass::BackwardGraphRewrite;
using ov::pass::GraphRewrite;
using ov::pass::MatcherPass;
using ov::pass::RecurrentGraphRewrite;
}  // namespace pass
}  // namespace ngraph
