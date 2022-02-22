// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

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
using ov::pass::pattern::RecurrentMatcher;
}  // namespace pattern
}  // namespace ngraph
