// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>

#include "ngraph/node.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"

namespace ov {
namespace pass {
namespace pattern {
namespace op {
class Label;
}

class Matcher;
class MatchState;
}  // namespace pattern
}  // namespace pass
}  // namespace ov
namespace ngraph {
namespace pattern {
namespace op {
using ov::pass::pattern::op::Label;
}

using ov::pass::pattern::Matcher;
using ov::pass::pattern::MatcherState;

using ov::pass::pattern::PatternValueMap;
using ov::pass::pattern::PatternValueMaps;
using ov::pass::pattern::RPatternValueMap;

using ov::pass::pattern::PatternMap;

using ov::pass::pattern::as_pattern_map;
using ov::pass::pattern::as_pattern_value_map;
using ov::pass::pattern::consumers_count;
using ov::pass::pattern::has_class;
using ov::pass::pattern::has_static_dim;
using ov::pass::pattern::has_static_dims;
using ov::pass::pattern::has_static_rank;
using ov::pass::pattern::has_static_shape;
using ov::pass::pattern::rank_equals;
using ov::pass::pattern::type_matches;
using ov::pass::pattern::type_matches_any;

namespace op {
using ov::pass::pattern::op::NodePredicate;
using ov::pass::pattern::op::ValuePredicate;

using ov::pass::pattern::op::as_value_predicate;
using ov::pass::pattern::op::Pattern;
}  // namespace op
}  // namespace pattern
}  // namespace ngraph
