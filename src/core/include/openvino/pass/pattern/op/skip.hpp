// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"

namespace ov {
namespace pass {
namespace pattern {
namespace op {
/// The graph value is added to the matched value list. If the predicate is true, the
/// match succeeds if the arguments match; if the predicate is false, the match succeeds
/// if the pattern input matches the graph value.
class OPENVINO_API Skip : public Pattern {
public:
    OPENVINO_RTTI("patternSkip");
    BWDCMP_RTTI_DECLARATION;
    Skip(const Output<Node>& arg, ValuePredicate pred) : Pattern({arg}, pred) {
        set_output_type(0, arg.get_element_type(), arg.get_partial_shape());
    }

    Skip(const Output<Node>& arg, NodePredicate pred = nullptr) : Pattern({arg}, as_value_predicate(pred)) {
        set_output_type(0, arg.get_element_type(), arg.get_partial_shape());
    }

    Skip(const OutputVector& args, ValuePredicate pred) : Pattern(args, pred) {
        set_output_type(0, args.at(0).get_element_type(), args.at(0).get_partial_shape());
    }

    Skip(const OutputVector& args, NodePredicate pred = nullptr) : Pattern(args, as_value_predicate(pred)) {
        set_output_type(0, args.at(0).get_element_type(), args.at(0).get_partial_shape());
    }

    bool match_value(pattern::Matcher* matcher,
                     const Output<Node>& pattern_value,
                     const Output<Node>& graph_value) override;
};
}  // namespace op
}  // namespace pattern
}  // namespace pass
}  // namespace ov
