// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>

#include "openvino/core/node.hpp"
#include "openvino/pass/pattern/op/predicate.hpp"

namespace ov::pass::pattern {
namespace op {
class Label;
}

class Matcher;
class MatcherState;

using RPatternValueMap = std::map<std::shared_ptr<Node>, OutputVector>;
using PatternValueMap = std::map<std::shared_ptr<Node>, Output<Node>>;
using PatternValueMaps = std::vector<PatternValueMap>;

using PatternMap = std::map<std::shared_ptr<Node>, std::shared_ptr<Node>>;

PatternMap as_pattern_map(const PatternValueMap& pattern_value_map);
PatternValueMap as_pattern_value_map(const PatternMap& pattern_map);

template <typename T>
op::Predicate has_class() {
    [[maybe_unused]] bool unused;
    auto pred = [unused = true](std::shared_ptr<Node> node) -> bool {
        return ov::is_type<T>(std::move(node));
    };
    return op::Predicate(pred, "has_class<" + std::string(typeid(T).name()) + ">()");
}
template <typename T>
op::Predicate class_other_than() {
    [[maybe_unused]] bool unused;
    auto pred = [unused = true](std::shared_ptr<Node> node) -> bool {
        return !ov::is_type<T>(std::move(node));
    };
    return op::Predicate(pred, "class_other_than<" + std::string(typeid(T).name()) + ">()");
}

OPENVINO_API op::Predicate consumers_count(size_t n);
OPENVINO_API op::Predicate consumers_more_than(size_t n);

OPENVINO_API op::Predicate has_static_dim(size_t pos);
OPENVINO_API op::Predicate has_static_dims(const std::vector<size_t>& dims);

OPENVINO_API op::Predicate has_static_shape();
OPENVINO_API op::Predicate has_static_rank();
OPENVINO_API op::Predicate rank_equals(const Dimension& expected_rank);
OPENVINO_API op::Predicate rank_more_than(const Dimension& expected_rank);

OPENVINO_API op::Predicate type_matches(const element::Type& type);
OPENVINO_API op::Predicate type_matches_any(const std::vector<element::Type>& types);

OPENVINO_API op::Predicate all_of(const std::vector<std::function<bool(Output<Node>)>>& predicates);

OPENVINO_API op::Predicate shape_matches(const std::string& shape_notation);

namespace op {

OPENVINO_API
ValuePredicate as_value_predicate(NodePredicate pred);

class OPENVINO_API Pattern : public Node {
public:
    /// \brief A base class for all the utility operators used to describe a pattern to match
    Pattern(const OutputVector& patterns, const Predicate& pred);
    Pattern(const OutputVector& patterns);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& /* new_args */) const override {
        OPENVINO_THROW("Uncopyable");
    }

    std::ostream& write_description(std::ostream& out, uint32_t depth) const override;
    virtual std::ostream& write_type_description(std::ostream& out) const;

protected:
    Predicate m_predicate;
};
}  // namespace op
}  // namespace ov::pass::pattern
