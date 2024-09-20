// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/pattern/op/pattern.hpp"

#include <algorithm>
#include <regex>

namespace ov {
namespace pass {
namespace pattern {
namespace op {
namespace {
constexpr bool node_value_true_predicate(const Output<Node>&) {
    return true;
}
}  // namespace

struct NodeValuePredicate {
    bool operator()(const Output<Node>& value) const {
        return pred(value.get_node_shared_ptr());
    }

    NodePredicate pred;
};

Pattern::Pattern(const OutputVector& patterns, ValuePredicate pred)
    : Node(patterns),
      m_predicate(pred ? std::move(pred) : node_value_true_predicate) {}

// The symbols are required to be in cpp file to workaround RTTI issue on Android LLVM
ValuePredicate Pattern::get_predicate() const {
    return m_predicate;
}

ValuePredicate as_value_predicate(NodePredicate pred) {
    if (pred) {
        return NodeValuePredicate{std::move(pred)};
    } else {
        return node_value_true_predicate;
    }
}

std::ostream& Pattern::write_type_description(std::ostream& out) const {
    auto version = get_type_info().version_id;
    if (version)
        out << version << "::" << get_type_info().name;
    else
        out << get_type_info().name;

    return out;
}

std::ostream& Pattern::write_description(std::ostream& out, uint32_t depth) const {
    write_type_description(out);

    if (depth > 0) {
        out << " (";
        std::string sep = "";
        for (const auto& arg : input_values()) {
            out << sep << arg;
            sep = ", ";
        }
        out << ") -> (";
        sep = "";
        for (size_t i = 0; i < get_output_size(); i++) {
            out << sep << get_output_element_type(i) << get_output_partial_shape(i);
            sep = ", ";
        }
        out << ")";
    }
    return out;
}

}  // namespace op

PatternMap as_pattern_map(const PatternValueMap& pattern_value_map) {
    PatternMap result;
    for (auto& kv : pattern_value_map) {
        result[kv.first] = kv.second.get_node_shared_ptr();
    }
    return result;
}

PatternValueMap as_pattern_value_map(const PatternMap& pattern_map) {
    PatternValueMap result;
    for (auto& kv : pattern_map) {
        result[kv.first] = kv.second;
    }
    return result;
}

std::function<bool(Output<Node>)> consumers_count(size_t n) {
    return [=](Output<Node> output) -> bool {
        return output.get_target_inputs().size() == n;
    };
}

std::function<bool(Output<Node>)> consumers_more_than(size_t n) {
    return [=](Output<Node> output) -> bool {
        return output.get_target_inputs().size() > n;
    };
}

std::function<bool(Output<Node>)> has_static_dim(size_t pos) {
    return [=](Output<Node> output) -> bool {
        const auto& shape = output.get_partial_shape();
        return shape.rank().is_static() && shape.rank().get_length() > static_cast<int64_t>(pos) &&
               shape[pos].is_static();
    };
}

std::function<bool(Output<Node>)> has_static_dims(const std::vector<size_t>& dims) {
    return [=](Output<Node> output) -> bool {
        const auto& shape = output.get_partial_shape();
        return shape.rank().is_static() &&
               shape.rank().get_length() > static_cast<int64_t>(*std::max_element(dims.begin(), dims.end())) &&
               std::all_of(dims.begin(), dims.end(), [&shape](size_t pos) {
                   return shape[pos].is_static();
               });
    };
}

std::function<bool(Output<Node>)> has_static_shape() {
    return [=](Output<Node> output) -> bool {
        return output.get_partial_shape().is_static();
    };
}

std::function<bool(Output<Node>)> has_static_rank() {
    return [=](Output<Node> output) -> bool {
        return output.get_partial_shape().rank().is_static();
    };
}

std::function<bool(Output<Node>)> rank_equals(const Dimension& expected_rank) {
    return [=](Output<Node> output) -> bool {
        return output.get_partial_shape().rank() == expected_rank;
    };
}

std::function<bool(Output<Node>)> type_matches(const element::Type& type) {
    return [=](Output<Node> output) -> bool {
        return output.get_element_type() == type;
    };
}

std::function<bool(Output<Node>)> type_matches_any(const std::vector<element::Type>& expected_types) {
    return [=](Output<Node> output) -> bool {
        const auto& output_type = output.get_element_type();
        return std::any_of(expected_types.begin(), expected_types.end(), [=](element::Type type) {
            return type == output_type;
        });
    };
}

std::function<bool(Output<Node>)> all_of(const std::vector<std::function<bool(Output<Node>)>>& predicates) {
    return [=](Output<Node> output) -> bool {
        for (auto& p : predicates) {
            if (!p(output))
                return false;
        }
        return true;
    };
}
}  // namespace pattern
}  // namespace pass
}  // namespace ov
