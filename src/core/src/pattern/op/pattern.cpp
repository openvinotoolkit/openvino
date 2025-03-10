// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/pattern/op/pattern.hpp"

#include <algorithm>
#include <regex>

#include "openvino/util/common_util.hpp"

namespace ov::pass::pattern {
namespace op {

Pattern::Pattern() : Node(), m_predicate() {}
Pattern::Pattern(const OutputVector& patterns) : Node(patterns), m_predicate() {}
Pattern::Pattern(const NodeVector& patterns) : Pattern(as_output_vector(patterns)) {}

Pattern::Pattern(const OutputVector& patterns, const op::Predicate& pred) : Node(patterns), m_predicate(pred) {}
Pattern::Pattern(const NodeVector& patterns, const op::Predicate& pred) : Pattern(as_output_vector(patterns), pred) {}

Predicate as_value_predicate(NodePredicate pred) {
    return Predicate(pred);
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

op::Predicate consumers_count(size_t n) {
    return op::Predicate(
        [=](const Output<Node>& output) -> bool {
            return output.get_target_inputs().size() == n;
        },
        "consumers_count(" + std::to_string(n) + ")");
}

op::Predicate consumers_more_than(size_t n) {
    return op::Predicate(
        [=](const Output<Node>& output) -> bool {
            return output.get_target_inputs().size() > n;
        },
        "consumers_more_than(" + std::to_string(n) + ")");
}

op::Predicate has_static_dim(size_t pos) {
    return op::Predicate(
        [=](const Output<Node>& output) -> bool {
            const auto& shape = output.get_partial_shape();
            return shape.rank().is_static() && shape.rank().get_length() > static_cast<int64_t>(pos) &&
                   shape[pos].is_static();
        },
        "has_static_dim(" + std::to_string(pos) + ")");
}

op::Predicate has_static_dims(const std::vector<size_t>& dims) {
    return op::Predicate(
        [=](const Output<Node>& output) -> bool {
            const auto& shape = output.get_partial_shape();
            return shape.rank().is_static() &&
                   shape.rank().get_length() > static_cast<int64_t>(*std::max_element(dims.begin(), dims.end())) &&
                   std::all_of(dims.begin(), dims.end(), [&shape](size_t pos) {
                       return shape[pos].is_static();
                   });
        },

        "has_static_dims({" + ov::util::join(dims) + "})");
}

op::Predicate has_static_shape() {
    // GCC treats an empty lambda (without captures / internal state) differently in different ABI versions
    bool gcc_abi_compatibility = true;
    return op::Predicate(
        [gcc_abi_compatibility](const Output<Node>& output) -> bool {
            return gcc_abi_compatibility && output.get_partial_shape().is_static();
        },
        "has_static_shape");
}

op::Predicate has_static_rank() {
    // GCC treats an empty lambda (without captures / internal state) differently in different ABI versions
    bool gcc_abi_compatibility = true;
    return op::Predicate(
        [gcc_abi_compatibility](const Output<Node>& output) -> bool {
            return gcc_abi_compatibility && output.get_partial_shape().rank().is_static();
        },
        "has_static_rank");
}

op::Predicate rank_equals(const Dimension& expected_rank) {
    return op::Predicate(
        [=](const Output<Node>& output) -> bool {
            return output.get_partial_shape().rank() == expected_rank;
        },
        "rank_equals(" + expected_rank.to_string() + ")");
}

op::Predicate rank_more_than(const Dimension& expected_rank) {
    OPENVINO_ASSERT(expected_rank.is_static(), "Predicate `rank_more_than` registered with dynamic rank");
    return op::Predicate(
        [=](const Output<Node>& output) -> bool {
            const auto& rank = output.get_partial_shape().rank();
            return rank.is_static() && rank.get_length() > expected_rank.get_length();
        },
        "rank_more_than(" + expected_rank.to_string() + ")");
}

op::Predicate type_matches(const element::Type& type) {
    return op::Predicate(
        [=](const Output<Node>& output) -> bool {
            return output.get_element_type() == type;
        },
        "type_matches(" + type.to_string() + ")");
}

op::Predicate type_matches_any(const std::vector<element::Type>& expected_types) {
    return op::Predicate(
        [=](const Output<Node>& output) -> bool {
            const auto& output_type = output.get_element_type();
            return std::any_of(expected_types.begin(), expected_types.end(), [=](element::Type type) {
                return type == output_type;
            });
        },
        "type_matches_any({" + ov::util::join(expected_types) + "})");
}

op::Predicate all_of(const std::vector<std::function<bool(Output<Node>)>>& predicates) {
    return op::Predicate(
        [=](const Output<Node>& output) -> bool {
            for (auto& p : predicates) {
                if (!p(output))
                    return false;
            }
            return true;
        },
        "all_of(...)");
}

namespace {
bool ends_with(std::string_view str, std::string_view suffix) {
    return str.size() >= suffix.size() && str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::pair<std::vector<std::pair<int64_t, std::string>>, int64_t> parse_notation(const std::string& notation) {
    auto s = notation;
    s.erase(remove_if(s.begin(), s.end(), isspace), s.end());
    const std::unordered_set<std::string> scalar = {"", "[]", "()"}, dynamic = {"?", "..."};
    if (scalar.count(s))
        return {{}, 0};
    if (dynamic.count(s))
        return {{}, -1};
    const std::unordered_set<char> brackets_to_remove = {'[', ']', '(', ')'};
    s.erase(remove_if(s.begin(),
                      s.end(),
                      [&](char c) {
                          return brackets_to_remove.count(c);
                      }),
            s.end());

    std::vector<std::string> parsed;
    size_t pos = 0, pos_next;
    std::string token;
    while ((pos_next = s.find(',', pos)) != std::string::npos) {
        token = s.substr(pos, pos_next - pos);
        parsed.push_back(token);
        pos = pos_next + 1;
    }
    // collect whole string if no delimiter is found
    token = s.substr(pos, pos_next);
    parsed.push_back(token);

    std::vector<std::pair<int64_t, std::string>> idx_to_name;

    bool ellipsis_visited = false;
    for (int64_t i = 0; i < static_cast<int64_t>(parsed.size()); ++i) {
        auto dimension = parsed[i];
        if (dimension.find("...") != std::string::npos) {
            OPENVINO_ASSERT(!ellipsis_visited, "Only one ellipsis is allowed for symbolic predicate notation");
            OPENVINO_ASSERT(ends_with(dimension, "..."), "Ellipsis usage is incorrect in symbolic predicate notation");
            ellipsis_visited = true;
            idx_to_name.emplace_back(i, dimension);
            continue;
        }
        idx_to_name.emplace_back((ellipsis_visited ? i - static_cast<int64_t>(parsed.size()) : i), dimension);
    }
    return {idx_to_name, (ellipsis_visited ? -2 : idx_to_name.size())};
}

std::pair<bool, int64_t> str2int(const std::string& str) {
    auto s = str.c_str();
    char* end;
    int64_t l;
    l = strtol(s, &end, 10);
    if (*s == '\0' || *end != '\0')
        return {1, 0};
    return {0, l};
}

struct GroupDetails {
    std::string name;
    int64_t begin = 0, end = 0;

    size_t size() const {
        return static_cast<size_t>(end - begin);
    }
};

}  // namespace

/**
 * @brief Checks if bounds can be propagated on Node output.
 * Shape Notation Rules and Examples:
 * Dimension variants:
 *   - digit -- for static dimension
 *   - string name -- for static or symbolic equivalence check / recording; no spaces or commas in the name
 *   - question mark -- for irrelevant dimensions that don't need recording or checking. Relevant for rank check.
 *   - ellipsis -- any number of dimensions, including no dimensions. Only one ellipsis is allowed. Irrelevant for
 * check / recording
 *   - named ellipsis aka named group of dimensions -  any number of dimensions, including no dimensions. Only one
 * ellipsis is allowed. Relevant for check / recording Shape may or may not be enclosed with brackets -- [] or ().
 *
 * Dimensions are delimited with commas. Spaces are irrelevant.
 *
 * Examples:
 * "[A, 3, C, D]" -- check for rank == 4; A, C, D checked / recorded the Match obj; static dim checked;
 * "[A, 3, ..., D]" -- check for rank >= 3; A, D checked / recorded the Match obj; static dim checked;
 * "[?, D]" -- check for rank == 2; D checked / recorded the Match obj; ? dim -- not checked and not recorded;
 * "[Batch, SequenceLength, ?]" -- check for rank == 3; Batch, SequenceLength checked / recorded the Match obj;
 * "[Batches..., OutputDim]" -- check for rank >= 1; Group of dimensions Batches and dimension OutputDim checked /
 * recorded the Match obj;
 *
 * @param shape_notation  string notation composed by the rules
 * @return Predicate which checks if Output<Node> shape matches the shape_notation
 */
op::Predicate shape_matches(const std::string& shape_notation) {
    auto item = parse_notation(shape_notation);
    const auto& idx_to_name = item.first;
    const auto& rank_restrictions = item.second;
    return op::Predicate(
        [idx_to_name, rank_restrictions](PatternSymbolMap& m, const Output<Node>& output) -> bool {
            const auto& shape = output.get_partial_shape();
            if (rank_restrictions == 0)  // scalar
                return shape.is_static() && shape.size() == 0;
            if (rank_restrictions == -1)  // fully dynamic
                return shape.rank().is_dynamic();
            if (rank_restrictions == -2 && (shape.rank().is_dynamic() || shape.size() + 1 < idx_to_name.size()))
                // minimum rank check; checking shape.size() + 1 because idx_to_name contains a record with group that
                // may match to an empty set of dimensions
                return false;
            if (rank_restrictions > 0 &&
                (shape.rank().is_dynamic() || static_cast<int64_t>(shape.size()) != rank_restrictions))
                return false;
            PatternSymbolMap local_m;
            GroupDetails group;
            for (const auto& [this_dim_idx, name] : idx_to_name) {
                if (name == "?" || name == "...")
                    continue;
                if (ends_with(name, "...")) {  // named group detected
                    group.name = {name.substr(0, name.size() - 3)};
                    group.begin = this_dim_idx;
                    continue;
                }
                if (!group.name.empty() && this_dim_idx < group.end)
                    group.end = this_dim_idx;
                const auto& this_dim = shape[this_dim_idx];
                const auto& [conversion_failed, converted_int] = str2int(name);
                if (conversion_failed) {  // failed the conversion -- this is a name
                    if (m.count(name) || local_m.count(name)) {
                        const auto& recorded_value = m.count(name) ? m.at(name) : local_m.at(name);
                        if (recorded_value.is_dynamic()) {
                            const auto& recorded_symbol = recorded_value.s();
                            if (!ov::symbol::are_equal(recorded_symbol, this_dim.get_symbol()))
                                return false;
                        } else if (recorded_value.is_integer()) {
                            if (this_dim.is_dynamic() || this_dim.get_length() != recorded_value.i())
                                return false;
                        } else {
                            return false;
                        }
                    } else {
                        if (this_dim.is_static())
                            local_m[name] = {static_cast<int64_t>(this_dim.get_length())};
                        else if (auto symbol = this_dim.get_symbol())
                            local_m[name] = {symbol};
                        else
                            return false;
                    }
                } else {  // this_dim is not a name, but an integer
                    if (this_dim.is_dynamic() || this_dim.get_length() != converted_int)
                        return false;
                }
            }

            if (!group.name.empty()) {
                const auto& shape_rank = shape.rank().get_length();
                OPENVINO_ASSERT(group.end <= 0);  // end == 0 means group is placed at the end of the notation
                group.end = group.end + shape_rank;

                if (m.count(group.name) || local_m.count(group.name)) {
                    const auto& recorded_value = m.count(group.name) ? m.at(group.name) : local_m.at(group.name);
                    OPENVINO_ASSERT(recorded_value.is_group(),
                                    "Mixing group and non group symbolic predicate notation");
                    const auto& recorded_group = recorded_value.g();
                    if (recorded_group.size() != group.size())
                        return false;
                    for (size_t i = 0; i < recorded_group.size(); ++i) {
                        const auto& recorded_i = recorded_group[i];
                        const auto& this_dim = shape[static_cast<std::ptrdiff_t>(group.begin + i)];
                        if (recorded_i.is_dynamic()) {
                            const auto& recorded_symbol = recorded_i.s();
                            if (!ov::symbol::are_equal(recorded_symbol, this_dim.get_symbol()))
                                return false;
                        } else if (recorded_i.is_integer()) {
                            if (this_dim.is_dynamic() || this_dim.get_length() != recorded_i.i())
                                return false;
                        } else {
                            return false;
                        }
                    }
                } else {
                    std::vector<PatternSymbolValue> group_value;
                    for (size_t i = 0; i < group.size(); ++i) {
                        const auto& this_dim = shape[static_cast<std::ptrdiff_t>(group.begin + i)];
                        if (this_dim.is_static())
                            group_value.emplace_back(static_cast<int64_t>(this_dim.get_length()));
                        else if (auto symbol = this_dim.get_symbol())
                            group_value.emplace_back(symbol);
                        else
                            return false;
                    }
                    local_m[group.name] = group_value;
                }
            }

            // only write locally collected data to the global map when the match is complete to avoid partially
            // collected data for the case when Predicate::operator|| was used
            m.insert(local_m.begin(), local_m.end());
            return true;
        },
        "shape_matches('" + shape_notation + "')");
}
}  // namespace ov::pass::pattern
