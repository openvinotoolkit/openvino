// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/pattern/op/pattern.hpp"

#include <algorithm>
#include <optional>
#include <regex>

#include "openvino/core/log_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/log.hpp"

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

#define ACCESSOR(type)                                                                \
    void on_adapter(const std::string& name, ValueAccessor<type>& adapter) override { \
        if (m_expected_attrs.count(name) != 0) {                                      \
            match(name, {adapter.get()});                                             \
        }                                                                             \
    };

#define ACCESSOR_V(type) ACCESSOR(type) ACCESSOR(std::vector<type>)

class AttributeMatchingVisitor : public ov::AttributeVisitor {
public:
    explicit AttributeMatchingVisitor(const Attributes& expected_attrs)
        : ov::AttributeVisitor(),
          m_expected_attrs{expected_attrs} {
        OPENVINO_ASSERT(!expected_attrs.empty(), "Please remove trivial attribute matching check");
        for (const auto& [name, expected_value] : expected_attrs)
            m_matched_attributes[name] = false;
    }

    ACCESSOR(bool)
    ACCESSOR_V(std::string)
    ACCESSOR_V(int8_t)
    ACCESSOR_V(int16_t)
    ACCESSOR_V(int32_t)
    ACCESSOR_V(int64_t)
    ACCESSOR_V(uint8_t)
    ACCESSOR_V(uint16_t)
    ACCESSOR_V(uint32_t)
    ACCESSOR_V(uint64_t)
    ACCESSOR_V(float)
    ACCESSOR_V(double)

    void on_adapter(const std::string& name, ValueAccessor<void>& adapter) override {
        OPENVINO_ASSERT(m_expected_attrs.count(name) == 0, "Can not compare void");
    };
    void on_adapter(const std::string& name, ValueAccessor<void*>& adapter) override {
        OPENVINO_ASSERT(m_expected_attrs.count(name) == 0, "Can not compare void*");
    };
    void on_adapter(const std::string& name, ValueAccessor<std::shared_ptr<ov::Model>>& adapter) override {
        OPENVINO_THROW_NOT_IMPLEMENTED("Can not compare models");
    };

    void match(const std::string& name, const ov::Any& node_attribute) {
        if (m_expected_attrs.count(name)) {
            try {
                const auto& attribute_type_id = node_attribute.type_info();

                auto expected_attribute = m_expected_attrs.at(name);
                const auto& expected_type_id = expected_attribute.type_info();

                if (attribute_type_id != expected_type_id) {  // type conversions for developer convenience
                    if (attribute_type_id == typeid(int64_t) && expected_type_id == typeid(int))
                        expected_attribute = ov::Any(static_cast<int64_t>(expected_attribute.as<int>()));
                    if (attribute_type_id == typeid(std::string) && expected_type_id == typeid(const char*))
                        expected_attribute = ov::Any(std::string(expected_attribute.as<const char*>()));
                    if (attribute_type_id == typeid(bool) && expected_type_id == typeid(int))
                        expected_attribute = ov::Any(static_cast<bool>(expected_attribute.as<int>()));
                    if (attribute_type_id == typeid(double) && expected_type_id == typeid(int))
                        expected_attribute = ov::Any(static_cast<double>(expected_attribute.as<int>()));
                    if (attribute_type_id == typeid(double) && expected_type_id == typeid(float))
                        expected_attribute = ov::Any(static_cast<double>(expected_attribute.as<float>()));
                }

                if (node_attribute.type_info() != expected_attribute.type_info())
                    OPENVINO_LOG_PATTERN1(name,
                                          node_attribute.type_info().name(),
                                          expected_attribute.type_info().name());
                bool status = node_attribute == expected_attribute;
                OPENVINO_LOG_PATTERN2(status, name, node_attribute, expected_attribute);
                m_matched_attributes[name] = status;
            } catch (...) {
                OPENVINO_LOG_PATTERN3(name);
                m_matched_attributes[name] = false;
            }
        } else {
            OPENVINO_LOG_PATTERN4(name);
        }
    }

    bool get_match_status() const {
        // TODO: provide additional logging in case of failure; however the on_adapter method is already logging a lot
        return std::all_of(m_matched_attributes.begin(), m_matched_attributes.end(), [](const auto& name_to_val) {
            return name_to_val.second == true;
        });
    }

private:
    const Attributes& m_expected_attrs;
    std::unordered_map<std::string, bool> m_matched_attributes;
};
}  // namespace

op::Predicate attrs_match(const Attributes& expected_attrs) {
    std::stringstream ss;
    ss << "{ ";
    bool first = true;
    for (const auto& [key, value] : expected_attrs) {
        if (!first)
            ss << ", ";
        first = false;
        ss << key << ": ";
        value.print(ss);
    }
    ss << " }";
    return op::Predicate(
        [expected_attrs](PatternSymbolMap&, const Output<Node>& output) -> bool {
            const auto& node = output.get_node_shared_ptr();
            AttributeMatchingVisitor visitor(expected_attrs);
            node->visit_attributes(visitor);
            return visitor.get_match_status();
        },
        "attrs_match(" + ss.str() + ")");
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
    while ((pos_next = s.find(',', pos)) != std::string::npos) {
        parsed.push_back(s.substr(pos, pos_next - pos));
        pos = pos_next + 1;
    }
    // collect whole string if no delimiter is found
    parsed.push_back(s.substr(pos, pos_next));

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
        auto to_implace = ellipsis_visited ? i - static_cast<int64_t>(parsed.size()) : i;
        idx_to_name.emplace_back(to_implace, dimension);
    }
    return {idx_to_name, (ellipsis_visited ? -2 : idx_to_name.size())};
}

std::optional<int64_t> str2int(const std::string& str) {
    auto s = str.c_str();
    char* end;
    int64_t l;
    l = strtol(s, &end, 10);
    if (*s == '\0' || *end != '\0')
        return {};
    return {l};
}

std::optional<double> str2double(const std::string& str) {
    auto s = str.c_str();
    char* end;
    double d;
    d = strtod(s, &end);
    if (*s == '\0' || *end != '\0')
        return {};
    return {d};
}

struct GroupDetails {
    std::string name;
    int64_t begin = 0, end = 0;

    size_t size() const {
        return static_cast<size_t>(end - begin);
    }
};

const PatternSymbolValue& get_element(const std::vector<PatternSymbolValue>& values, int64_t idx) {
    auto size = static_cast<int64_t>(values.size());
    if (idx < 0)
        idx += size;
    OPENVINO_ASSERT(idx < size, "Unexpected index");
    return values[static_cast<size_t>(idx)];
}

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
    const auto& position_to_name = item.first;
    const auto& rank_restrictions = item.second;
    return op::Predicate(
        [position_to_name, rank_restrictions](PatternSymbolMap& m, const Output<Node>& output) -> bool {
            const auto& shape = output.get_partial_shape();
            if (rank_restrictions == 0)  // scalar
                return shape.is_static() && shape.size() == 0;
            if (rank_restrictions == -1)  // fully dynamic
                return shape.rank().is_dynamic();
            if (rank_restrictions == -2 && (shape.rank().is_dynamic() || shape.size() + 1 < position_to_name.size()))
                // minimum rank check; checking shape.size() + 1 because position_to_name contains a record with group
                // that may match to an empty set of dimensions
                return false;
            if (rank_restrictions > 0 &&
                (shape.rank().is_dynamic() || static_cast<int64_t>(shape.size()) != rank_restrictions))
                return false;
            PatternSymbolMap local_m;
            GroupDetails group;
            for (const auto& [position, expected_as_string] : position_to_name) {
                if (!group.name.empty() && position < group.end)
                    // placement is intentional -- record end of the group even if ? is placed after the group
                    // after epsilon all positions are negative (python style)
                    group.end = position;
                if (expected_as_string == "?" || expected_as_string == "...")
                    continue;
                if (ends_with(expected_as_string, "...")) {  // named group detected
                    group.name = {expected_as_string.substr(0, expected_as_string.size() - 3)};
                    group.begin = position;
                    continue;
                }

                const auto& actual_dim = shape[position];
                const auto& actual_value = actual_dim.is_static() ? PatternSymbolValue(actual_dim.get_length())
                                                                  : PatternSymbolValue(actual_dim.get_symbol());
                const auto& converted_int = str2int(expected_as_string);
                if (!converted_int) {  // failed the conversion -- this is a name
                    const auto& name = expected_as_string;
                    if (m.count(name) || local_m.count(name)) {
                        const auto& recorded_value = m.count(name) ? m.at(name) : local_m.at(name);
                        if (actual_value != recorded_value)
                            return false;
                    } else {
                        if (actual_dim.is_static() || actual_dim.get_symbol() != nullptr)
                            local_m[name] = actual_value;
                        else
                            return false;
                    }
                } else {  // this_dim is not a name, but an integer
                    if (actual_dim.is_dynamic() || actual_dim.get_length() != converted_int.value())
                        return false;
                }
            }

            if (!group.name.empty()) {  // named group functionality used
                OPENVINO_ASSERT(group.end <= 0);
                // after epsilon all positions are negative (python style)
                // end == 0 means group is placed at the end of the notation
                const auto& shape_rank = shape.rank().get_length();
                group.end = group.end + shape_rank;

                if (m.count(group.name) || local_m.count(group.name)) {
                    const auto& recorded_value = m.count(group.name) ? m.at(group.name) : local_m.at(group.name);
                    OPENVINO_ASSERT(recorded_value.is_group(),
                                    "Mixing group and non-group symbolic predicate notation");
                    // group notation -- "Name..." and non-group notation "Name" are mutually exclusive
                    // group and non-group namespace is intentionally shared
                    const auto& recorded_group = recorded_value.g();
                    if (recorded_group.size() != group.size())
                        return false;
                    for (size_t i = 0; i < recorded_group.size(); ++i) {
                        const auto& actual_dim = shape[static_cast<std::ptrdiff_t>(group.begin + i)];
                        if (actual_dim.is_dynamic() && !actual_dim.get_symbol())
                            return false;
                        const auto& actual_value = actual_dim.is_static() ? PatternSymbolValue(actual_dim.get_length())
                                                                          : PatternSymbolValue(actual_dim.get_symbol());
                        if (recorded_group[i] != actual_value)
                            return false;
                    }
                } else {
                    std::vector<PatternSymbolValue> group_value;
                    for (size_t i = 0; i < group.size(); ++i) {
                        const auto& actual_dim = shape[static_cast<std::ptrdiff_t>(group.begin + i)];
                        if (actual_dim.is_dynamic() && !actual_dim.get_symbol())
                            return false;
                        const auto& actual_value = actual_dim.is_static() ? PatternSymbolValue(actual_dim.get_length())
                                                                          : PatternSymbolValue(actual_dim.get_symbol());
                        group_value.emplace_back(actual_value);
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

op::Predicate value_matches(const std::string& value_notation) {
    auto item = parse_notation(value_notation);
    const auto& position_to_name = item.first;
    const auto& element_count_restrictions = item.second;
    return op::Predicate(
        [position_to_name, element_count_restrictions](PatternSymbolMap& m, const Output<Node>& output) -> bool {
            const auto& constant = ov::as_type_ptr<ov::op::v0::Constant>(output.get_node_shared_ptr());
            if (!constant)
                return false;

            const auto& shape = constant->get_shape();
            const auto& element_count = shape_size(shape);
            if (element_count_restrictions == 0)  // empty
                return element_count == 0;
            if (element_count_restrictions == -1)  // fully dynamic (impossible to have dynamic number of elements)
                return false;
            if (element_count_restrictions == -2 && element_count + 1 < position_to_name.size())
                // minimum num element check; checking element_count + 1 because idx_to_name contains a record with
                // group that may match to an empty set of elements
                return false;
            if (element_count_restrictions > 0 && static_cast<int64_t>(element_count) != element_count_restrictions)
                return false;

            const auto& et = constant->get_element_type();
            if (!et.is_integral() && !et.is_real())
                return false;
            bool is_integral = constant->get_element_type().is_integral();
            const auto& values = is_integral ? PatternSymbolValue::make_value_vector(constant->cast_vector<int64_t>())
                                             : PatternSymbolValue::make_value_vector(constant->cast_vector<double>());

            PatternSymbolMap local_m;
            GroupDetails group;
            for (const auto& [position, expected_as_string] : position_to_name) {
                if (!group.name.empty() && position < group.end)
                    // placement is intentional -- record end of the group even if ? is placed after the group
                    // after epsilon all positions are negative (python style)
                    group.end = position;
                if (expected_as_string == "?" || expected_as_string == "...")
                    continue;
                if (ends_with(expected_as_string, "...")) {  // named group detected
                    group.name = {expected_as_string.substr(0, expected_as_string.size() - 3)};
                    group.begin = position;
                    continue;
                }
                const auto& actual_value = get_element(values, position);
                const auto& converted_int = str2int(expected_as_string);
                const auto& converted_double = str2double(expected_as_string);
                if (!converted_int && !converted_double) {  // failed the conversion -- this is a name
                    const auto& name = expected_as_string;
                    if (m.count(name) || local_m.count(name)) {
                        // we have encountered the value under same name -- comparing it with the actual value
                        const auto& recorded_value = m.count(name) ? m.at(name) : local_m.at(name);
                        if (recorded_value != actual_value)
                            return false;
                    } else {
                        OPENVINO_ASSERT(actual_value.is_integer() || actual_value.is_double());
                        local_m[name] = actual_value;
                    }
                } else if (converted_int) {  // comparison to static integer value was requested
                    if (actual_value != PatternSymbolValue(converted_int.value()))
                        return false;
                } else if (converted_double) {  // comparison to static double value was requested
                    if (actual_value != PatternSymbolValue(converted_double.value()))
                        return false;
                }
            }

            if (!group.name.empty()) {  // named group functionality used
                OPENVINO_ASSERT(group.end <= 0);
                // after epsilon all positions are negative (python style)
                // end == 0 means group is placed last in the notation string
                group.end = group.end + element_count;

                if (m.count(group.name) || local_m.count(group.name)) {
                    const auto& recorded_value = m.count(group.name) ? m.at(group.name) : local_m.at(group.name);
                    OPENVINO_ASSERT(recorded_value.is_group(),
                                    "Mixing group and non-group symbolic predicate notation");
                    // group notation -- "Name..." and non-group notation "Name" are mutually exclusive
                    // group and non-group namespace is shared
                    const auto& recorded_group = recorded_value.g();
                    if (recorded_group.size() != group.size())
                        return false;
                    for (size_t i = 0; i < recorded_group.size(); ++i)
                        if (get_element(values, group.begin + i) != recorded_group[i])
                            return false;
                } else {
                    std::vector<PatternSymbolValue> group_value;
                    for (size_t i = 0; i < group.size(); ++i)
                        group_value.emplace_back(get_element(values, group.begin + i));
                    local_m[group.name] = group_value;
                }
            }

            // only write locally collected data to the global map when the match is complete to avoid partially
            // collected data for the case when Predicate::operator|| was used
            m.insert(local_m.begin(), local_m.end());
            return true;
        },
        "value_matches('" + value_notation + "')");
}
}  // namespace ov::pass::pattern
