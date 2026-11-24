// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "host_mask_templates.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <vector>

#include "openvino/op/add.hpp"
#include "openvino/op/bitwise_and.hpp"
#include "openvino/op/bitwise_or.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"

namespace ov::npuw::detail {

namespace {

class HostMaskTemplates {
    using Value = ov::Output<ov::Node>;
    using Dims = std::vector<int64_t>;

    std::shared_ptr<ov::op::v0::Parameter> m_row;
    int64_t m_context;
    int64_t m_query = 0;
    int64_t m_history = 0;
    int64_t m_kind;
    std::map<Value, std::optional<Dims>> m_shapes;

    template <class T>
    static std::shared_ptr<T> op(const Value& value) {
        return value.get_index() == 0 ? ov::as_type_ptr<T>(value.get_node_shared_ptr()) : nullptr;
    }

    static bool shape(const Value& value, const ov::Shape& expected) {
        return value.get_partial_shape().is_static() && value.get_shape() == expected;
    }

    static bool integer(const ov::element::Type& type) {
        return type == ov::element::i32 || type == ov::element::i64;
    }

    static bool constant(const Value& value, const Dims& expected) {
        auto node = op<ov::op::v0::Constant>(value);
        return node && integer(node->get_element_type()) && ov::shape_size(node->get_shape()) == expected.size() &&
               node->cast_vector<int64_t>() == expected;
    }

    static std::optional<double> number(const Value& value) {
        auto node = op<ov::op::v0::Constant>(value);
        if (!node || !ov::shape_size(node->get_shape()) || !node->get_all_data_elements_bitwise_identical()) {
            return std::nullopt;
        }
        return node->cast_vector<double>(1)[0];
    }

    static bool number(const Value& value, double expected) {
        const auto scalar = number(value);
        return scalar && *scalar == expected;
    }

    static std::optional<double> masked(const Value& value, bool finite = false) {
        if (value.get_element_type() != ov::element::f32 && value.get_element_type() != ov::element::f16) {
            return std::nullopt;
        }
        const auto scalar = number(value);
        return scalar && *scalar <= -1e4 && (!finite || std::isfinite(*scalar)) ? scalar : std::nullopt;
    }

    template <class T>
    static std::shared_ptr<T> binary(const Value& value) {
        auto node = op<T>(value);
        return node && node->get_autob().m_type == ov::op::AutoBroadcastType::NUMPY ? node : nullptr;
    }

    static bool shape_of(const Value& value, const Value& source) {
        auto node = op<ov::op::v3::ShapeOf>(value);
        return node && node->input_value(0) == source;
    }

    static bool shape_control_layout(const Value& value) {
        if (!value.get_partial_shape().is_static()) {
            return false;
        }
        const auto& dimensions = value.get_shape();
        return dimensions.size() <= 1 || (dimensions.size() == 2 && dimensions[0] == 1);
    }

    // Only small shape-control expressions are bound here. Mask values and predicates are never evaluated.
    std::optional<Dims> shape_dims(const Value& value, size_t depth = 0) {
        if (depth > 64 || !integer(value.get_element_type()) || !shape_control_layout(value) ||
            ov::shape_size(value.get_shape()) > 8) {
            return std::nullopt;
        }
        const auto found = m_shapes.find(value);
        if (found != m_shapes.end()) {
            return found->second;
        }
        auto input = [&](size_t index) {
            return shape_dims(value.get_node()->input_value(index), depth + 1);
        };
        std::optional<Dims> result;
        if (auto node = op<ov::op::v0::Constant>(value)) {
            result = node->cast_vector<int64_t>();
        } else if (auto node = op<ov::op::v3::ShapeOf>(value)) {
            if (node->get_input_partial_shape(0).is_static()) {
                const auto dimensions = node->get_input_shape(0);
                result = Dims(dimensions.begin(), dimensions.end());
            }
        } else if (auto node = op<ov::op::v0::Convert>(value)) {
            if (integer(node->get_input_element_type(0))) {
                result = input(0);
            }
        } else if (auto node = op<ov::op::v8::Gather>(value)) {
            if (!node->get_input_partial_shape(0).is_static() || !node->get_input_partial_shape(1).is_static() ||
                node->get_input_shape(0).size() != 1 || node->get_input_shape(1).size() > 1 ||
                node->get_batch_dims() != 0 || !constant(node->input_value(2), {0}) ||
                !shape(value, node->get_input_shape(1))) {
                return std::nullopt;
            }
            auto data = input(0), indices = input(1);
            if (data && indices && ov::shape_size(value.get_shape()) == indices->size()) {
                Dims selected;
                for (auto index : *indices) {
                    if (index < 0 || static_cast<size_t>(index) >= data->size()) {
                        selected.clear();
                        break;
                    }
                    selected.push_back((*data)[index]);
                }
                if (selected.size() == indices->size()) {
                    result = selected;
                }
            }
        } else if (auto node = op<ov::op::v1::Reshape>(value)) {
            if (shape_control_layout(node->input_value(0)) && node->get_input_partial_shape(1).is_static() &&
                node->get_input_shape(1).size() == 1 &&
                ov::shape_size(node->get_input_shape(1)) == value.get_shape().size() &&
                ov::shape_size(node->get_input_shape(0)) == ov::shape_size(node->get_output_shape(0)) &&
                op<ov::op::v0::Constant>(node->input_value(1))) {
                result = input(0);
            }
        } else if (op<ov::op::v0::Squeeze>(value) || op<ov::op::v0::Unsqueeze>(value)) {
            auto node = value.get_node();
            if (shape_control_layout(node->input_value(0)) &&
                ov::shape_size(node->get_input_shape(0)) == ov::shape_size(value.get_shape()) &&
                (node->get_input_size() == 1 ||
                 (op<ov::op::v0::Constant>(node->input_value(1)) && node->get_input_shape(1).size() <= 1))) {
                result = input(0);
            }
        } else if (auto node = op<ov::op::v0::Concat>(value)) {
            if (node->get_axis() == 0 && value.get_shape().size() == 1) {
                Dims joined;
                bool valid = true;
                for (const auto& part : node->input_values()) {
                    if (!part.get_partial_shape().is_static() || part.get_shape().size() != 1) {
                        valid = false;
                        break;
                    }
                    auto dimensions = shape_dims(part, depth + 1);
                    if (!dimensions) {
                        valid = false;
                        break;
                    }
                    joined.insert(joined.end(), dimensions->begin(), dimensions->end());
                }
                if (valid) {
                    result = joined;
                }
            }
        } else if (auto node = ov::as_type_ptr<ov::op::v1::Split>(value.get_node_shared_ptr())) {
            auto data = input(0);
            if (!node->get_input_partial_shape(0).is_static()) {
                return std::nullopt;
            }
            const auto dimensions = node->get_input_shape(0);
            if (data && !dimensions.empty() && dimensions.size() <= 2 &&
                (dimensions.size() == 1 || dimensions[0] == 1) &&
                constant(node->input_value(1), {static_cast<int64_t>(dimensions.size() - 1)}) &&
                node->get_num_splits() == dimensions.back() && value.get_index() < data->size() &&
                ov::shape_size(value.get_shape()) == 1) {
                result = Dims{(*data)[value.get_index()]};
            }
        } else if (auto node = op<ov::op::v1::ReduceProd>(value)) {
            auto dimensions = input(0);
            if (dimensions && op<ov::op::v3::ShapeOf>(node->input_value(0)) && node->get_keep_dims() &&
                constant(node->input_value(1), {0}) && node->get_input_shape(0).size() == 1 && shape(value, {1})) {
                int64_t product = 1;
                for (auto dimension : *dimensions) {
                    if (dimension < 0 || dimension > std::numeric_limits<int32_t>::max() ||
                        (dimension && product > std::numeric_limits<int32_t>::max() / dimension)) {
                        return std::nullopt;
                    }
                    product *= dimension;
                }
                result = Dims{product};
            }
        } else if (auto node = op<ov::op::v3::Broadcast>(value)) {
            auto dimensions = input(0), target = input(1);
            if (dimensions && target && dimensions->size() == 1 && *target == Dims{2} &&
                node->get_input_shape(0).size() <= 1 && node->get_input_shape(1).size() == 1 && shape(value, {2}) &&
                node->get_broadcast_spec().m_type == ov::op::BroadcastType::NUMPY) {
                result = Dims(2, dimensions->front());
            }
        } else if (binary<ov::op::v1::Add>(value) || binary<ov::op::v1::Subtract>(value)) {
            auto left = input(0), right = input(1);
            if (left && right && left->size() == 1 && right->size() == 1 &&
                left->front() >= -std::numeric_limits<int32_t>::max() &&
                left->front() <= std::numeric_limits<int32_t>::max() &&
                right->front() >= -std::numeric_limits<int32_t>::max() &&
                right->front() <= std::numeric_limits<int32_t>::max()) {
                result = Dims{left->front() + (op<ov::op::v1::Add>(value) ? right->front() : -right->front())};
            }
        }
        if (result && result->size() != ov::shape_size(value.get_shape())) {
            result = std::nullopt;
        }
        if (result && value.get_element_type() == ov::element::i32 &&
            std::any_of(result->begin(), result->end(), [](int64_t dimension) {
                return dimension < std::numeric_limits<int32_t>::min() ||
                       dimension > std::numeric_limits<int32_t>::max();
            })) {
            result = std::nullopt;
        }
        m_shapes.emplace(value, result);
        return result;
    }

    bool dimensions(const Value& value, const Dims& expected) {
        const auto actual = shape_dims(value);
        return actual && *actual == expected;
    }

    static std::optional<Value> unsqueeze(const Value& value, int64_t axis) {
        auto node = op<ov::op::v0::Unsqueeze>(value);
        return node && constant(node->input_value(1), {axis}) ? std::optional<Value>(node->input_value(0))
                                                              : std::nullopt;
    }

    static std::optional<Value> convert(const Value& value, const ov::element::Type& type) {
        auto node = op<ov::op::v0::Convert>(value);
        return node && value.get_element_type() == type ? std::optional<Value>(node->input_value(0)) : std::nullopt;
    }

    bool range(const Value& value, int64_t start, int64_t count) {
        auto node = op<ov::op::v4::Range>(value);
        return node && integer(value.get_element_type()) && shape(value, {static_cast<size_t>(count)}) &&
               dimensions(node->input_value(0), {start}) && dimensions(node->input_value(1), {start + count}) &&
               constant(node->input_value(2), {1});
    }

    bool columns(const Value& value) {
        auto a = unsqueeze(value, 2);
        auto b = a ? unsqueeze(*a, 1) : std::nullopt;
        auto c = b ? unsqueeze(*b, 0) : std::nullopt;
        return c && value.get_element_type() == ov::element::i64 && range(*c, 0, m_context);
    }

    bool rows(const Value& value) {
        auto a = unsqueeze(value, 3);
        auto b = a ? unsqueeze(*a, 1) : std::nullopt;
        auto c = b ? unsqueeze(*b, 0) : std::nullopt;
        if (!c || value.get_element_type() != ov::element::i64) {
            return false;
        }
        if (range(*c, m_history, m_query)) {
            return true;
        }
        auto add = binary<ov::op::v1::Add>(*c);
        if (add && range(add->input_value(0), 0, m_query) && dimensions(add->input_value(1), {m_history})) {
            return true;
        }
        auto first = add ? op<ov::op::v8::Gather>(add->input_value(1)) : nullptr;
        return first && range(add->input_value(0), 0, m_query) && range(first->input_value(0), m_history, m_query) &&
               constant(first->input_value(1), {0}) && constant(first->input_value(2), {0}) &&
               first->get_batch_dims() == 0;
    }

    bool positions(const Value& value) {
        auto a = unsqueeze(value, 3);
        auto b = a ? unsqueeze(*a, 0) : std::nullopt;
        auto parameter = b ? op<ov::op::v0::Parameter>(*b) : nullptr;
        return parameter && shape(*b, {1, static_cast<size_t>(m_query)}) && b->get_element_type() == ov::element::i64 &&
               (parameter->get_friendly_name() == "position_ids" || b->get_names().count("position_ids"));
    }

    bool batch_index(const Value& value) {
        auto a = unsqueeze(value, 3);
        auto b = a ? unsqueeze(*a, 2) : std::nullopt;
        auto c = b ? unsqueeze(*b, 1) : std::nullopt;
        return c && range(*c, 0, 1);
    }

    bool row_boolean(const Value& value) {
        auto source = convert(value, ov::element::boolean);
        return source && *source == m_row->output(0);
    }

    bool row_layout(const Value& value) {
        auto a = unsqueeze(value, 2);
        auto b = a ? unsqueeze(*a, 1) : std::nullopt;
        return b && *b == m_row->output(0);
    }

    bool padded_row(const Value& value) {
        auto pad = op<ov::op::v12::Pad>(value);
        return pad && pad->get_input_size() == 4 && pad->get_pad_mode() == ov::op::PadMode::CONSTANT &&
               row_boolean(pad->input_value(0)) && dimensions(pad->input_value(1), {0, 0}) &&
               dimensions(pad->input_value(2), {0, m_history}) &&
               pad->input_value(3).get_element_type() == ov::element::boolean && number(pad->input_value(3), 0);
    }

    bool key_padding(const Value& value, bool padded) {
        auto outer = op<ov::op::v1::Reshape>(value);
        auto inner = outer ? op<ov::op::v1::Reshape>(outer->input_value(0)) : nullptr;
        auto gather = inner ? op<ov::op::v8::Gather>(inner->input_value(0)) : nullptr;
        if (!gather || outer->get_special_zero() || inner->get_special_zero() ||
            !constant(inner->input_value(1), {-1}) || gather->get_batch_dims() != 0 ||
            !constant(gather->input_value(2), {0})) {
            return false;
        }
        const auto index = gather->input_value(1);
        auto add = binary<ov::op::v1::Add>(index);
        auto flat = op<ov::op::v1::Reshape>(gather->input_value(0));
        auto multiply = add ? binary<ov::op::v1::Multiply>(add->input_value(1)) : nullptr;
        auto col = add ? convert(add->input_value(0), ov::element::i32) : std::nullopt;
        auto batch = multiply ? convert(multiply->input_value(0), ov::element::i32) : std::nullopt;
        auto width = multiply ? ov::as_type_ptr<ov::op::v1::Split>(multiply->get_input_node_shared_ptr(1)) : nullptr;
        if (!flat || !flat->get_special_zero() || !col || !columns(*col) || !batch || !batch_index(*batch) || !width ||
            multiply->input_value(1).get_index() != 1 || width->get_num_splits() != 2 ||
            !constant(width->input_value(1), {0}) || !shape_of(width->input_value(0), flat->input_value(0)) ||
            !shape_of(outer->input_value(1), index)) {
            return false;
        }
        auto pattern = op<ov::op::v0::Concat>(flat->input_value(1));
        auto product =
            pattern && pattern->get_input_size() == 2 ? op<ov::op::v1::ReduceProd>(pattern->input_value(0)) : nullptr;
        return product && pattern->get_axis() == 0 && product->get_keep_dims() &&
               shape_of(product->input_value(0), flat->input_value(0)) && constant(product->input_value(1), {0}) &&
               constant(pattern->input_value(1), {-1}) &&
               (padded ? padded_row(flat->input_value(0)) : row_boolean(flat->input_value(0))) &&
               shape(value, {1, 1, 1, static_cast<size_t>(m_context)});
    }

    bool query_padding(const Value& value) {
        auto reshape = op<ov::op::v1::Reshape>(value);
        auto slice = reshape ? op<ov::op::v8::Slice>(reshape->input_value(0)) : nullptr;
        return slice && slice->get_input_size() == 5 && !reshape->get_special_zero() &&
               constant(reshape->input_value(1), {-1, 1}) && row_boolean(slice->input_value(0)) &&
               dimensions(slice->input_value(1), {m_history}) && dimensions(slice->input_value(2), {m_context}) &&
               constant(slice->input_value(3), {1}) && constant(slice->input_value(4), {1});
    }

    bool causal(const Value& value) {
        auto node = binary<ov::op::v1::LessEqual>(value);
        return node && columns(node->input_value(0)) && rows(node->input_value(1));
    }

    std::optional<Value> broadcast(const Value& value, const Dims& target, ov::op::BroadcastType mode) {
        auto node = op<ov::op::v3::Broadcast>(value);
        return node && node->get_broadcast_spec().m_type == mode && dimensions(node->input_value(1), target)
                   ? std::optional<Value>(node->input_value(0))
                   : std::nullopt;
    }

    bool window_bound(const Value& value, bool position) {
        auto greater = binary<ov::op::v1::Greater>(value);
        auto add = greater ? binary<ov::op::v1::Add>(greater->input_value(1)) : nullptr;
        return add && m_kind > m_context && columns(greater->input_value(0)) &&
               (position ? positions(add->input_value(0)) : rows(add->input_value(0))) &&
               shape(add->input_value(1), {1, 1, 1, 1}) && add->get_input_element_type(1) == ov::element::i64 &&
               constant(add->input_value(1), {-m_kind});
    }

    bool window_predicate(const Value& value) {
        auto query = binary<ov::op::v13::BitwiseAnd>(value);
        auto group = query ? binary<ov::op::v13::BitwiseAnd>(query->input_value(0)) : nullptr;
        auto bounds = group ? binary<ov::op::v13::BitwiseAnd>(group->input_value(0)) : nullptr;
        auto alternative = group ? binary<ov::op::v13::BitwiseOr>(group->input_value(1)) : nullptr;
        auto past = alternative ? binary<ov::op::v1::Less>(alternative->input_value(1)) : nullptr;
        return past && bounds && query_padding(query->input_value(1)) && window_bound(bounds->input_value(0), true) &&
               causal(bounds->input_value(1)) && window_bound(alternative->input_value(0), false) &&
               columns(past->input_value(0)) && dimensions(past->input_value(1), {m_history});
    }

    std::optional<Value> select_predicate(const Value& value) {
        auto select = binary<ov::op::v1::Select>(value);
        if (!select || !number(select->input_value(1), 0) || !masked(select->input_value(2))) {
            return std::nullopt;
        }
        return broadcast(select->input_value(0), {1, 1, m_query, m_context}, ov::op::BroadcastType::BIDIRECTIONAL);
    }

    std::optional<Value> bitwise_predicate(const Value& value, bool padded) {
        auto root = select_predicate(value);
        auto padding = root ? binary<ov::op::v13::BitwiseAnd>(*root) : nullptr;
        auto initial = padding ? binary<ov::op::v13::BitwiseAnd>(padding->input_value(0)) : nullptr;
        if (!initial || !number(initial->input_value(0), 1) ||
            initial->get_input_element_type(0) != ov::element::boolean ||
            !key_padding(padding->input_value(1), padded)) {
            return std::nullopt;
        }
        return initial->input_value(1);
    }

    bool match_f1(const Value& value) {
        auto predicate = bitwise_predicate(value, false);
        return predicate && causal(*predicate);
    }

    bool match_f2(const Value& value) {
        auto predicate = bitwise_predicate(value, false);
        return predicate && window_predicate(*predicate);
    }

    bool match_f3(const Value& value) {
        auto predicate = bitwise_predicate(value, true);
        return predicate && causal(*predicate);
    }

    bool legacy_triangle(const Value& value, double mask) {
        auto select = binary<ov::op::v1::Select>(value);
        auto less = select ? binary<ov::op::v1::Less>(select->input_value(0)) : nullptr;
        auto view = less ? op<ov::op::v1::Reshape>(less->input_value(1)) : nullptr;
        auto add = view ? binary<ov::op::v1::Add>(view->input_value(0)) : nullptr;
        auto fill =
            select ? broadcast(select->input_value(2), {m_query, m_query}, ov::op::BroadcastType::NUMPY) : std::nullopt;
        return add && fill && range(less->input_value(0), 0, m_query) && add->input_value(0) == less->input_value(0) &&
               number(add->input_value(1), 1) && view->get_special_zero() && constant(view->input_value(1), {0, 1}) &&
               number(select->input_value(1), 0) && number(*fill, mask);
    }

    bool legacy_concat(const Value& value, double mask) {
        auto concat = op<ov::op::v0::Concat>(value);
        if (!concat || concat->get_input_size() != 2 || concat->get_axis() != -1) {
            return false;
        }
        auto zeros = broadcast(concat->input_value(0), {m_query, m_history}, ov::op::BroadcastType::NUMPY);
        return zeros && number(*zeros, 0) && legacy_triangle(concat->input_value(1), mask);
    }

    bool legacy_lower_band(const Value& value, double mask) {
        auto select = binary<ov::op::v1::Select>(value);
        auto lower = select ? binary<ov::op::v1::Select>(select->input_value(0)) : nullptr;
        auto compare = lower ? binary<ov::op::v1::LessEqual>(lower->input_value(0)) : nullptr;
        auto col = compare ? unsqueeze(compare->input_value(0), 0) : std::nullopt;
        auto row = compare ? unsqueeze(compare->input_value(1), 1) : std::nullopt;
        auto row_range = row ? op<ov::op::v4::Range>(*row) : nullptr;
        auto start = row_range ? convert(row_range->input_value(0), ov::element::i32) : std::nullopt;
        auto add = start ? binary<ov::op::v1::Add>(*start) : nullptr;
        auto offset = add ? number(add->input_value(1)) : std::nullopt;
        auto ones = lower ? op<ov::op::v3::Broadcast>(lower->input_value(1)) : nullptr;
        if (!select || !col || !row_range || !offset || !ones || !dimensions(add->input_value(0), {m_history})) {
            return false;
        }
        const double displacement = offset.value();
        if (!std::isfinite(displacement) || displacement > -m_context ||
            displacement < std::numeric_limits<int32_t>::min() / 2 || std::trunc(displacement) != displacement) {
            return false;
        }
        const auto first = m_history + static_cast<int64_t>(displacement);
        return col->get_element_type() == ov::element::i32 && row->get_element_type() == ov::element::i32 &&
               range(*col, 0, m_context) && range(*row, first, m_query) &&
               ones->get_broadcast_spec().m_type == ov::op::BroadcastType::NUMPY && number(ones->input_value(0), 1) &&
               ones->get_input_element_type(0) == ov::element::boolean &&
               shape_of(ones->input_value(1), select->input_value(2)) && number(lower->input_value(2), 0) &&
               number(select->input_value(1), mask) && legacy_concat(select->input_value(2), mask);
    }

    bool legacy_padding(const Value& value, double mask, bool unity) {
        auto select = binary<ov::op::v1::Select>(value);
        auto condition = select ? convert(select->input_value(0), ov::element::boolean) : std::nullopt;
        auto inverted = condition ? binary<ov::op::v1::Subtract>(*condition) : nullptr;
        if (!inverted || !number(select->input_value(1), mask) || select->input_value(2) != *condition ||
            !number(inverted->input_value(0), 1)) {
            return false;
        }
        auto converted = inverted->input_value(1);
        if (unity) {
            auto multiply = binary<ov::op::v1::Multiply>(converted);
            if (!multiply || !number(multiply->input_value(1), 1)) {
                return false;
            }
            converted = multiply->input_value(0);
        }
        auto expanded = convert(converted, value.get_element_type());
        auto row = expanded ? broadcast(*expanded, {1, 1, m_query, m_context}, ov::op::BroadcastType::BIDIRECTIONAL)
                            : std::nullopt;
        return row && row_layout(*row);
    }

    bool match_legacy(const Value& value, bool lower_band) {
        auto select = binary<ov::op::v1::Select>(value);
        auto mask = select ? masked(select->input_value(1)) : std::nullopt;
        auto padding = select ? convert(select->input_value(0), ov::element::boolean) : std::nullopt;
        auto expanded =
            select ? broadcast(select->input_value(2), {1, 1, m_query, m_context}, ov::op::BroadcastType::BIDIRECTIONAL)
                   : std::nullopt;
        auto a = expanded ? unsqueeze(*expanded, 1) : std::nullopt;
        auto causal = a ? unsqueeze(*a, 0) : std::nullopt;
        return mask && padding && causal && legacy_padding(*padding, *mask, lower_band) &&
               (lower_band ? legacy_lower_band(*causal, *mask) : legacy_concat(*causal, *mask));
    }

    bool match_f4a(const Value& value) {
        return match_legacy(value, false);
    }

    bool match_f4b(const Value& value) {
        return match_legacy(value, true);
    }

    bool arithmetic_padding(const Value& value, double mask) {
        auto multiply = binary<ov::op::v1::Multiply>(value);
        auto inverted = multiply ? binary<ov::op::v1::Subtract>(multiply->input_value(0)) : nullptr;
        auto unity = inverted ? binary<ov::op::v1::Multiply>(inverted->input_value(1)) : nullptr;
        auto row = unity ? convert(unity->input_value(0), value.get_element_type()) : std::nullopt;
        return row && number(multiply->input_value(1), mask) && number(inverted->input_value(0), 1) &&
               number(unity->input_value(1), 1) && row_layout(*row);
    }

    bool match_f5(const Value& value) {
        auto expanded = op<ov::op::v3::Broadcast>(value);
        auto target = expanded ? binary<ov::op::v1::Maximum>(expanded->input_value(1)) : nullptr;
        auto add = expanded ? binary<ov::op::v1::Add>(expanded->input_value(0)) : nullptr;
        if (!target || !add || expanded->get_broadcast_spec().m_type != ov::op::BroadcastType::NUMPY ||
            !shape_of(target->input_value(0), expanded->input_value(0)) ||
            !dimensions(target->input_value(1), {1, 1, 1, 1})) {
            return false;
        }
        auto a = unsqueeze(add->input_value(0), 1);
        auto b = a ? unsqueeze(*a, 0) : std::nullopt;
        auto select = b ? binary<ov::op::v1::Select>(*b) : nullptr;
        auto compare = select ? binary<ov::op::v1::GreaterEqual>(select->input_value(0)) : nullptr;
        auto col = compare ? unsqueeze(compare->input_value(0), 0) : std::nullopt;
        auto row = compare ? unsqueeze(compare->input_value(1), 1) : std::nullopt;
        auto fill = select ? broadcast(select->input_value(1), {m_query, m_context}, ov::op::BroadcastType::NUMPY)
                           : std::nullopt;
        auto mask = fill ? masked(*fill, true) : std::nullopt;
        return mask && col && row && col->get_element_type() == ov::element::i32 &&
               row->get_element_type() == ov::element::i32 && range(*col, 0, m_context) &&
               range(*row, m_history + 1, m_query) && number(select->input_value(2), 0) &&
               arithmetic_padding(add->input_value(1), *mask);
    }

public:
    HostMaskTemplates(const std::shared_ptr<ov::op::v0::Parameter>& row, int64_t kind)
        : m_row(row),
          m_context(static_cast<int64_t>(row->get_shape()[1])),
          m_kind(kind) {}

    bool matches(Value value, const ov::PartialShape& query_shape, int64_t& f2_window) {
        f2_window = 0;
        if ((value.get_element_type() != ov::element::f16 && value.get_element_type() != ov::element::f32) ||
            !value.get_partial_shape().is_static() || value.get_shape().size() != 4 || !query_shape.is_static() ||
            query_shape.size() != 4 || value.get_shape()[0] != 1 || value.get_shape()[1] != 1 ||
            value.get_shape()[3] != static_cast<size_t>(m_context) ||
            value.get_shape()[2] != static_cast<size_t>(query_shape[2].get_length()) || m_context <= 0 ||
            m_context > std::numeric_limits<int32_t>::max() / 2) {
            return false;
        }
        m_query = static_cast<int64_t>(value.get_shape()[2]);
        if (auto suffix = op<ov::op::v8::Slice>(value)) {
            if (suffix->get_input_size() != 5) {
                return false;
            }
            const auto& original = suffix->get_input_partial_shape(0);
            if (original.is_static() && original.size() == 4 && constant(suffix->input_value(4), {2})) {
                const auto count = original[2].get_length();
                if (m_query != 1 || !constant(suffix->input_value(1), {count - 1}) ||
                    !constant(suffix->input_value(2), {count}) || !constant(suffix->input_value(3), {1})) {
                    return false;
                }
                m_query = count;
                value = suffix->input_value(0);
            }
        }
        if (m_query <= 0 || m_query > m_context) {
            return false;
        }
        m_history = m_context - m_query;
        if (auto slice = op<ov::op::v8::Slice>(value)) {
            if (slice->get_input_size() != 5 || !constant(slice->input_value(1), {0}) ||
                !dimensions(slice->input_value(2), {m_context}) || !constant(slice->input_value(3), {1}) ||
                !constant(slice->input_value(4), {3}) ||
                !shape(slice->input_value(0), {1, 1, static_cast<size_t>(m_query), static_cast<size_t>(m_context)})) {
                return false;
            }
            value = slice->input_value(0);
        }
        if (match_f1(value)) {
            return true;
        }
        if (match_f2(value)) {
            f2_window = m_kind;
            return true;
        }
        return match_f3(value) || match_f4a(value) || match_f4b(value) || match_f5(value);
    }
};

}  // namespace

bool match_host_mask_template(const ov::Output<ov::Node>& mask,
                              const std::shared_ptr<ov::op::v0::Parameter>& row,
                              int64_t kind,
                              const ov::PartialShape& query_shape,
                              int64_t& f2_window) {
    return HostMaskTemplates(row, kind).matches(mask, query_shape, f2_window);
}

}  // namespace ov::npuw::detail
