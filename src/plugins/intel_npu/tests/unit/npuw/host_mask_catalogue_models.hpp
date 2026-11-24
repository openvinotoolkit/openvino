// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>

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

namespace ov::test::npuw::host_mask_catalogue {

using Value = ov::Output<ov::Node>;

enum class Family { F1, F2, F3, F4a, F4b, F5 };
enum class Layout { Direct, ContextSlice, QuerySuffix, Phi4, Phi4RC2, Gemma3n };

inline auto scalar(int64_t value, const ov::element::Type& type = ov::element::i64) {
    return ov::op::v0::Constant::create(type, ov::Shape{}, {value});
}

inline auto vector(std::initializer_list<int64_t> values, const ov::element::Type& type = ov::element::i64) {
    return ov::op::v0::Constant::create(type, ov::Shape{values.size()}, values);
}

inline auto unsqueeze(const Value& value, int64_t axis) {
    return std::make_shared<ov::op::v0::Unsqueeze>(value, scalar(axis));
}

inline auto as_vector(const Value& value) {
    return std::make_shared<ov::op::v1::Reshape>(value, vector({-1}), false);
}

inline auto dim(const Value& value,
                int64_t axis,
                bool keep_dim = false,
                const ov::element::Type& type = ov::element::i64) {
    auto shape = std::make_shared<ov::op::v3::ShapeOf>(value, type);
    return std::make_shared<ov::op::v8::Gather>(shape, keep_dim ? vector({axis}) : scalar(axis), scalar(0), 0);
}

inline auto range(const Value& start, const Value& stop, const ov::element::Type& type = ov::element::i64) {
    return std::make_shared<ov::op::v4::Range>(start, stop, scalar(1, ov::element::i32), type);
}

inline auto broadcast(const Value& value, const Value& shape, bool bidirectional = false) {
    return std::make_shared<ov::op::v3::Broadcast>(
        value,
        shape,
        bidirectional ? ov::op::BroadcastType::BIDIRECTIONAL : ov::op::BroadcastType::NUMPY);
}

inline auto concat(const ov::OutputVector& values, int64_t axis = 0) {
    return std::make_shared<ov::op::v0::Concat>(values, axis);
}

inline auto context_slice(const Value& value, const Value& end) {
    return std::make_shared<ov::op::v8::Slice>(value, vector({0}), as_vector(end), vector({1}), vector({3}));
}

struct Mask {
    Value root;
    Value columns;
    Value rows;
    Value history;
    Value context;
    std::shared_ptr<ov::op::v1::Select> select;
    std::shared_ptr<ov::op::v13::BitwiseAnd> predicates;
    std::shared_ptr<ov::op::v1::Greater> position_bound;
    std::shared_ptr<ov::op::v1::Greater> row_bound;
    std::shared_ptr<ov::op::v13::BitwiseOr> window_or;
    std::shared_ptr<ov::op::v1::Split> index_split;
    std::shared_ptr<ov::op::v1::Add> lower_start;
};

// These builders retain the exported shape expressions and expanded batch Gather.
inline Mask make_mask(Family family,
                      const Value& row,
                      const Value& past,
                      const Value& sequence,
                      const Value& positions,
                      int64_t window = 262144,
                      Layout layout = Layout::Direct) {
    using namespace ov;
    auto h = dim(past, 2);
    Value q_raw = dim(sequence, 1);
    Value qv = dim(sequence, 1, true);
    if (family == Family::F4a || family == Family::F5) {
        q_raw = std::make_shared<op::v0::Squeeze>(qv);
    } else if (family == Family::F4b || (family == Family::F1 && layout == Layout::Direct)) {
        qv = as_vector(q_raw);
    }
    Value q = q_raw;
    Value batch = dim(sequence, 0, true);
    if (layout == Layout::Phi4 || layout == Layout::Phi4RC2 || family == Family::F3) {
        qv = as_vector(q_raw);
        q = std::make_shared<op::v0::Squeeze>(qv, scalar(0, element::i32));
    }
    if (layout == Layout::Phi4RC2 || family == Family::F3) {
        batch = as_vector(dim(sequence, 0));
    }
    auto total = std::make_shared<op::v1::Add>(h, q);
    auto cv = dim(row, 1, true);
    auto cscalar = dim(row, 1);
    auto target = concat({batch, vector({1}), qv, as_vector(total)});
    auto zero = op::v0::Constant::create(element::f32, Shape{}, {0});
    auto lowest = op::v0::Constant::create(element::f32, Shape{}, {std::numeric_limits<float>::lowest()});
    auto one4 = op::v0::Constant::create(element::f32, Shape{1, 1, 1, 1}, {1});
    Mask result;
    result.history = h;
    result.context = total;

    if (family == Family::F1 || family == Family::F2 || family == Family::F3) {
        auto c = unsqueeze(unsqueeze(unsqueeze(range(scalar(0, element::i32), total), 0), 1), 2);
        Value row_range = range(h, std::make_shared<op::v1::Add>(h, q_raw));
        if (family == Family::F1 && layout == Layout::Direct) {
            auto first = std::make_shared<op::v8::Gather>(row_range, scalar(0), scalar(0), 0);
            row_range = std::make_shared<op::v1::Add>(
                range(scalar(0, element::i32),
                      std::make_shared<op::v0::Squeeze>(as_vector(q_raw), scalar(0, element::i32))),
                first);
        } else if (family == Family::F1 && layout == Layout::Gemma3n) {
            row_range = std::make_shared<op::v1::Add>(range(scalar(0, element::i32), q_raw), h);
        }
        auto r = unsqueeze(unsqueeze(unsqueeze(row_range, 0), 1), 3);
        result.columns = c;
        result.rows = r;
        Value predicate = std::make_shared<op::v1::LessEqual>(c, r);
        auto a = std::make_shared<op::v0::Convert>(row, element::boolean);
        if (family == Family::F2) {
            auto p = unsqueeze(unsqueeze(positions, 0), 3);
            auto offset = op::v0::Constant::create(element::i64, Shape{1, 1, 1, 1}, {-window});
            result.position_bound = std::make_shared<op::v1::Greater>(c, std::make_shared<op::v1::Add>(p, offset));
            result.row_bound = std::make_shared<op::v1::Greater>(c, std::make_shared<op::v1::Add>(r, offset));
            result.window_or =
                std::make_shared<op::v13::BitwiseOr>(result.row_bound, std::make_shared<op::v1::Less>(c, h));
            auto query_padding = std::make_shared<op::v8::Slice>(a,
                                                                 as_vector(h),
                                                                 as_vector(std::make_shared<op::v1::Add>(h, q_raw)),
                                                                 vector({1}),
                                                                 vector({1}));
            predicate = std::make_shared<op::v13::BitwiseAnd>(
                std::make_shared<op::v13::BitwiseAnd>(
                    std::make_shared<op::v13::BitwiseAnd>(result.position_bound, predicate),
                    result.window_or),
                std::make_shared<op::v1::Reshape>(query_padding, vector({-1, 1}), false));
        }
        Value indexed = a;
        if (family == Family::F3) {
            auto padding_pairs = std::make_shared<op::v1::Reshape>(
                concat({vector({0}), unsqueeze(std::make_shared<op::v1::Subtract>(total, q_raw), 0)}),
                vector({-1, 2}, element::i32),
                false);
            auto split = std::make_shared<op::v1::Split>(padding_pairs, scalar(1, element::i32), 2);
            auto begin =
                concat({vector({0}), std::make_shared<op::v0::Squeeze>(split->output(0), scalar(1, element::i32))});
            auto end =
                concat({vector({0}), std::make_shared<op::v0::Squeeze>(split->output(1), scalar(1, element::i32))});
            indexed = std::make_shared<op::v12::Pad>(a,
                                                     begin,
                                                     end,
                                                     op::v0::Constant::create(element::boolean, Shape{}, {false}),
                                                     op::PadMode::CONSTANT);
        }
        auto shape = std::make_shared<op::v3::ShapeOf>(indexed, element::i32);
        auto flat_shape = concat(
            {std::make_shared<op::v1::ReduceProd>(shape, scalar(0, element::i32), true), vector({-1}, element::i32)});
        auto flat = std::make_shared<op::v1::Reshape>(indexed, flat_shape, true);
        Value batch_scalar = dim(sequence, 0);
        if (layout == Layout::Phi4 || (family == Family::F1 && layout == Layout::Direct)) {
            batch_scalar = std::make_shared<op::v0::Squeeze>(batch);
        }
        auto bi = unsqueeze(unsqueeze(unsqueeze(range(scalar(0, element::i32), batch_scalar), 1), 2), 3);
        result.index_split = std::make_shared<op::v1::Split>(shape, scalar(0, element::i32), 2);
        auto idx = std::make_shared<op::v1::Add>(
            std::make_shared<op::v0::Convert>(c, element::i32),
            std::make_shared<op::v1::Multiply>(std::make_shared<op::v0::Convert>(bi, element::i32),
                                               result.index_split->output(1)));
        auto gather = std::make_shared<op::v8::Gather>(flat, idx, scalar(0, element::i32), 0);
        auto key_padding =
            std::make_shared<op::v1::Reshape>(std::make_shared<op::v1::Reshape>(gather, vector({-1}), false),
                                              std::make_shared<op::v3::ShapeOf>(idx, element::i32),
                                              false);
        result.predicates = std::make_shared<op::v13::BitwiseAnd>(
            std::make_shared<op::v13::BitwiseAnd>(op::v0::Constant::create(element::boolean, Shape{}, {true}),
                                                  predicate),
            key_padding);
        result.select = std::make_shared<op::v1::Select>(broadcast(result.predicates, target, true),
                                                         zero,
                                                         op::v0::Constant::create(element::f32, Shape{}, {-65504}));
        result.root = result.select;
    } else if (family == Family::F4a || family == Family::F4b) {
        auto padding_target = concat({batch, vector({1}), qv, cv});
        Value converted =
            std::make_shared<op::v0::Convert>(broadcast(unsqueeze(unsqueeze(row, 1), 2), padding_target, true),
                                              element::f32);
        if (family == Family::F4b) {
            converted = std::make_shared<op::v1::Multiply>(converted, one4);
        }
        auto inverted = std::make_shared<op::v1::Subtract>(one4, converted);
        auto encoded = std::make_shared<op::v1::Select>(std::make_shared<op::v0::Convert>(inverted, element::boolean),
                                                        lowest,
                                                        inverted);
        auto relative = range(scalar(0, element::i32), q_raw);
        auto query_column = std::make_shared<op::v1::Reshape>(std::make_shared<op::v1::Add>(relative, vector({1})),
                                                              vector({0, 1}),
                                                              true);
        auto triangle = std::make_shared<op::v1::Select>(std::make_shared<op::v1::Less>(relative, query_column),
                                                         zero,
                                                         broadcast(lowest, broadcast(qv, vector({2}, element::i32))));
        auto past_width = std::make_shared<op::v1::Subtract>(std::make_shared<op::v1::Add>(q_raw, h), q_raw);
        auto zeros = broadcast(zero, concat({qv, unsqueeze(past_width, 0)}));
        Value causal = concat({zeros, triangle}, -1);
        if (family == Family::F4b) {
            auto qi32 = std::make_shared<op::v0::Convert>(q_raw, element::i32);
            auto lower_columns =
                unsqueeze(range(scalar(0, element::i32),
                                std::make_shared<op::v1::Add>(dim(zeros, 1, false, element::i32), qi32),
                                element::i32),
                          0);
            result.lower_start = std::make_shared<op::v1::Add>(past_width, scalar(-262145));
            auto start = std::make_shared<op::v0::Convert>(result.lower_start, element::i32);
            auto lower_rows = unsqueeze(range(start, std::make_shared<op::v1::Add>(qi32, start), element::i32), 1);
            auto lower =
                std::make_shared<op::v1::Select>(std::make_shared<op::v1::LessEqual>(lower_columns, lower_rows),
                                                 broadcast(op::v0::Constant::create(element::boolean, Shape{}, {true}),
                                                           std::make_shared<op::v3::ShapeOf>(causal, element::i32)),
                                                 op::v0::Constant::create(element::boolean, Shape{}, {false}));
            causal = std::make_shared<op::v1::Select>(lower, lowest, causal);
        }
        auto causal_target =
            concat({batch, vector({1}), qv, std::make_shared<op::v1::Add>(qv, unsqueeze(past_width, 0))});
        result.root =
            std::make_shared<op::v1::Select>(std::make_shared<op::v0::Convert>(encoded, element::boolean),
                                             lowest,
                                             broadcast(unsqueeze(unsqueeze(causal, 0), 1), causal_target, true));
        if (family == Family::F4b) {
            result.root = context_slice(result.root, total);
        }
    } else {
        auto c = unsqueeze(
            range(scalar(0, element::i32), std::make_shared<op::v0::Convert>(cscalar, element::i32), element::i32),
            0);
        auto start = std::make_shared<op::v0::Convert>(
            std::make_shared<op::v1::Add>(std::make_shared<op::v1::Subtract>(cscalar, q_raw), scalar(1)),
            element::i32);
        auto r = unsqueeze(
            range(start,
                  std::make_shared<op::v1::Add>(std::make_shared<op::v0::Convert>(q_raw, element::i32), start),
                  element::i32),
            1);
        auto causal = std::make_shared<op::v1::Select>(std::make_shared<op::v1::GreaterEqual>(c, r),
                                                       broadcast(lowest, concat({qv, cv})),
                                                       zero);
        auto converted = std::make_shared<op::v0::Convert>(unsqueeze(unsqueeze(row, 1), 2), element::f32);
        auto padding = std::make_shared<op::v1::Multiply>(
            std::make_shared<op::v1::Subtract>(one4, std::make_shared<op::v1::Multiply>(converted, one4)),
            op::v0::Constant::create(element::f32, Shape{1, 1, 1, 1}, {std::numeric_limits<float>::lowest()}));
        auto sum = std::make_shared<op::v1::Add>(unsqueeze(unsqueeze(causal, 0), 1), padding);
        auto shape = std::make_shared<op::v1::Maximum>(std::make_shared<op::v3::ShapeOf>(sum),
                                                       concat({batch, vector({1}), vector({1}), vector({1})}));
        result.root = context_slice(broadcast(sum, shape), total);
    }
    if (layout == Layout::ContextSlice || layout == Layout::Phi4RC2) {
        result.root = context_slice(result.root, total);
    } else if (layout == Layout::QuerySuffix) {
        result.root = std::make_shared<op::v8::Slice>(result.root, vector({7}), vector({8}), vector({1}), vector({2}));
    }
    result.root.get_node_shared_ptr()->set_friendly_name("old_full_mask");
    return result;
}

}  // namespace ov::test::npuw::host_mask_catalogue
