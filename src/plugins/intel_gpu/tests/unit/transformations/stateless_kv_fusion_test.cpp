// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include <openvino/core/model.hpp>
#include <openvino/pass/manager.hpp>
#include "common_test_utils/ov_test_utils.hpp"
#include <transformations/utils/utils.hpp>

#include <plugin/transformations/stateless_kv_fusion.hpp>

#include "intel_gpu/op/sdpa.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sink.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/variadic_split.hpp"
#include "intel_gpu/op/stateless_kv.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace {

std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> make_sequence_lengths(const ov::Output<ov::Node>& seqlens_k) {
    auto seqlens_i64 = std::make_shared<ov::op::v0::Convert>(seqlens_k, ov::element::i64);
    auto one = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto real_seqlens = std::make_shared<ov::op::v1::Add>(seqlens_i64, one);
    auto present_seqlen = std::make_shared<ov::op::v1::Reshape>(real_seqlens, one, false);
    return std::make_pair(real_seqlens->output(0), present_seqlen->output(0));
}

std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> make_past_cur_seqlens(const ov::Output<ov::Node>& present_seqlen,
                                                                            const ov::Output<ov::Node>& query,
                                                                            const ov::Output<ov::Node>& axis) {
    auto query_shape = std::make_shared<ov::op::v3::ShapeOf>(query);
    auto gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    auto current_seqlen = std::make_shared<ov::op::v8::Gather>(query_shape, axis, gather_axis);
    auto past_seqlen = std::make_shared<ov::op::v1::Subtract>(present_seqlen, current_seqlen);
    return {past_seqlen->output(0), current_seqlen->output(0)};
}

ov::Output<ov::Node> make_position_ids(const ov::Output<ov::Node>& past_seqlen, size_t current_seqlen) {
    std::vector<int64_t> position_values(current_seqlen);
    std::iota(position_values.begin(), position_values.end(), 0);
    auto position_base = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{current_seqlen}, position_values);
    return std::make_shared<ov::op::v1::Add>(position_base, past_seqlen);
}

ov::Output<ov::Node> make_position_ids(const ov::Output<ov::Node>& past_seqlen, const ov::Output<ov::Node>& current_seqlen) {
    auto zero = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto one = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto current_seqlen_scalar = std::make_shared<ov::op::v0::Squeeze>(current_seqlen);
    auto position_base = std::make_shared<ov::op::v4::Range>(zero, current_seqlen_scalar, one, ov::element::i64);
    return std::make_shared<ov::op::v1::Add>(position_base, past_seqlen);
}

}  // namespace

// Full static, NPU preferred path, ShapeOf/Range are constant-folded and ScatterUpdate feeds SDPA directly
TEST_F(TransformationTestsF, StatelessKVFusion_Update) {
    static constexpr size_t current_seqlen = 2;
    const ov::PartialShape current_shape{1, 2, current_seqlen, 4};
    const ov::PartialShape past_shape{1, 2, 8, 4};
    const auto order = ov::intel_gpu::op::SDPA::default_order(4);
    const auto make_static_past_seqlen = [](const ov::Output<ov::Node>& present_seqlen) {
        auto neg_current_seqlen = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-static_cast<int64_t>(current_seqlen)});
        return std::make_shared<ov::op::v1::Add>(present_seqlen, neg_current_seqlen)->output(0);
    };

    {
        auto query = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, current_shape);
        auto key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, current_shape);
        auto value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, current_shape);
        auto past_key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, past_shape);
        auto past_value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, past_shape);
        auto seqlens_k = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1, 1});

        const auto [real_seqlen, present_seqlen] = make_sequence_lengths(seqlens_k);
        auto past_seqlen = make_static_past_seqlen(present_seqlen);
        auto position_ids = make_position_ids(past_seqlen, current_seqlen);
        auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
        auto present_key = std::make_shared<ov::op::v3::ScatterUpdate>(past_key, position_ids, key, axis);
        auto present_value = std::make_shared<ov::op::v3::ScatterUpdate>(past_value, position_ids, value, axis);

        auto sdpa = std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{query, present_key, present_value}, true, order, order, order, order);
        model = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(sdpa),
                                                             std::make_shared<ov::op::v0::Result>(present_key),
                                                             std::make_shared<ov::op::v0::Result>(present_value)},
                                            ov::ParameterVector{query, key, value, past_key, past_value, seqlens_k});
        manager.register_pass<StatelessKVFusion>();
        disable_result_friendly_names_check();
    }
    {
        auto query = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, current_shape);
        auto key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, current_shape);
        auto value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, current_shape);
        auto past_key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, past_shape);
        auto past_value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, past_shape);
        auto seqlens_k = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1, 1});

        const auto [real_seqlen, present_seqlen] = make_sequence_lengths(seqlens_k);
        auto past_seqlen = make_static_past_seqlen(present_seqlen);
        auto position_ids = make_position_ids(past_seqlen, current_seqlen);
        auto key_cache = std::make_shared<ov::intel_gpu::op::StatelessKV>(past_key, key, real_seqlen, position_ids, 2, true);
        auto value_cache = std::make_shared<ov::intel_gpu::op::StatelessKV>(past_value, value, real_seqlen, position_ids, 2, true);

        auto sdpa =
            std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{query, key_cache->output(1), value_cache->output(1)}, true, order, order, order, order);
        model_ref = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(sdpa),
                                                                 std::make_shared<ov::op::v0::Result>(key_cache->output(0)),
                                                                 std::make_shared<ov::op::v0::Result>(value_cache->output(0))},
                                                ov::ParameterVector{query, key, value, past_key, past_value, seqlens_k});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
        comparator.enable(FunctionsComparator::CONST_VALUES);
    }
}

// Static pastKV and dynamic Q ScatterUpdate with VariadicSplit towards SDPA
TEST_F(TransformationTestsF, StatelessKVFusion_UpdateSplit) {
    const ov::PartialShape current_shape{1, 2, -1, 4};
    const ov::PartialShape past_shape{1, 2, 8, 4};
    const auto order = ov::intel_gpu::op::SDPA::default_order(4);

    {
        auto query = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, current_shape);
        auto key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, current_shape);
        auto value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, current_shape);
        auto past_key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, past_shape);
        auto past_value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, past_shape);
        auto seqlens_k = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1, 1});

        const auto [real_seqlen, present_seqlen] = make_sequence_lengths(seqlens_k);
        auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
        auto [past_seqlen, current_seqlen] = make_past_cur_seqlens(present_seqlen, query, axis);
        auto position_ids = make_position_ids(past_seqlen, current_seqlen);
        auto present_key = std::make_shared<ov::op::v3::ScatterUpdate>(past_key, position_ids, key, axis);
        auto present_value = std::make_shared<ov::op::v3::ScatterUpdate>(past_value, position_ids, value, axis);
        auto split_tail = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        auto split_lengths = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{present_seqlen, split_tail}, 0);
        auto key_split = std::make_shared<ov::op::v1::VariadicSplit>(present_key, axis, split_lengths);
        auto value_split = std::make_shared<ov::op::v1::VariadicSplit>(present_value, axis, split_lengths);

        auto sdpa =
            std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{query, key_split->output(0), value_split->output(0)}, true, order, order, order, order);
        model = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(sdpa),
                                                             std::make_shared<ov::op::v0::Result>(present_key),
                                                             std::make_shared<ov::op::v0::Result>(present_value)},
                                            ov::ParameterVector{query, key, value, past_key, past_value, seqlens_k});
        manager.register_pass<StatelessKVFusion>();
        disable_result_friendly_names_check();
    }
    {
        auto query = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, current_shape);
        auto key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, current_shape);
        auto value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, current_shape);
        auto past_key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, past_shape);
        auto past_value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, past_shape);
        auto seqlens_k = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1, 1});

        const auto [real_seqlen, present_seqlen] = make_sequence_lengths(seqlens_k);
        auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
        auto [past_seqlen, current_seqlen] = make_past_cur_seqlens(present_seqlen, query, axis);
        auto position_ids = make_position_ids(past_seqlen, current_seqlen);
        auto key_cache = std::make_shared<ov::intel_gpu::op::StatelessKV>(past_key, key, real_seqlen, position_ids, 2, true);
        auto value_cache = std::make_shared<ov::intel_gpu::op::StatelessKV>(past_value, value, real_seqlen, position_ids, 2, true);

        auto sdpa =
            std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{query, key_cache->output(1), value_cache->output(1)}, true, order, order, order, order);
        model_ref = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(sdpa),
                                                                 std::make_shared<ov::op::v0::Result>(key_cache->output(0)),
                                                                 std::make_shared<ov::op::v0::Result>(value_cache->output(0))},
                                                ov::ParameterVector{query, key, value, past_key, past_value, seqlens_k});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
        comparator.enable(FunctionsComparator::CONST_VALUES);
    }
}

// Dynamic pastKV, using Split and Concat towards SDPA
TEST_F(TransformationTestsF, StatelessKVFusion_SplitConcat) {
    const ov::PartialShape cache_shape{1, 2, -1, 4};
    const auto order = ov::intel_gpu::op::SDPA::default_order(4);
    {
        auto query = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto past_key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto past_value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto seqlens_k = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1, 1});

        const auto [real_seqlen, present_seqlen] = make_sequence_lengths(seqlens_k);
        auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
        auto [past_seqlen, current_seqlen] = make_past_cur_seqlens(present_seqlen, query, axis);
        auto start = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        auto step = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        auto past_key_slice = std::make_shared<ov::op::v8::Slice>(past_key, start, past_seqlen, step, axis);
        auto past_value_slice = std::make_shared<ov::op::v8::Slice>(past_value, start, past_seqlen, step, axis);
        auto present_key = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{past_key_slice, key}, 2);
        auto present_value = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{past_value_slice, value}, 2);

        auto sdpa = std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{query, present_key, present_value}, true, order, order, order, order);
        model = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(sdpa),
                                                             std::make_shared<ov::op::v0::Result>(present_key),
                                                             std::make_shared<ov::op::v0::Result>(present_value)},
                                            ov::ParameterVector{query, key, value, past_key, past_value, seqlens_k});
        manager.register_pass<StatelessKVFusion>();
        disable_result_friendly_names_check();
    }
    {
        auto query = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto past_key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto past_value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto seqlens_k = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1, 1});

        const auto [real_seqlen, present_seqlen] = make_sequence_lengths(seqlens_k);
        auto key_cache = std::make_shared<ov::intel_gpu::op::StatelessKV>(past_key, key, real_seqlen, 2, true);
        auto value_cache = std::make_shared<ov::intel_gpu::op::StatelessKV>(past_value, value, real_seqlen, 2, true);

        auto sdpa =
            std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{query, key_cache->output(1), value_cache->output(1)}, true, order, order, order, order);
        model_ref = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(sdpa),
                                                                 std::make_shared<ov::op::v0::Result>(key_cache->output(0)),
                                                                 std::make_shared<ov::op::v0::Result>(value_cache->output(0))},
                                                ov::ParameterVector{query, key, value, past_key, past_value, seqlens_k});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
        comparator.enable(FunctionsComparator::CONST_VALUES);
    }
}

// Dynamic pastKV, using total_seq_len rather than seqs+1
TEST_F(TransformationTestsF, StatelessKVFusion_SplitConcat_TotalSeq) {
    const ov::PartialShape cache_shape{1, 2, -1, 4};
    const auto order = ov::intel_gpu::op::SDPA::default_order(4);
    {
        auto query = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto past_key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto past_value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto total_sequence_length = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});

        auto total_sequence_length_i64 = std::make_shared<ov::op::v0::Convert>(total_sequence_length, ov::element::i64);
        auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
        auto [past_seqlen, current_seqlen] = make_past_cur_seqlens(total_sequence_length_i64, query, axis);
        auto start = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        auto step = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        auto past_key_slice = std::make_shared<ov::op::v8::Slice>(past_key, start, past_seqlen, step, axis);
        auto past_value_slice = std::make_shared<ov::op::v8::Slice>(past_value, start, past_seqlen, step, axis);
        auto present_key = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{past_key_slice, key}, 2);
        auto present_value = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{past_value_slice, value}, 2);

        auto sdpa = std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{query, present_key, present_value}, true, order, order, order, order);
        model = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(sdpa),
                                                             std::make_shared<ov::op::v0::Result>(present_key),
                                                             std::make_shared<ov::op::v0::Result>(present_value)},
                                            ov::ParameterVector{query, key, value, past_key, past_value, total_sequence_length});
        manager.register_pass<StatelessKVFusion>();
        disable_result_friendly_names_check();
    }
    {
        auto query = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto past_key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto past_value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto total_sequence_length = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});

        auto key_cache = std::make_shared<ov::intel_gpu::op::StatelessKV>(past_key, key, total_sequence_length, 2, true);
        auto value_cache = std::make_shared<ov::intel_gpu::op::StatelessKV>(past_value, value, total_sequence_length, 2, true);

        auto sdpa =
            std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{query, key_cache->output(1), value_cache->output(1)}, true, order, order, order, order);
        model_ref = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(sdpa),
                                                                 std::make_shared<ov::op::v0::Result>(key_cache->output(0)),
                                                                 std::make_shared<ov::op::v0::Result>(value_cache->output(0))},
                                                ov::ParameterVector{query, key, value, past_key, past_value, total_sequence_length});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
        comparator.enable(FunctionsComparator::CONST_VALUES);
    }
}

// Full static, multi nodes should share seqk+1
TEST_F(TransformationTestsF, StatelessKVFusion_MultiSDPA_StaticSharedSeqlensK) {
    static constexpr size_t current_seqlen = 2;
    const ov::PartialShape current_shape{1, 2, current_seqlen, 4};
    const ov::PartialShape past_shape{1, 2, 8, 4};
    const auto order = ov::intel_gpu::op::SDPA::default_order(4);
    const auto make_static_past_seqlen = [](const ov::Output<ov::Node>& present_seqlen) {
        auto neg_current_seqlen = ov::op::v0::Constant::create(
            ov::element::i64,
            ov::Shape{1},
            {-static_cast<int64_t>(current_seqlen)});
        return std::make_shared<ov::op::v1::Add>(present_seqlen, neg_current_seqlen)->output(0);
    };

    {
        auto query = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, current_shape);
        auto key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, current_shape);
        auto value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, current_shape);
        auto past_key_0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, past_shape);
        auto past_value_0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, past_shape);
        auto past_key_1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, past_shape);
        auto past_value_1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, past_shape);
        auto seqlens_k = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1, 1});

        const auto [real_seqlen_0, present_seqlen_0] = make_sequence_lengths(seqlens_k);
        auto past_seqlen_0 = make_static_past_seqlen(present_seqlen_0);
        auto position_ids_0 = make_position_ids(past_seqlen_0, current_seqlen);
        auto axis_0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
        auto present_key_0 = std::make_shared<ov::op::v3::ScatterUpdate>(past_key_0, position_ids_0, key, axis_0);
        auto present_value_0 = std::make_shared<ov::op::v3::ScatterUpdate>(past_value_0, position_ids_0, value, axis_0);
        auto sdpa_0 = std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{query, present_key_0, present_value_0}, true, order, order, order, order);

        const auto [real_seqlen_1, present_seqlen_1] = make_sequence_lengths(seqlens_k);
        auto past_seqlen_1 = make_static_past_seqlen(present_seqlen_1);
        auto position_ids_1 = make_position_ids(past_seqlen_1, current_seqlen);
        auto axis_1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
        auto present_key_1 = std::make_shared<ov::op::v3::ScatterUpdate>(past_key_1, position_ids_1, sdpa_0, axis_1);
        auto present_value_1 = std::make_shared<ov::op::v3::ScatterUpdate>(past_value_1, position_ids_1, sdpa_0, axis_1);
        auto sdpa_1 = std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{sdpa_0, present_key_1, present_value_1}, true, order, order, order, order);

        model = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(sdpa_1),
                                                             std::make_shared<ov::op::v0::Result>(present_key_0),
                                                             std::make_shared<ov::op::v0::Result>(present_value_0),
                                                             std::make_shared<ov::op::v0::Result>(present_key_1),
                                                             std::make_shared<ov::op::v0::Result>(present_value_1)},
                                            ov::ParameterVector{query, key, value, past_key_0, past_value_0, past_key_1, past_value_1, seqlens_k});
        manager.register_pass<StatelessKVFusion>();
        disable_result_friendly_names_check();
    }
    {
        auto query = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, current_shape);
        auto key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, current_shape);
        auto value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, current_shape);
        auto past_key_0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, past_shape);
        auto past_value_0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, past_shape);
        auto past_key_1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, past_shape);
        auto past_value_1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, past_shape);
        auto seqlens_k = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1, 1});

        const auto [real_seqlen_0, present_seqlen_0] = make_sequence_lengths(seqlens_k);
        auto past_seqlen_0 = make_static_past_seqlen(present_seqlen_0);
        auto position_ids_0 = make_position_ids(past_seqlen_0, current_seqlen);
        auto key_cache_0 = std::make_shared<ov::intel_gpu::op::StatelessKV>(past_key_0, key, real_seqlen_0, position_ids_0, 2, true);
        auto value_cache_0 = std::make_shared<ov::intel_gpu::op::StatelessKV>(past_value_0, value, real_seqlen_0, position_ids_0, 2, true);
        auto sdpa_0 = std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{query, key_cache_0->output(1), value_cache_0->output(1)},
                                                                true,
                                                                order,
                                                                order,
                                                                order,
                                                                order);

        const auto [real_seqlen_1, present_seqlen_1] = make_sequence_lengths(seqlens_k);
        auto past_seqlen_1 = make_static_past_seqlen(present_seqlen_1);
        auto position_ids_1 = make_position_ids(past_seqlen_1, current_seqlen);
        auto key_cache_1 = std::make_shared<ov::intel_gpu::op::StatelessKV>(past_key_1, sdpa_0, real_seqlen_0, position_ids_1, 2, true);
        auto value_cache_1 = std::make_shared<ov::intel_gpu::op::StatelessKV>(past_value_1, sdpa_0, real_seqlen_0, position_ids_1, 2, true);
        auto sdpa_1 = std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{sdpa_0, key_cache_1->output(1), value_cache_1->output(1)},
                                                                true,
                                                                order,
                                                                order,
                                                                order,
                                                                order);

        model_ref = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(sdpa_1),
                                                                 std::make_shared<ov::op::v0::Result>(key_cache_0->output(0)),
                                                                 std::make_shared<ov::op::v0::Result>(value_cache_0->output(0)),
                                                                 std::make_shared<ov::op::v0::Result>(key_cache_1->output(0)),
                                                                 std::make_shared<ov::op::v0::Result>(value_cache_1->output(0))},
                                                ov::ParameterVector{query, key, value, past_key_0, past_value_0, past_key_1, past_value_1, seqlens_k});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
        comparator.enable(FunctionsComparator::CONST_VALUES);
    }
}

// Dynamic, multi nodes should share seqs+1
TEST_F(TransformationTestsF, StatelessKVFusion_MultiSDPA_SharedSeqlensK) {
    const ov::PartialShape cache_shape{1, 2, -1, 4};
    const auto order = ov::intel_gpu::op::SDPA::default_order(4);
    {
        auto query = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto past_key_0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto past_value_0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto past_key_1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto past_value_1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto seqlens_k = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1, 1});

        const auto [real_seqlen_0, present_seqlen_0] = make_sequence_lengths(seqlens_k);
        auto axis_0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
        auto [past_seqlen_0, current_seqlen_0] = make_past_cur_seqlens(present_seqlen_0, query, axis_0);
        auto start_0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        auto step_0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        auto past_key_slice_0 = std::make_shared<ov::op::v8::Slice>(past_key_0, start_0, past_seqlen_0, step_0, axis_0);
        auto past_value_slice_0 = std::make_shared<ov::op::v8::Slice>(past_value_0, start_0, past_seqlen_0, step_0, axis_0);
        auto present_key_0 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{past_key_slice_0, key}, 2);
        auto present_value_0 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{past_value_slice_0, value}, 2);

        auto sdpa_0 = std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{query, present_key_0, present_value_0}, true, order, order, order, order);

        const auto [real_seqlen_1, present_seqlen_1] = make_sequence_lengths(seqlens_k);
        auto axis_1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
        auto [past_seqlen_1, current_seqlen_1] = make_past_cur_seqlens(present_seqlen_1, sdpa_0, axis_1);
        auto start_1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        auto step_1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        auto past_key_slice_1 = std::make_shared<ov::op::v8::Slice>(past_key_1, start_1, past_seqlen_1, step_1, axis_1);
        auto past_value_slice_1 = std::make_shared<ov::op::v8::Slice>(past_value_1, start_1, past_seqlen_1, step_1, axis_1);
        auto present_key_1 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{past_key_slice_1, sdpa_0}, 2);
        auto present_value_1 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{past_value_slice_1, sdpa_0}, 2);
        auto sdpa_1 = std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{sdpa_0, present_key_1, present_value_1}, true, order, order, order, order);

        model = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(sdpa_1),
                                                             std::make_shared<ov::op::v0::Result>(present_key_0),
                                                             std::make_shared<ov::op::v0::Result>(present_value_0),
                                                             std::make_shared<ov::op::v0::Result>(present_key_1),
                                                             std::make_shared<ov::op::v0::Result>(present_value_1)},
                                            ov::ParameterVector{query, key, value, past_key_0, past_value_0, past_key_1, past_value_1, seqlens_k});
        manager.register_pass<StatelessKVFusion>();
        disable_result_friendly_names_check();
    }
    {
        auto query = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto past_key_0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto past_value_0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto past_key_1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto past_value_1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto seqlens_k = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1, 1});

        const auto [real_seqlen, present_seqlen] = make_sequence_lengths(seqlens_k);
        auto key_cache_0 = std::make_shared<ov::intel_gpu::op::StatelessKV>(past_key_0, key, real_seqlen, 2, true);
        auto value_cache_0 = std::make_shared<ov::intel_gpu::op::StatelessKV>(past_value_0, value, real_seqlen, 2, true);

        auto sdpa_0 = std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{query, key_cache_0->output(1), value_cache_0->output(1)},
                                                                true,
                                                                order,
                                                                order,
                                                                order,
                                                                order);

        auto key_cache_1 = std::make_shared<ov::intel_gpu::op::StatelessKV>(past_key_1, sdpa_0, real_seqlen, 2, true);
        auto value_cache_1 = std::make_shared<ov::intel_gpu::op::StatelessKV>(past_value_1, sdpa_0, real_seqlen, 2, true);
        auto sdpa_1 = std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{sdpa_0, key_cache_1->output(1), value_cache_1->output(1)},
                                                                true,
                                                                order,
                                                                order,
                                                                order,
                                                                order);

        model_ref = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(sdpa_1),
                                                                 std::make_shared<ov::op::v0::Result>(key_cache_0->output(0)),
                                                                 std::make_shared<ov::op::v0::Result>(value_cache_0->output(0)),
                                                                 std::make_shared<ov::op::v0::Result>(key_cache_1->output(0)),
                                                                 std::make_shared<ov::op::v0::Result>(value_cache_1->output(0))},
                                                ov::ParameterVector{query, key, value, past_key_0, past_value_0, past_key_1, past_value_1, seqlens_k});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
        comparator.enable(FunctionsComparator::CONST_VALUES);
    }
}

// Dynamic, multi nodes should share seqk+1
TEST_F(TransformationTestsF, StatelessKVFusion_MultiSDPA_SharedPresentLen) {
    const ov::PartialShape cache_shape{1, 2, -1, 4};
    const auto order = ov::intel_gpu::op::SDPA::default_order(4);
    {
        auto query = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto past_key_0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto past_value_0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto past_key_1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto past_value_1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto seqlens_k = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1, 1});

        const auto [real_seqlen, present_seqlen] = make_sequence_lengths(seqlens_k);
        auto axis_0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
        auto [past_seqlen_0, current_seqlen_0] = make_past_cur_seqlens(present_seqlen, query, axis_0);
        auto start_0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        auto step_0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        auto past_key_slice_0 = std::make_shared<ov::op::v8::Slice>(past_key_0, start_0, past_seqlen_0, step_0, axis_0);
        auto past_value_slice_0 = std::make_shared<ov::op::v8::Slice>(past_value_0, start_0, past_seqlen_0, step_0, axis_0);
        auto present_key_0 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{past_key_slice_0, key}, 2);
        auto present_value_0 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{past_value_slice_0, value}, 2);

        auto sdpa_0 = std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{query, present_key_0, present_value_0}, true, order, order, order, order);

        auto axis_1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
        auto [past_seqlen_1, current_seqlen_1] = make_past_cur_seqlens(present_seqlen, sdpa_0, axis_1);
        auto start_1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        auto step_1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        auto past_key_slice_1 = std::make_shared<ov::op::v8::Slice>(past_key_1, start_1, past_seqlen_1, step_1, axis_1);
        auto past_value_slice_1 = std::make_shared<ov::op::v8::Slice>(past_value_1, start_1, past_seqlen_1, step_1, axis_1);
        auto present_key_1 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{past_key_slice_1, sdpa_0}, 2);
        auto present_value_1 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{past_value_slice_1, sdpa_0}, 2);
        auto sdpa_1 = std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{sdpa_0, present_key_1, present_value_1}, true, order, order, order, order);

        model = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(sdpa_1),
                                                             std::make_shared<ov::op::v0::Result>(present_key_0),
                                                             std::make_shared<ov::op::v0::Result>(present_value_0),
                                                             std::make_shared<ov::op::v0::Result>(present_key_1),
                                                             std::make_shared<ov::op::v0::Result>(present_value_1)},
                                            ov::ParameterVector{query, key, value, past_key_0, past_value_0, past_key_1, past_value_1, seqlens_k});
        manager.register_pass<StatelessKVFusion>();
        disable_result_friendly_names_check();
    }
    {
        auto query = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto past_key_0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto past_value_0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto past_key_1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto past_value_1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto seqlens_k = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1, 1});

        const auto [real_seqlen, present_seqlen] = make_sequence_lengths(seqlens_k);
        auto key_cache_0 = std::make_shared<ov::intel_gpu::op::StatelessKV>(past_key_0, key, real_seqlen, 2, true);
        auto value_cache_0 = std::make_shared<ov::intel_gpu::op::StatelessKV>(past_value_0, value, real_seqlen, 2, true);

        auto sdpa_0 = std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{query, key_cache_0->output(1), value_cache_0->output(1)},
                                                                true,
                                                                order,
                                                                order,
                                                                order,
                                                                order);

        auto key_cache_1 = std::make_shared<ov::intel_gpu::op::StatelessKV>(past_key_1, sdpa_0, real_seqlen, 2, true);
        auto value_cache_1 = std::make_shared<ov::intel_gpu::op::StatelessKV>(past_value_1, sdpa_0, real_seqlen, 2, true);
        auto sdpa_1 = std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{sdpa_0, key_cache_1->output(1), value_cache_1->output(1)},
                                                                true,
                                                                order,
                                                                order,
                                                                order,
                                                                order);

        model_ref = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(sdpa_1),
                                                                 std::make_shared<ov::op::v0::Result>(key_cache_0->output(0)),
                                                                 std::make_shared<ov::op::v0::Result>(value_cache_0->output(0)),
                                                                 std::make_shared<ov::op::v0::Result>(key_cache_1->output(0)),
                                                                 std::make_shared<ov::op::v0::Result>(value_cache_1->output(0))},
                                                ov::ParameterVector{query, key, value, past_key_0, past_value_0, past_key_1, past_value_1, seqlens_k});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
        comparator.enable(FunctionsComparator::CONST_VALUES);
    }
}

// Dynamic with total_seq_len, multi nodes should share total_seq_len
TEST_F(TransformationTestsF, StatelessKVFusion_MultiSDPA_SharedTotalSeq) {
    const ov::PartialShape cache_shape{1, 2, -1, 4};
    const auto order = ov::intel_gpu::op::SDPA::default_order(4);
    {
        auto query = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto past_key_0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto past_value_0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto past_key_1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto past_value_1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto total_sequence_length = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});

        auto total_sequence_length_i64_0 = std::make_shared<ov::op::v0::Convert>(total_sequence_length, ov::element::i64);
        auto axis_0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
        auto [past_seqlen_0, current_seqlen_0] = make_past_cur_seqlens(total_sequence_length_i64_0, query, axis_0);
        auto start_0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        auto step_0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        auto past_key_slice_0 = std::make_shared<ov::op::v8::Slice>(past_key_0, start_0, past_seqlen_0, step_0, axis_0);
        auto past_value_slice_0 = std::make_shared<ov::op::v8::Slice>(past_value_0, start_0, past_seqlen_0, step_0, axis_0);
        auto present_key_0 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{past_key_slice_0, key}, 2);
        auto present_value_0 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{past_value_slice_0, value}, 2);

        auto sdpa_0 = std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{query, present_key_0, present_value_0}, true, order, order, order, order);

        auto total_sequence_length_i64_1 = std::make_shared<ov::op::v0::Convert>(total_sequence_length, ov::element::i64);
        auto axis_1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
        auto [past_seqlen_1, current_seqlen_1] = make_past_cur_seqlens(total_sequence_length_i64_1, sdpa_0, axis_1);
        auto start_1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        auto step_1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        auto past_key_slice_1 = std::make_shared<ov::op::v8::Slice>(past_key_1, start_1, past_seqlen_1, step_1, axis_1);
        auto past_value_slice_1 = std::make_shared<ov::op::v8::Slice>(past_value_1, start_1, past_seqlen_1, step_1, axis_1);
        auto present_key_1 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{past_key_slice_1, sdpa_0}, 2);
        auto present_value_1 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{past_value_slice_1, sdpa_0}, 2);
        auto sdpa_1 = std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{sdpa_0, present_key_1, present_value_1}, true, order, order, order, order);

        model = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(sdpa_1),
                                                             std::make_shared<ov::op::v0::Result>(present_key_0),
                                                             std::make_shared<ov::op::v0::Result>(present_value_0),
                                                             std::make_shared<ov::op::v0::Result>(present_key_1),
                                                             std::make_shared<ov::op::v0::Result>(present_value_1)},
                                            ov::ParameterVector{query, key, value, past_key_0, past_value_0, past_key_1, past_value_1, total_sequence_length});
        manager.register_pass<StatelessKVFusion>();
        disable_result_friendly_names_check();
    }
    {
        auto query = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto past_key_0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto past_value_0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto past_key_1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto past_value_1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        auto total_sequence_length = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});

        auto key_cache_0 = std::make_shared<ov::intel_gpu::op::StatelessKV>(past_key_0, key, total_sequence_length, 2, true);
        auto value_cache_0 = std::make_shared<ov::intel_gpu::op::StatelessKV>(past_value_0, value, total_sequence_length, 2, true);

        auto sdpa_0 = std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{query, key_cache_0->output(1), value_cache_0->output(1)},
                                                                true,
                                                                order,
                                                                order,
                                                                order,
                                                                order);

        auto key_cache_1 = std::make_shared<ov::intel_gpu::op::StatelessKV>(past_key_1, sdpa_0, total_sequence_length, 2, true);
        auto value_cache_1 = std::make_shared<ov::intel_gpu::op::StatelessKV>(past_value_1, sdpa_0, total_sequence_length, 2, true);
        auto sdpa_1 = std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{sdpa_0, key_cache_1->output(1), value_cache_1->output(1)},
                                                                true,
                                                                order,
                                                                order,
                                                                order,
                                                                order);

        model_ref =
            std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(sdpa_1),
                                                         std::make_shared<ov::op::v0::Result>(key_cache_0->output(0)),
                                                         std::make_shared<ov::op::v0::Result>(value_cache_0->output(0)),
                                                         std::make_shared<ov::op::v0::Result>(key_cache_1->output(0)),
                                                         std::make_shared<ov::op::v0::Result>(value_cache_1->output(0))},
                                        ov::ParameterVector{query, key, value, past_key_0, past_value_0, past_key_1, past_value_1, total_sequence_length});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
        comparator.enable(FunctionsComparator::CONST_VALUES);
    }
}
