// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/op/sdpa.hpp"
#include "intel_gpu/op/stateless_kv.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/pass/manager.hpp"
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
#include "plugin/transformations/stateless_kv_fusion.hpp"
#include "transformations/utils/utils.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace {

using SDPAWithPresentKV = std::tuple<std::shared_ptr<ov::intel_gpu::op::SDPA>, ov::Output<ov::Node>, ov::Output<ov::Node>>;

template <size_t N = 1>
std::array<std::shared_ptr<ov::op::v0::Parameter>, 2 * N + 4> make_parameters(const ov::PartialShape& qkv_shape,
                                                                              const ov::PartialShape& kv_shape,
                                                                              bool is_total_sequence_length) {
    std::array<std::shared_ptr<ov::op::v0::Parameter>, 2 * N + 4> parameters{};
    const auto sequence_length_shape = is_total_sequence_length ? ov::PartialShape{1} : ov::PartialShape{1, 1};
    parameters[0] = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, sequence_length_shape);
    parameters[1] = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, qkv_shape);
    parameters[2] = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, qkv_shape);
    parameters[3] = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, qkv_shape);
    for (size_t idx = 4; idx < parameters.size(); ++idx) {
        parameters[idx] = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, kv_shape);
    }
    return parameters;
}

std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> make_sequence_lengths(const ov::Output<ov::Node>& seqlens_k) {
    auto seqlens_i64 = std::make_shared<ov::op::v0::Convert>(seqlens_k, ov::element::i64);
    auto one = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto real_seqlens = std::make_shared<ov::op::v1::Add>(seqlens_i64, one);
    auto present_seqlen = std::make_shared<ov::op::v1::Reshape>(real_seqlens, one, false);
    return std::make_pair(real_seqlens->output(0), present_seqlen->output(0));
}

template <bool Older = false>
std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> make_past_cur_seqlens(const ov::Output<ov::Node>& present_seqlen, const ov::Output<ov::Node>& query) {
    auto query_shape = std::make_shared<ov::op::v3::ShapeOf>(query);
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
    auto gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    using Gather = std::conditional_t<Older, ov::op::v7::Gather, ov::op::v8::Gather>;
    auto current_seqlen = std::make_shared<Gather>(query_shape, axis, gather_axis);
    auto past_seqlen = std::make_shared<ov::op::v1::Subtract>(present_seqlen, current_seqlen);
    return {past_seqlen->output(0), current_seqlen->output(0)};
}

ov::Output<ov::Node> make_position_ids(const ov::Output<ov::Node>& past_seqlen, const ov::Output<ov::Node>& current_seqlen) {
    auto zero = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto one = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto current_seqlen_scalar = std::make_shared<ov::op::v0::Squeeze>(current_seqlen);
    auto position_base = std::make_shared<ov::op::v4::Range>(zero, current_seqlen_scalar, one, ov::element::i64);
    return std::make_shared<ov::op::v1::Add>(position_base, past_seqlen);
}

std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> make_static_posids(const ov::Output<ov::Node>& seqlens_k, size_t current_seqlen) {
    const auto [real_seqlen, present_seqlen] = make_sequence_lengths(seqlens_k);
    auto negative_current_seqlen = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-static_cast<int64_t>(current_seqlen)});
    auto past_seqlen = std::make_shared<ov::op::v1::Add>(present_seqlen, negative_current_seqlen);
    std::vector<int64_t> position_values(current_seqlen);
    std::iota(position_values.begin(), position_values.end(), 0);
    auto position_base = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{current_seqlen}, position_values);
    auto position_ids = std::make_shared<ov::op::v1::Add>(position_base, past_seqlen);
    return {real_seqlen, position_ids};
}

SDPAWithPresentKV build_slice_concat_pattern(const ov::Output<ov::Node>& query,
                                             const ov::Output<ov::Node>& past_key,
                                             const ov::Output<ov::Node>& past_value,
                                             const ov::Output<ov::Node>& key,
                                             const ov::Output<ov::Node>& value,
                                             const ov::Output<ov::Node>& present_seqlen) {
    const auto order = ov::intel_gpu::op::SDPA::default_order(4);
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
    const auto past_seqlen = make_past_cur_seqlens(present_seqlen, query).first;
    auto start = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto step = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto past_key_slice = std::make_shared<ov::op::v8::Slice>(past_key, start, past_seqlen, step, axis);
    auto past_value_slice = std::make_shared<ov::op::v8::Slice>(past_value, start, past_seqlen, step, axis);
    auto present_key = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{past_key_slice, key}, 2);
    auto present_value = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{past_value_slice, value}, 2);
    auto sdpa = std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{query, present_key, present_value}, true, order, order, order, order);
    return {sdpa, present_key, present_value};
}

template <bool Older = false>
SDPAWithPresentKV build_update_split_pattern(const ov::Output<ov::Node>& query,
                                             const ov::Output<ov::Node>& past_key,
                                             const ov::Output<ov::Node>& past_value,
                                             const ov::Output<ov::Node>& key,
                                             const ov::Output<ov::Node>& value,
                                             const ov::Output<ov::Node>& present_seqlen) {
    const auto order = ov::intel_gpu::op::SDPA::default_order(4);
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
    const auto [past_seqlen, current_seqlen] = make_past_cur_seqlens<Older>(present_seqlen, query);
    const auto position_ids = make_position_ids(past_seqlen, current_seqlen);
    ov::Output<ov::Node> present_key = std::make_shared<ov::op::v3::ScatterUpdate>(past_key, position_ids, key, axis);
    ov::Output<ov::Node> present_value = std::make_shared<ov::op::v3::ScatterUpdate>(past_value, position_ids, value, axis);
    auto split_tail = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    auto split_lengths = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{present_seqlen, split_tail}, 0);
    auto key_split = std::make_shared<ov::op::v1::VariadicSplit>(present_key, axis, split_lengths);
    auto value_split = std::make_shared<ov::op::v1::VariadicSplit>(present_value, axis, split_lengths);
    auto sdpa =
        std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{query, key_split->output(0), value_split->output(0)}, true, order, order, order, order);
    return {sdpa, present_key, present_value};
}

SDPAWithPresentKV build_static_update_pattern(const ov::Output<ov::Node>& query,
                                              const ov::Output<ov::Node>& past_key,
                                              const ov::Output<ov::Node>& past_value,
                                              const ov::Output<ov::Node>& key,
                                              const ov::Output<ov::Node>& value,
                                              const ov::Output<ov::Node>& seqlens_k,
                                              size_t current_seqlen) {
    const auto order = ov::intel_gpu::op::SDPA::default_order(4);
    const auto position_ids = make_static_posids(seqlens_k, current_seqlen).second;
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
    auto present_key = std::make_shared<ov::op::v3::ScatterUpdate>(past_key, position_ids, key, axis);
    auto present_value = std::make_shared<ov::op::v3::ScatterUpdate>(past_value, position_ids, value, axis);
    auto sdpa = std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{query, present_key, present_value}, true, order, order, order, order);
    return {sdpa, present_key->output(0), present_value->output(0)};
}

template <typename... Args>
SDPAWithPresentKV build_stateless_kv_sdpa(const ov::Output<ov::Node>& query,
                                          const ov::Output<ov::Node>& past_key,
                                          const ov::Output<ov::Node>& past_value,
                                          const ov::Output<ov::Node>& key,
                                          const ov::Output<ov::Node>& value,
                                          const ov::Output<ov::Node>& seq_len,
                                          const Args&... args) {
    const auto order = ov::intel_gpu::op::SDPA::default_order(4);
    auto key_cache = std::make_shared<ov::intel_gpu::op::StatelessKV>(past_key, key, seq_len, args...);
    auto value_cache = std::make_shared<ov::intel_gpu::op::StatelessKV>(past_value, value, seq_len, args...);
    auto sdpa =
        std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{query, key_cache->output(1), value_cache->output(1)}, true, order, order, order, order);
    return {sdpa, key_cache->output(0), value_cache->output(0)};
}

template <size_t N>
std::shared_ptr<ov::Model> make_model(const std::array<std::shared_ptr<ov::op::v0::Parameter>, N>& parameters, const ov::OutputVector& outputs) {
    ov::ResultVector results;
    results.reserve(outputs.size());
    for (const auto& output : outputs) {
        results.push_back(std::make_shared<ov::op::v0::Result>(output));
    }
    return std::make_shared<ov::Model>(results, ov::ParameterVector(parameters.begin(), parameters.end()));
}

template <size_t N = 1, typename Builder>
void build_stateless_kv_fusion_test(TransformationTestsF* test,
                                    int64_t current_seqlen,
                                    bool is_total_sequence_length,
                                    const Builder& builder,
                                    bool is_dynamic_past = false) {
    const auto current_shape = ov::PartialShape{1, 2, current_seqlen, 4};
    const auto past_shape = is_dynamic_past ? current_shape : ov::PartialShape{1, 2, 8, 4};
    {
        const auto parameters = make_parameters<N>(current_shape, past_shape, is_total_sequence_length);
        test->model = make_model(parameters, builder(false, parameters));
        test->manager.register_pass<StatelessKVFusion>();
        test->disable_result_friendly_names_check();
    }
    {
        const auto parameters = make_parameters<N>(current_shape, past_shape, is_total_sequence_length);
        test->model_ref = make_model(parameters, builder(true, parameters));
        test->comparator.enable(FunctionsComparator::ATTRIBUTES);
        test->comparator.enable(FunctionsComparator::CONST_VALUES);
    }
}

}  // namespace

// Full static, NPU preferred path, ShapeOf/Range are constant-folded and ScatterUpdate feeds SDPA directly
TEST_F(TransformationTestsF, StatelessKVFusion_Update) {
    static constexpr int64_t current_seqlen = 2;
    build_stateless_kv_fusion_test<1>(this, current_seqlen, false, [](bool is_ref, const auto& parameters) {
        const auto& [seqlens_k, query, key, value, past_key, past_value] = parameters;
        if (!is_ref) {
            const auto [sdpa, present_key, present_value] = build_static_update_pattern(query, past_key, past_value, key, value, seqlens_k, current_seqlen);
            return ov::OutputVector{sdpa, present_key, present_value};
        }

        const auto [real_seqlen, position_ids] = make_static_posids(seqlens_k, current_seqlen);
        const auto [sdpa, present_key, present_value] = build_stateless_kv_sdpa(query, past_key, past_value, key, value, real_seqlen, position_ids, 2, true);
        return ov::OutputVector{sdpa, present_key, present_value};
    });
}

// Static pastKV and dynamic Q ScatterUpdate with VariadicSplit towards SDPA
TEST_F(TransformationTestsF, StatelessKVFusion_UpdateSplit) {
    static constexpr int64_t current_seqlen = -1;
    build_stateless_kv_fusion_test<1>(this, current_seqlen, false, [](bool is_ref, const auto& parameters) {
        const auto& [seqlens_k, query, key, value, past_key, past_value] = parameters;
        const auto [real_seqlen, present_seqlen] = make_sequence_lengths(seqlens_k);
        if (!is_ref) {
            const auto [sdpa, present_key, present_value] = build_update_split_pattern<false>(query, past_key, past_value, key, value, present_seqlen);
            return ov::OutputVector{sdpa, present_key, present_value};
        }

        auto [past_seqlen, current_seqlen] = make_past_cur_seqlens(present_seqlen, query);
        auto position_ids = make_position_ids(past_seqlen, current_seqlen);
        const auto [sdpa, present_key, present_value] = build_stateless_kv_sdpa(query, past_key, past_value, key, value, real_seqlen, position_ids, 2, true);
        return ov::OutputVector{sdpa, present_key, present_value};
    });
}

TEST_F(TransformationTestsF, StatelessKVFusion_UpdateSplit_Older) {
    static constexpr int64_t current_seqlen = -1;
    build_stateless_kv_fusion_test<1>(this, current_seqlen, false, [](bool is_ref, const auto& parameters) {
        const auto& [seqlens_k, query, key, value, past_key, past_value] = parameters;
        const auto [real_seqlen, present_seqlen] = make_sequence_lengths(seqlens_k);
        if (!is_ref) {
            const auto [sdpa, present_key, present_value] = build_update_split_pattern<true>(query, past_key, past_value, key, value, present_seqlen);
            return ov::OutputVector{sdpa, present_key, present_value};
        }

        auto [past_seqlen, current_seqlen] = make_past_cur_seqlens<true>(present_seqlen, query);
        auto position_ids = make_position_ids(past_seqlen, current_seqlen);
        const auto [sdpa, present_key, present_value] = build_stateless_kv_sdpa(query, past_key, past_value, key, value, real_seqlen, position_ids, 2, true);
        return ov::OutputVector{sdpa, present_key, present_value};
    });
}

// Dynamic pastKV, using Split and Concat towards SDPA
TEST_F(TransformationTestsF, StatelessKVFusion_SplitConcat) {
    static constexpr int64_t current_seqlen = -1;
    build_stateless_kv_fusion_test<1>(
        this,
        current_seqlen,
        false,
        [](bool is_ref, const auto& parameters) {
            const auto& [seqlens_k, query, key, value, past_key, past_value] = parameters;
            const auto [real_seqlen, present_seqlen] = make_sequence_lengths(seqlens_k);
            if (!is_ref) {
                const auto [sdpa, present_key, present_value] = build_slice_concat_pattern(query, past_key, past_value, key, value, present_seqlen);
                return ov::OutputVector{sdpa, present_key, present_value};
            }

            const auto [sdpa, present_key, present_value] = build_stateless_kv_sdpa(query, past_key, past_value, key, value, real_seqlen, 2, true);
            return ov::OutputVector{sdpa, present_key, present_value};
        },
        true);
}

// Dynamic pastKV, using total_seq_len rather than seqs+1
TEST_F(TransformationTestsF, StatelessKVFusion_SplitConcat_TotalSeq) {
    static constexpr int64_t current_seqlen = -1;
    build_stateless_kv_fusion_test<1>(
        this,
        current_seqlen,
        true,
        [](bool is_ref, const auto& parameters) {
            const auto& [total_sequence_length, query, key, value, past_key, past_value] = parameters;
            if (!is_ref) {
                auto total_sequence_length_i64 = std::make_shared<ov::op::v0::Convert>(total_sequence_length, ov::element::i64);
                const auto [sdpa, present_key, present_value] = build_slice_concat_pattern(query, past_key, past_value, key, value, total_sequence_length_i64);
                return ov::OutputVector{sdpa, present_key, present_value};
            }

            const auto [sdpa, present_key, present_value] = build_stateless_kv_sdpa(query, past_key, past_value, key, value, total_sequence_length, 2, true);
            return ov::OutputVector{sdpa, present_key, present_value};
        },
        true);
}

// Full static, multi nodes should share seqk+1
TEST_F(TransformationTestsF, StatelessKVFusion_MultiSDPA_StaticSharedSeqlensK) {
    static constexpr int64_t current_seqlen = 2;
    build_stateless_kv_fusion_test<2>(this, current_seqlen, false, [](bool is_ref, const auto& parameters) {
        const auto& [seqlens_k, query, key, value, past_key_0, past_value_0, past_key_1, past_value_1] = parameters;
        if (!is_ref) {
            const auto [sdpa_0, present_key_0, present_value_0] =
                build_static_update_pattern(query, past_key_0, past_value_0, key, value, seqlens_k, current_seqlen);
            const auto [sdpa_1, present_key_1, present_value_1] =
                build_static_update_pattern(sdpa_0, past_key_1, past_value_1, sdpa_0, sdpa_0, seqlens_k, current_seqlen);
            return ov::OutputVector{sdpa_1, present_key_0, present_value_0, present_key_1, present_value_1};
        }

        const auto [real_seqlen_0, position_ids_0] = make_static_posids(seqlens_k, current_seqlen);
        const auto [sdpa_0, present_key_0, present_value_0] =
            build_stateless_kv_sdpa(query, past_key_0, past_value_0, key, value, real_seqlen_0, position_ids_0, 2, true);
        const auto [real_seqlen_1, position_ids_1] = make_static_posids(seqlens_k, current_seqlen);
        const auto [sdpa_1, present_key_1, present_value_1] =
            build_stateless_kv_sdpa(sdpa_0, past_key_1, past_value_1, sdpa_0, sdpa_0, real_seqlen_0, position_ids_1, 2, true);
        return ov::OutputVector{sdpa_1, present_key_0, present_value_0, present_key_1, present_value_1};
    });
}

// Dynamic, multi nodes should share seqs+1
TEST_F(TransformationTestsF, StatelessKVFusion_MultiSDPA_SharedSeqlensK) {
    static constexpr int64_t current_seqlen = -1;
    build_stateless_kv_fusion_test<2>(
        this,
        current_seqlen,
        false,
        [](bool is_ref, const auto& parameters) {
            const auto& [seqlens_k, query, key, value, past_key_0, past_value_0, past_key_1, past_value_1] = parameters;
            const auto [real_seqlen_0, present_seqlen_0] = make_sequence_lengths(seqlens_k);
            if (!is_ref) {
                const auto [sdpa_0, present_key_0, present_value_0] = build_slice_concat_pattern(query, past_key_0, past_value_0, key, value, present_seqlen_0);
                const auto [real_seqlen_1, present_seqlen_1] = make_sequence_lengths(seqlens_k);
                const auto [sdpa_1, present_key_1, present_value_1] =
                    build_slice_concat_pattern(sdpa_0, past_key_1, past_value_1, sdpa_0, sdpa_0, present_seqlen_1);
                return ov::OutputVector{sdpa_1, present_key_0, present_value_0, present_key_1, present_value_1};
            }

            const auto [sdpa_0, present_key_0, present_value_0] = build_stateless_kv_sdpa(query, past_key_0, past_value_0, key, value, real_seqlen_0, 2, true);
            const auto [sdpa_1, present_key_1, present_value_1] =
                build_stateless_kv_sdpa(sdpa_0, past_key_1, past_value_1, sdpa_0, sdpa_0, real_seqlen_0, 2, true);
            return ov::OutputVector{sdpa_1, present_key_0, present_value_0, present_key_1, present_value_1};
        },
        true);
}

// Dynamic, multi nodes should share seqk+1
TEST_F(TransformationTestsF, StatelessKVFusion_MultiSDPA_SharedPresentLen) {
    static constexpr int64_t current_seqlen = -1;
    build_stateless_kv_fusion_test<2>(
        this,
        current_seqlen,
        false,
        [](bool is_ref, const auto& parameters) {
            const auto& [seqlens_k, query, key, value, past_key_0, past_value_0, past_key_1, past_value_1] = parameters;
            const auto [real_seqlen, present_seqlen] = make_sequence_lengths(seqlens_k);
            if (!is_ref) {
                const auto [sdpa_0, present_key_0, present_value_0] = build_slice_concat_pattern(query, past_key_0, past_value_0, key, value, present_seqlen);
                const auto [sdpa_1, present_key_1, present_value_1] =
                    build_slice_concat_pattern(sdpa_0, past_key_1, past_value_1, sdpa_0, sdpa_0, present_seqlen);
                return ov::OutputVector{sdpa_1, present_key_0, present_value_0, present_key_1, present_value_1};
            }

            const auto [sdpa_0, present_key_0, present_value_0] = build_stateless_kv_sdpa(query, past_key_0, past_value_0, key, value, real_seqlen, 2, true);
            const auto [sdpa_1, present_key_1, present_value_1] =
                build_stateless_kv_sdpa(sdpa_0, past_key_1, past_value_1, sdpa_0, sdpa_0, real_seqlen, 2, true);
            return ov::OutputVector{sdpa_1, present_key_0, present_value_0, present_key_1, present_value_1};
        },
        true);
}

// Dynamic with total_seq_len, multi nodes should share total_seq_len
TEST_F(TransformationTestsF, StatelessKVFusion_MultiSDPA_SharedTotalSeq) {
    static constexpr int64_t current_seqlen = -1;
    build_stateless_kv_fusion_test<2>(
        this,
        current_seqlen,
        true,
        [](bool is_ref, const auto& parameters) {
            const auto& [total_sequence_length, query, key, value, past_key_0, past_value_0, past_key_1, past_value_1] = parameters;
            if (!is_ref) {
                auto total_sequence_length_i64_0 = std::make_shared<ov::op::v0::Convert>(total_sequence_length, ov::element::i64);
                const auto [sdpa_0, present_key_0, present_value_0] =
                    build_slice_concat_pattern(query, past_key_0, past_value_0, key, value, total_sequence_length_i64_0);
                auto total_sequence_length_i64_1 = std::make_shared<ov::op::v0::Convert>(total_sequence_length, ov::element::i64);
                const auto [sdpa_1, present_key_1, present_value_1] =
                    build_slice_concat_pattern(sdpa_0, past_key_1, past_value_1, sdpa_0, sdpa_0, total_sequence_length_i64_1);
                return ov::OutputVector{sdpa_1, present_key_0, present_value_0, present_key_1, present_value_1};
            }

            const auto [sdpa_0, present_key_0, present_value_0] =
                build_stateless_kv_sdpa(query, past_key_0, past_value_0, key, value, total_sequence_length, 2, true);
            const auto [sdpa_1, present_key_1, present_value_1] =
                build_stateless_kv_sdpa(sdpa_0, past_key_1, past_value_1, sdpa_0, sdpa_0, total_sequence_length, 2, true);
            return ov::OutputVector{sdpa_1, present_key_0, present_value_0, present_key_1, present_value_1};
        },
        true);
}
