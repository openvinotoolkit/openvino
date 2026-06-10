// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <openvino/runtime/core.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/pass/manager.hpp"
#include "plugin/transformations/transpose_fusion.hpp"

using namespace testing;
using namespace ov::intel_gpu;
using namespace ov;
using namespace ov::opset13;

namespace ov {
namespace test {
namespace intel_gpu {

namespace {

// VideoChat-Flash ViT attention QKV projection sub-graph.
//
//   Parameter[?, ?, 3*H*S]
//     -> Reshape([0,0,3,H,S])     -> [?,?,3,H,S]
//     -> Transpose([2,0,3,1,4])   -> [3,?,H,?,S]
//     -> Split(axis=0, num=3)     -> [1,?,H,?,S] x3
//     -> Squeeze(axis=0) x3       -> [?,H,?,S]   x3
//     (Q/K) -> Transpose([0,2,1,3]) -> [?,?,H,S] -> flatten Reshape([0,0,H*S]) -> [?,?,H*S]
//     (V)   -> kept as [?,H,?,S] for SDPA
constexpr int64_t H = 8;
constexpr int64_t S = 64;
constexpr int64_t F = 3 * H * S;  // 1536
constexpr int64_t HS = H * S;     // 512

std::shared_ptr<Reshape> make_input_reshape(const std::shared_ptr<Parameter>& input) {
    auto pattern = Constant::create(element::i64, Shape{5}, std::vector<int64_t>{0, 0, 3, H, S});
    auto reshape = std::make_shared<Reshape>(input, pattern, true);
    reshape->set_friendly_name("qkv_reshape");
    return reshape;
}

// Source: Q, K, V all flattened (Transpose([0,2,1,3]) + flatten Reshape).
std::shared_ptr<ov::Model> build_model_all_flatten() {
    auto input = std::make_shared<Parameter>(element::f16, PartialShape{-1, -1, F});
    input->set_friendly_name("input");

    auto reshape = make_input_reshape(input);

    auto transpose_order = Constant::create(element::i64, Shape{5}, std::vector<int64_t>{2, 0, 3, 1, 4});
    auto transpose = std::make_shared<Transpose>(reshape, transpose_order);
    transpose->set_friendly_name("qkv_transpose");

    auto split_axis = Constant::create(element::i64, Shape{}, std::vector<int64_t>{0});
    auto split = std::make_shared<Split>(transpose, split_axis, 3);
    split->set_friendly_name("qkv_split");

    OutputVector results;
    for (size_t i = 0; i < 3; i++) {
        auto squeeze_axes = Constant::create(element::i64, Shape{1}, std::vector<int64_t>{0});
        auto squeeze = std::make_shared<ov::op::v0::Squeeze>(split->output(i), squeeze_axes);

        auto perm = Constant::create(element::i64, Shape{4}, std::vector<int64_t>{0, 2, 1, 3});
        auto qkv_transpose = std::make_shared<Transpose>(squeeze, perm);

        auto flat_pattern = Constant::create(element::i64, Shape{3}, std::vector<int64_t>{0, 0, HS});
        auto flatten = std::make_shared<Reshape>(qkv_transpose, flat_pattern, true);
        results.push_back(flatten);
    }

    return std::make_shared<ov::Model>(results, ParameterVector{input});
}

// Target: Transpose removed, Split moved to axis=2, Squeeze+Transpose+flatten
// collapsed into a single base-mode Reshape([0,0,H*S]) per slice.
std::shared_ptr<ov::Model> build_target_all_flatten() {
    auto input = std::make_shared<Parameter>(element::f16, PartialShape{-1, -1, F});
    input->set_friendly_name("input");

    auto reshape = make_input_reshape(input);

    auto split_axis = Constant::create(element::i64, Shape{}, std::vector<int64_t>{2});
    auto split = std::make_shared<Split>(reshape, split_axis, 3);
    split->set_friendly_name("qkv_split");

    OutputVector results;
    for (size_t i = 0; i < 3; i++) {
        auto flat_pattern = Constant::create(element::i64, Shape{3}, std::vector<int64_t>{0, 0, HS});
        auto flatten = std::make_shared<Reshape>(split->output(i), flat_pattern, true);
        results.push_back(flatten);
    }

    return std::make_shared<ov::Model>(results, ParameterVector{input});
}

// Source: Q, K flattened; V kept as [?,H,?,S] (direct to SDPA, no Transpose/flatten).
std::shared_ptr<ov::Model> build_model_v_kept() {
    auto input = std::make_shared<Parameter>(element::f16, PartialShape{-1, -1, F});
    input->set_friendly_name("input");

    auto reshape = make_input_reshape(input);

    auto transpose_order = Constant::create(element::i64, Shape{5}, std::vector<int64_t>{2, 0, 3, 1, 4});
    auto transpose = std::make_shared<Transpose>(reshape, transpose_order);
    transpose->set_friendly_name("qkv_transpose");

    auto split_axis = Constant::create(element::i64, Shape{}, std::vector<int64_t>{0});
    auto split = std::make_shared<Split>(transpose, split_axis, 3);
    split->set_friendly_name("qkv_split");

    OutputVector results;
    for (size_t i = 0; i < 3; i++) {
        auto squeeze_axes = Constant::create(element::i64, Shape{1}, std::vector<int64_t>{0});
        auto squeeze = std::make_shared<ov::op::v0::Squeeze>(split->output(i), squeeze_axes);

        if (i < 2) {
            auto perm = Constant::create(element::i64, Shape{4}, std::vector<int64_t>{0, 2, 1, 3});
            auto qkv_transpose = std::make_shared<Transpose>(squeeze, perm);
            auto flat_pattern = Constant::create(element::i64, Shape{3}, std::vector<int64_t>{0, 0, HS});
            auto flatten = std::make_shared<Reshape>(qkv_transpose, flat_pattern, true);
            results.push_back(flatten);
        } else {
            results.push_back(squeeze);  // V: [?,H,?,S]
        }
    }

    return std::make_shared<ov::Model>(results, ParameterVector{input});
}

// Target for V-kept case: Q/K collapse to single Reshape; V becomes
// Squeeze(axis=2) + compensating Transpose([0,2,1,3]) to restore [?,H,?,S].
std::shared_ptr<ov::Model> build_target_v_kept() {
    auto input = std::make_shared<Parameter>(element::f16, PartialShape{-1, -1, F});
    input->set_friendly_name("input");

    auto reshape = make_input_reshape(input);

    auto split_axis = Constant::create(element::i64, Shape{}, std::vector<int64_t>{2});
    auto split = std::make_shared<Split>(reshape, split_axis, 3);
    split->set_friendly_name("qkv_split");

    OutputVector results;
    for (size_t i = 0; i < 3; i++) {
        if (i < 2) {
            auto flat_pattern = Constant::create(element::i64, Shape{3}, std::vector<int64_t>{0, 0, HS});
            auto flatten = std::make_shared<Reshape>(split->output(i), flat_pattern, true);
            results.push_back(flatten);
        } else {
            auto squeeze_axes = Constant::create(element::i64, Shape{1}, std::vector<int64_t>{2});
            auto squeeze = std::make_shared<ov::op::v0::Squeeze>(split->output(i), squeeze_axes);
            auto perm = Constant::create(element::i64, Shape{4}, std::vector<int64_t>{0, 2, 1, 3});
            auto v_transpose = std::make_shared<Transpose>(squeeze, perm);
            results.push_back(v_transpose);
        }
    }

    return std::make_shared<ov::Model>(results, ParameterVector{input});
}

}  // namespace

// All three slices flattened: Transpose+Split+Squeeze+Transpose+flatten collapses
// to Split(axis=2)+Reshape, eliminating both Transpose ops.
TEST_F(TransformationTestsF, QKVSplitReshapeAllFlatten) {
    disable_rt_info_check();
    {
        model = build_model_all_flatten();
        manager.register_pass<TransposeFusion>();
    }
    {
        model_ref = build_target_all_flatten();
    }
}

// Q/K flattened, V kept as [?,H,?,S]: V slice gets a compensating Transpose.
TEST_F(TransformationTestsF, QKVSplitReshapeVKept) {
    disable_rt_info_check();
    {
        model = build_model_v_kept();
        manager.register_pass<TransposeFusion>();
    }
    {
        model_ref = build_target_v_kept();
    }
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
