// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {
using ov::test::InputShape;

// End-to-end check for QKVSplitReshapeMatcher (VideoChat-Flash ViT).
//
//   Parameter[?, ?, 3*H*S]
//     -> Reshape([0,0,3,H,S])     -> [?,?,3,H,S]
//     -> Transpose([2,0,3,1,4])   -> [3,?,H,?,S]
//     -> Split(axis=0, num=3)     -> [1,?,H,?,S] x3
//     -> Squeeze(axis=0) x3       -> [?,H,?,S]   x3
//     -> Transpose([0,2,1,3]) x3  -> [?,?,H,S]   x3
//     -> Reshape([0,0,H*S]) x3    -> [?,?,H*S]   x3  (Q, K, V outputs)
//
// On GPU the matcher rewrites this to Split(axis=2)+Reshape and enables the
// in-place crop. The reference (template plugin) keeps the original graph, so
// run() validates numerical equivalence of the optimization.
typedef std::tuple<ov::element::Type,  // model precision
                   int64_t,            // num heads (H)
                   int64_t,            // head size (S)
                   std::string         // device
                   >
    QKVSplitReshapeParams;

class QKVSplitReshapeTest : public testing::WithParamInterface<QKVSplitReshapeParams>, virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<QKVSplitReshapeParams>& obj) {
        const auto& [precision, num_head, head_size, device] = obj.param;
        std::ostringstream result;
        result << "precision=" << precision << "_H=" << num_head << "_S=" << head_size << "_device=" << device;
        return result.str();
    }

protected:
    void SetUp() override {
        const auto& [precision, num_head, head_size, device] = this->GetParam();
        targetDevice = device;

        const int64_t F_i = 3 * num_head * head_size;
        const size_t F = static_cast<size_t>(F_i);
        const int64_t HS = num_head * head_size;

        // Dynamic batch and sequence; two static instantiations to exercise dynamism.
        InputShape qkv_shape{ov::PartialShape{-1, -1, F_i}, {ov::Shape{1, 16, F}, ov::Shape{1, 32, F}}};
        init_input_shapes({qkv_shape});

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(precision, inputDynamicShapes[0])};
        params[0]->set_friendly_name("qkv");

        auto reshape_pattern = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 0, 3, num_head, head_size});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(params[0], reshape_pattern, true);

        auto transpose_order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{2, 0, 3, 1, 4});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(reshape, transpose_order);

        auto split_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, std::vector<int64_t>{0});
        auto split = std::make_shared<ov::op::v1::Split>(transpose, split_axis, 3);

        ov::ResultVector results;
        for (size_t i = 0; i < 3; i++) {
            auto squeeze_axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
            auto squeeze = std::make_shared<ov::op::v0::Squeeze>(split->output(i), squeeze_axes);

            auto perm = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 2, 1, 3});
            auto qkv_transpose = std::make_shared<ov::op::v1::Transpose>(squeeze, perm);

            auto flat_pattern = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 0, HS});
            auto flatten = std::make_shared<ov::op::v1::Reshape>(qkv_transpose, flat_pattern, true);

            // A trivial scale keeps a real consumer downstream of the flatten.
            auto scale = ov::op::v0::Constant::create(precision, ov::Shape{1}, std::vector<float>{1.0f});
            auto scaled = std::make_shared<ov::op::v1::Multiply>(flatten, scale);

            results.push_back(std::make_shared<ov::op::v0::Result>(scaled));
        }

        function = std::make_shared<ov::Model>(results, params, "qkv_split_reshape");
        abs_threshold = 0.01f;
        rel_threshold = 0.01f;
    }
};

TEST_P(QKVSplitReshapeTest, Inference) {
    run();
}

const auto testParams_smoke = ::testing::Combine(::testing::Values(ov::element::f16),
                                                 ::testing::Values(static_cast<int64_t>(8)),   // num_head
                                                 ::testing::Values(static_cast<int64_t>(64)),  // head_size
                                                 ::testing::Values(ov::test::utils::DEVICE_GPU));

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_qkv_split_reshape, QKVSplitReshapeTest, testParams_smoke, QKVSplitReshapeTest::getTestCaseName);

// Same QKV split pattern, but each flatten output feeds a last-dim REDUCTION
// (ReduceMean of x^2 = the first step of the q_norm/k_norm RMSNorm in the real
// VideoChat-Flash ViT) instead of an element-wise op.
//
// This is the case the element-wise Multiply consumer above cannot exercise:
// after QKVSplitReshapeMatcher enables the in-place crop, the flatten output
// [?, seq, H*S] aliases the interleaved [?, seq, 3*H*S] buffer with a dynamic
// padding offset. An element-wise op is padding-transparent, but a reduction
// over the padded last dim must apply the propagated offset/extent correctly;
// if it does not, neighbouring Q/K/V slices leak into the mean and the result
// diverges from the template reference.
class QKVSplitReshapeReduceTest : public testing::WithParamInterface<QKVSplitReshapeParams>,
                                  virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<QKVSplitReshapeParams>& obj) {
        const auto& [precision, num_head, head_size, device] = obj.param;
        std::ostringstream result;
        result << "precision=" << precision << "_H=" << num_head << "_S=" << head_size << "_device=" << device;
        return result.str();
    }

protected:
    void SetUp() override {
        const auto& [precision, num_head, head_size, device] = this->GetParam();
        targetDevice = device;

        const int64_t F_i = 3 * num_head * head_size;
        const size_t F = static_cast<size_t>(F_i);
        const int64_t HS = num_head * head_size;

        InputShape qkv_shape{ov::PartialShape{-1, -1, F_i}, {ov::Shape{1, 16, F}, ov::Shape{1, 32, F}}};
        init_input_shapes({qkv_shape});

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(precision, inputDynamicShapes[0])};
        params[0]->set_friendly_name("qkv");

        auto reshape_pattern = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 0, 3, num_head, head_size});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(params[0], reshape_pattern, true);

        auto transpose_order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{2, 0, 3, 1, 4});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(reshape, transpose_order);

        auto split_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, std::vector<int64_t>{0});
        auto split = std::make_shared<ov::op::v1::Split>(transpose, split_axis, 3);

        auto two = ov::op::v0::Constant::create(precision, ov::Shape{}, std::vector<float>{2.0f});
        auto reduce_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});

        ov::ResultVector results;
        for (size_t i = 0; i < 3; i++) {
            auto squeeze_axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
            auto squeeze = std::make_shared<ov::op::v0::Squeeze>(split->output(i), squeeze_axes);

            auto perm = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 2, 1, 3});
            auto qkv_transpose = std::make_shared<ov::op::v1::Transpose>(squeeze, perm);

            auto flat_pattern = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 0, HS});
            auto flatten = std::make_shared<ov::op::v1::Reshape>(qkv_transpose, flat_pattern, true);

            // RMSNorm first step: mean over the last (padded) dim of the squared values.
            auto squared = std::make_shared<ov::op::v1::Power>(flatten, two);
            auto mean = std::make_shared<ov::op::v1::ReduceMean>(squared, reduce_axis, true);

            results.push_back(std::make_shared<ov::op::v0::Result>(mean));
        }

        function = std::make_shared<ov::Model>(results, params, "qkv_split_reshape_reduce");
        abs_threshold = 0.01f;
        rel_threshold = 0.01f;
    }
};

TEST_P(QKVSplitReshapeReduceTest, Inference) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_qkv_split_reshape_reduce, QKVSplitReshapeReduceTest, testParams_smoke, QKVSplitReshapeReduceTest::getTestCaseName);
}  // namespace
