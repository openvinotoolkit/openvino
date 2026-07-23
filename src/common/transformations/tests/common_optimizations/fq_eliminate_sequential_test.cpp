// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fq_eliminate_sequential.hpp"

#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <vector>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/manager.hpp"

using namespace ov;
using namespace std;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;

using ov::test::utils::make_fake_quantize;

struct SequentialFqParams {
    std::vector<float> fq1;
    size_t fq1_levels;
    std::vector<float> fq2;
    size_t fq2_levels;
    bool expect_eliminate;
    std::string name;
};

class SequentialFakeQuantizeTests : public TransformationTestsF,
                                    public testing::WithParamInterface<SequentialFqParams> {
public:
    static std::string get_test_case_name(testing::TestParamInfo<SequentialFqParams> info) {
        return info.param.name;
    }
};

TEST_P(SequentialFakeQuantizeTests, CompareFunctions) {
    const auto& p = GetParam();
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 16, 16});
        auto fq1 =
            make_fake_quantize(input, element::f32, p.fq1_levels, {1}, {p.fq1[0]}, {p.fq1[1]}, {p.fq1[2]}, {p.fq1[3]});
        auto fq2 =
            make_fake_quantize(fq1, element::f32, p.fq2_levels, {1}, {p.fq2[0]}, {p.fq2[1]}, {p.fq2[2]}, {p.fq2[3]});
        auto abs = std::make_shared<v0::Abs>(fq2);
        model = std::make_shared<ov::Model>(OutputVector{abs}, ParameterVector{input});
    }
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 16, 16});
        auto fq1 =
            make_fake_quantize(input, element::f32, p.fq1_levels, {1}, {p.fq1[0]}, {p.fq1[1]}, {p.fq1[2]}, {p.fq1[3]});
        std::shared_ptr<Node> tail = fq1;
        if (!p.expect_eliminate) {
            tail = make_fake_quantize(fq1,
                                      element::f32,
                                      p.fq2_levels,
                                      {1},
                                      {p.fq2[0]},
                                      {p.fq2[1]},
                                      {p.fq2[2]},
                                      {p.fq2[3]});
        }
        auto abs = std::make_shared<v0::Abs>(tail);
        model_ref = std::make_shared<ov::Model>(OutputVector{abs}, ParameterVector{input});
    }

    manager.register_pass<ov::pass::FakeQuantizeEliminateSequential>();

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

const std::vector<SequentialFqParams> sequential_fq_params = {
    {{-1.0f, 1.0f, -1.0f, 1.0f}, 256, {-1.0f, 1.0f, -1.0f, 1.0f}, 256, true, "identical"},
    {{-1.0f, 1.0f, -1.0f, 1.0f}, 256, {-0.5f, 0.5f, -1.0f, 1.0f}, 256, false, "different_input_range"},
    {{-1.0f, 1.0f, -1.0f, 1.0f}, 256, {-1.0f, 1.0f, -2.0f, 2.0f}, 256, false, "different_output_range"},
    {{-2.0f, 2.0f, -2.0f, 2.0f}, 256, {-2.0f, 2.0f, -2.0f, 2.0f}, 257, false, "different_levels"},
    // Out-of-range scenario from the geekbench_ai model 011: FQ2 range is slightly looser than FQ1,
    // so the ranges differ and FQ2 must be kept.
    {{-17.819f, 4.900f, -17.819f, 4.900f}, 256, {-17.799f, 5.124f, -17.799f, 5.124f}, 256, false, "out_of_range"},
};

INSTANTIATE_TEST_SUITE_P(SequentialFakeQuantize,
                         SequentialFakeQuantizeTests,
                         testing::ValuesIn(sequential_fq_params),
                         SequentialFakeQuantizeTests::get_test_case_name);

// Builds a value-preserving chain of ops on top of "input" and returns the chain tail.
using ChainBuilder = std::function<std::shared_ptr<Node>(const Output<Node>&)>;

struct SequentialFqChainParams {
    ChainBuilder build_chain;
    std::string name;
};

class SequentialFakeQuantizeChainTests : public TransformationTestsF,
                                         public testing::WithParamInterface<SequentialFqChainParams> {
public:
    static std::string get_test_case_name(testing::TestParamInfo<SequentialFqChainParams> info) {
        return info.param.name;
    }
};

// FQ1 and FQ2 (identical to FQ1) are separated by a value-preserving chain: FQ2 is eliminated while
// the chain is preserved.
TEST_P(SequentialFakeQuantizeChainTests, EliminateThroughChain) {
    const auto& p = GetParam();
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 16, 16});
        auto fq1 = make_fake_quantize(input, element::f32, 256, {1}, {-1.0f}, {1.0f}, {-1.0f}, {1.0f});
        auto chain = p.build_chain(fq1);
        auto fq2 = make_fake_quantize(chain, element::f32, 256, {1}, {-1.0f}, {1.0f}, {-1.0f}, {1.0f});
        auto abs = std::make_shared<v0::Abs>(fq2);
        model = std::make_shared<ov::Model>(OutputVector{abs}, ParameterVector{input});
    }
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 16, 16});
        auto fq1 = make_fake_quantize(input, element::f32, 256, {1}, {-1.0f}, {1.0f}, {-1.0f}, {1.0f});
        auto chain = p.build_chain(fq1);
        auto abs = std::make_shared<v0::Abs>(chain);
        model_ref = std::make_shared<ov::Model>(OutputVector{abs}, ParameterVector{input});
    }

    manager.register_pass<ov::pass::FakeQuantizeEliminateSequential>();

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

const std::vector<SequentialFqChainParams> sequential_fq_chain_params = {
    {[](const Output<Node>& input) -> std::shared_ptr<Node> {
         auto target_shape = v0::Constant::create(element::i64, Shape{3}, {1, 3, 256});
         return std::make_shared<v1::Reshape>(input, target_shape, false);
     },
     "reshape"},
    {[](const Output<Node>& input) -> std::shared_ptr<Node> {
         auto order = v0::Constant::create(element::i64, Shape{4}, {0, 1, 3, 2});
         return std::make_shared<v1::Transpose>(input, order);
     },
     "transpose"},
    {[](const Output<Node>& input) -> std::shared_ptr<Node> {
         auto target_shape = v0::Constant::create(element::i64, Shape{3}, {1, 3, 256});
         auto reshape = std::make_shared<v1::Reshape>(input, target_shape, false);
         auto order = v0::Constant::create(element::i64, Shape{3}, {0, 2, 1});
         return std::make_shared<v1::Transpose>(reshape, order);
     },
     "reshape_transpose_chain"},
    {[](const Output<Node>& input) -> std::shared_ptr<Node> {
         auto squeeze_axes = v0::Constant::create(element::i64, Shape{1}, {0});
         auto squeeze = std::make_shared<v0::Squeeze>(input, squeeze_axes);
         auto unsqueeze_axes = v0::Constant::create(element::i64, Shape{1}, {0});
         return std::make_shared<v0::Unsqueeze>(squeeze, unsqueeze_axes);
     },
     "squeeze_unsqueeze"},
};

INSTANTIATE_TEST_SUITE_P(SequentialFakeQuantizeChain,
                         SequentialFakeQuantizeChainTests,
                         testing::ValuesIn(sequential_fq_chain_params),
                         SequentialFakeQuantizeChainTests::get_test_case_name);

// The intermediate Reshape feeds more than one consumer. Eliminating FQ2 only rewires FQ2's own
// output, so the other consumer is unaffected and the transformation still fires.
TEST_F(TransformationTestsF, eliminate_when_intermediate_op_has_multiple_consumers) {
    auto target_shape = v0::Constant::create(element::i64, Shape{3}, {1, 3, 256});
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 16, 16});
        auto fq1 = make_fake_quantize(input, element::f32, 256, {1}, {-1.0f}, {1.0f}, {-1.0f}, {1.0f});
        auto reshape = std::make_shared<v1::Reshape>(fq1, target_shape, false);
        auto fq2 = make_fake_quantize(reshape, element::f32, 256, {1}, {-1.0f}, {1.0f}, {-1.0f}, {1.0f});
        auto abs1 = std::make_shared<v0::Abs>(fq2);
        auto abs2 = std::make_shared<v0::Abs>(reshape);
        model = std::make_shared<ov::Model>(OutputVector{abs1, abs2}, ParameterVector{input});
    }
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 16, 16});
        auto fq1 = make_fake_quantize(input, element::f32, 256, {1}, {-1.0f}, {1.0f}, {-1.0f}, {1.0f});
        auto reshape = std::make_shared<v1::Reshape>(fq1, target_shape, false);
        auto abs1 = std::make_shared<v0::Abs>(reshape);
        auto abs2 = std::make_shared<v0::Abs>(reshape);
        model_ref = std::make_shared<ov::Model>(OutputVector{abs1, abs2}, ParameterVector{input});
    }

    manager.register_pass<ov::pass::FakeQuantizeEliminateSequential>();

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

// FQ1 and FQ2 use range constants with equal flattened values but different shapes (per-channel
// {1, 3, 1, 1} vs spatial {3, 1}), so they broadcast differently and are not interchangeable. FQ2
// must be kept: comparing only the flattened values would wrongly treat the two as identical.
TEST_F(TransformationTestsF, keep_sequential_fake_quantize_with_same_values_different_shapes) {
    const std::vector<float> in_low{-1.0f, -2.0f, -3.0f};
    const std::vector<float> in_high{1.0f, 2.0f, 3.0f};
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 3, 1});
        auto fq1 = make_fake_quantize(input, element::f32, 256, {1, 3, 1, 1}, in_low, in_high, in_low, in_high);
        auto fq2 = make_fake_quantize(fq1, element::f32, 256, {3, 1}, in_low, in_high, in_low, in_high);
        auto abs = std::make_shared<v0::Abs>(fq2);
        model = std::make_shared<ov::Model>(OutputVector{abs}, ParameterVector{input});
    }

    manager.register_pass<ov::pass::FakeQuantizeEliminateSequential>();

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

// FQ1 and FQ2 use per-channel ranges (constants broadcast over the channel axis). FQ2 is identical to
// FQ1, so it re-applies the same quantization and is eliminated.
TEST_F(TransformationTestsF, eliminate_sequential_per_channel_fake_quantize) {
    const Shape per_channel_shape{1, 3, 1, 1};
    const std::vector<float> in_low{-1.0f, -2.0f, -3.0f};
    const std::vector<float> in_high{1.0f, 2.0f, 3.0f};
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 16, 16});
        auto fq1 = make_fake_quantize(input, element::f32, 256, per_channel_shape, in_low, in_high, in_low, in_high);
        auto fq2 = make_fake_quantize(fq1, element::f32, 256, per_channel_shape, in_low, in_high, in_low, in_high);
        auto abs = std::make_shared<v0::Abs>(fq2);
        model = std::make_shared<ov::Model>(OutputVector{abs}, ParameterVector{input});
    }
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 16, 16});
        auto fq1 = make_fake_quantize(input, element::f32, 256, per_channel_shape, in_low, in_high, in_low, in_high);
        auto abs = std::make_shared<v0::Abs>(fq1);
        model_ref = std::make_shared<ov::Model>(OutputVector{abs}, ParameterVector{input});
    }

    manager.register_pass<ov::pass::FakeQuantizeEliminateSequential>();

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

// FQ1 feeds several consumers that are all identical to it: every one re-applies FQ1's quantization
// and is eliminated, leaving the consumers connected directly to FQ1.
TEST_F(TransformationTestsF, eliminate_when_fq1_has_several_identical_consumers) {
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 16, 16});
        auto fq1 = make_fake_quantize(input, element::f32, 256, {1}, {-1.0f}, {1.0f}, {-1.0f}, {1.0f});
        auto fq2_a = make_fake_quantize(fq1, element::f32, 256, {1}, {-1.0f}, {1.0f}, {-1.0f}, {1.0f});
        auto fq2_b = make_fake_quantize(fq1, element::f32, 256, {1}, {-1.0f}, {1.0f}, {-1.0f}, {1.0f});
        auto abs_a = std::make_shared<v0::Abs>(fq2_a);
        auto abs_b = std::make_shared<v0::Abs>(fq2_b);
        model = std::make_shared<ov::Model>(OutputVector{abs_a, abs_b}, ParameterVector{input});
    }
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 16, 16});
        auto fq1 = make_fake_quantize(input, element::f32, 256, {1}, {-1.0f}, {1.0f}, {-1.0f}, {1.0f});
        auto abs_a = std::make_shared<v0::Abs>(fq1);
        auto abs_b = std::make_shared<v0::Abs>(fq1);
        model_ref = std::make_shared<ov::Model>(OutputVector{abs_a, abs_b}, ParameterVector{input});
    }

    manager.register_pass<ov::pass::FakeQuantizeEliminateSequential>();

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

// FQ1 branches into an identical FakeQuantize (eliminated) and a different one (kept): only the
// redundant branch is folded.
TEST_F(TransformationTestsF, eliminate_only_identical_consumer_when_fq1_branches) {
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 16, 16});
        auto fq1 = make_fake_quantize(input, element::f32, 256, {1}, {-1.0f}, {1.0f}, {-1.0f}, {1.0f});
        auto fq_same = make_fake_quantize(fq1, element::f32, 256, {1}, {-1.0f}, {1.0f}, {-1.0f}, {1.0f});
        auto fq_diff = make_fake_quantize(fq1, element::f32, 256, {1}, {-2.0f}, {2.0f}, {-2.0f}, {2.0f});
        auto abs_same = std::make_shared<v0::Abs>(fq_same);
        auto abs_diff = std::make_shared<v0::Abs>(fq_diff);
        model = std::make_shared<ov::Model>(OutputVector{abs_same, abs_diff}, ParameterVector{input});
    }
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 16, 16});
        auto fq1 = make_fake_quantize(input, element::f32, 256, {1}, {-1.0f}, {1.0f}, {-1.0f}, {1.0f});
        auto fq_diff = make_fake_quantize(fq1, element::f32, 256, {1}, {-2.0f}, {2.0f}, {-2.0f}, {2.0f});
        auto abs_same = std::make_shared<v0::Abs>(fq1);
        auto abs_diff = std::make_shared<v0::Abs>(fq_diff);
        model_ref = std::make_shared<ov::Model>(OutputVector{abs_same, abs_diff}, ParameterVector{input});
    }

    manager.register_pass<ov::pass::FakeQuantizeEliminateSequential>();

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

// FQ1 branches into an identical FakeQuantize (eliminated) and a direct non-FakeQuantize consumer
// (left intact).
TEST_F(TransformationTestsF, eliminate_identical_consumer_with_direct_branch) {
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 16, 16});
        auto fq1 = make_fake_quantize(input, element::f32, 256, {1}, {-1.0f}, {1.0f}, {-1.0f}, {1.0f});
        auto fq2 = make_fake_quantize(fq1, element::f32, 256, {1}, {-1.0f}, {1.0f}, {-1.0f}, {1.0f});
        auto abs_fq = std::make_shared<v0::Abs>(fq2);
        auto abs_direct = std::make_shared<v0::Abs>(fq1);
        model = std::make_shared<ov::Model>(OutputVector{abs_fq, abs_direct}, ParameterVector{input});
    }
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 16, 16});
        auto fq1 = make_fake_quantize(input, element::f32, 256, {1}, {-1.0f}, {1.0f}, {-1.0f}, {1.0f});
        auto abs_fq = std::make_shared<v0::Abs>(fq1);
        auto abs_direct = std::make_shared<v0::Abs>(fq1);
        model_ref = std::make_shared<ov::Model>(OutputVector{abs_fq, abs_direct}, ParameterVector{input});
    }

    manager.register_pass<ov::pass::FakeQuantizeEliminateSequential>();

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}
