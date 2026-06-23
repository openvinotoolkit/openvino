// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fq_eliminate_sequential.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/fake_quantize.hpp"
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

namespace {
std::shared_ptr<v0::FakeQuantize> make_fake_quantize(const Output<Node>& data,
                                                     float input_low,
                                                     float input_high,
                                                     float output_low,
                                                     float output_high,
                                                     size_t levels) {
    auto in_low = v0::Constant::create(element::f32, Shape{1}, {input_low});
    auto in_high = v0::Constant::create(element::f32, Shape{1}, {input_high});
    auto out_low = v0::Constant::create(element::f32, Shape{1}, {output_low});
    auto out_high = v0::Constant::create(element::f32, Shape{1}, {output_high});
    return std::make_shared<v0::FakeQuantize>(data, in_low, in_high, out_low, out_high, levels);
}
}  // namespace

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
        auto fq1 = make_fake_quantize(input, p.fq1[0], p.fq1[1], p.fq1[2], p.fq1[3], p.fq1_levels);
        auto fq2 = make_fake_quantize(fq1, p.fq2[0], p.fq2[1], p.fq2[2], p.fq2[3], p.fq2_levels);
        auto abs = std::make_shared<v0::Abs>(fq2);
        model = std::make_shared<ov::Model>(OutputVector{abs}, ParameterVector{input});
    }
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 16, 16});
        auto fq1 = make_fake_quantize(input, p.fq1[0], p.fq1[1], p.fq1[2], p.fq1[3], p.fq1_levels);
        std::shared_ptr<Node> tail = fq1;
        if (!p.expect_eliminate) {
            tail = make_fake_quantize(fq1, p.fq2[0], p.fq2[1], p.fq2[2], p.fq2[3], p.fq2_levels);
        }
        auto abs = std::make_shared<v0::Abs>(tail);
        model_ref = std::make_shared<ov::Model>(OutputVector{abs}, ParameterVector{input});
    }

    manager.register_pass<ov::pass::FakeQuantizeEliminateSequential>();

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

const std::vector<SequentialFqParams> sequential_fq_params = {
    {{-1.0f, 1.0f, -1.0f, 1.0f}, 256, {-1.0f, 1.0f, -1.0f, 1.0f}, 256, true, "exact_match"},
    {{-1.0f, 1.0f, -1.0f, 1.0f}, 256, {-0.5f, 0.5f, -1.0f, 1.0f}, 256, false, "mismatch_ranges"},
    {{-1.0f, 1.0f, -1.0f, 1.0f}, 256, {-2.0f, 2.0f, -2.0f, 2.0f}, 1021, true, "subrange_aligned"},
    {{-1.0f, 1.0f, -1.0f, 1.0f}, 256, {-2.0f, 2.0f, -2.0f, 2.0f}, 256, false, "subrange_unaligned"},
    {{-2.0f, 2.0f, -2.0f, 2.0f}, 256, {-2.0f, 2.0f, -2.0f, 2.0f}, 511, true, "same_range_aligned"},
    {{-2.0f, 2.0f, -2.0f, 2.0f}, 256, {-2.0f, 2.0f, -2.0f, 2.0f}, 257, false, "same_range_unaligned"},
};

INSTANTIATE_TEST_SUITE_P(SequentialFakeQuantize,
                         SequentialFakeQuantizeTests,
                         testing::ValuesIn(sequential_fq_params),
                         SequentialFakeQuantizeTests::get_test_case_name);

struct MergedFqParams {
    std::vector<float> fq1;
    size_t fq1_levels;
    std::vector<float> fq2;
    size_t fq2_levels;
    std::string name;
};

class MergedFakeQuantizeTests : public TransformationTestsF, public testing::WithParamInterface<MergedFqParams> {
public:
    static std::string get_test_case_name(testing::TestParamInfo<MergedFqParams> info) {
        return info.param.name;
    }
};

TEST_P(MergedFakeQuantizeTests, CompareFunctions) {
    const auto& p = GetParam();
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 16, 16});
        auto fq1 = make_fake_quantize(input, p.fq1[0], p.fq1[1], p.fq1[2], p.fq1[3], p.fq1_levels);
        auto fq2 = make_fake_quantize(fq1, p.fq2[0], p.fq2[1], p.fq2[2], p.fq2[3], p.fq2_levels);
        auto abs = std::make_shared<v0::Abs>(fq2);
        model = std::make_shared<ov::Model>(OutputVector{abs}, ParameterVector{input});
    }
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 16, 16});
        // Reference model with merged FQ: FQ1's input range -> FQ2's output range, FQ1's levels
        auto merged_fq = make_fake_quantize(input, p.fq1[0], p.fq1[1], p.fq2[2], p.fq2[3], p.fq1_levels);
        auto abs = std::make_shared<v0::Abs>(merged_fq);
        model_ref = std::make_shared<ov::Model>(OutputVector{abs}, ParameterVector{input});
    }

    manager.register_pass<ov::pass::FakeQuantizeEliminateSequential>();

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

const std::vector<MergedFqParams> merged_fq_params = {
    {{-2.0f, 2.0f, -1.0f, 1.0f}, 256, {-1.0f, 1.0f, -1.0f, 0.0f}, 256, "merge_compressed_output"},
    {{-4.0f, 4.0f, -2.0f, 2.0f}, 256, {-2.0f, 2.0f, -2.0f, 0.0f}, 256, "merge_different_input_ranges"},
    {{-3.0f, 3.0f, -1.5f, 1.5f}, 512, {-1.5f, 1.5f, -1.5f, 0.0f}, 512, "merge_even_levels"},
};

INSTANTIATE_TEST_SUITE_P(MergedFakeQuantize,
                         MergedFakeQuantizeTests,
                         testing::ValuesIn(merged_fq_params),
                         MergedFakeQuantizeTests::get_test_case_name);

TEST_F(TransformationTestsF, eliminate_sequential_fake_quantize_subgraph) {
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 16, 16});
        auto fq1 = make_fake_quantize(input, -1.0f, 1.0f, -1.0f, 1.0f, 256);
        auto fq2 = make_fake_quantize(fq1, -1.0f, 1.0f, -1.0f, 1.0f, 256);
        auto abs = std::make_shared<v0::Abs>(fq2);
        model = std::make_shared<ov::Model>(OutputVector{abs}, ParameterVector{input});
    }
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 16, 16});
        auto fq1 = make_fake_quantize(input, -1.0f, 1.0f, -1.0f, 1.0f, 256);
        auto abs = std::make_shared<v0::Abs>(fq1);
        model_ref = std::make_shared<ov::Model>(OutputVector{abs}, ParameterVector{input});
    }

    manager.register_pass<ov::pass::FakeQuantizeEliminateSequential>();

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, do_not_eliminate_sequential_fake_quantize_subgraph) {
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 16, 16});
        auto fq1 = make_fake_quantize(input, -2.0f, 2.0f, -2.0f, 2.0f, 256);
        auto fq2 = make_fake_quantize(fq1, -2.0f, 2.0f, -2.0f, 2.0f, 257);
        auto abs = std::make_shared<v0::Abs>(fq2);
        model = std::make_shared<ov::Model>(OutputVector{abs}, ParameterVector{input});
    }
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 16, 16});
        auto fq1 = make_fake_quantize(input, -2.0f, 2.0f, -2.0f, 2.0f, 256);
        auto fq2 = make_fake_quantize(fq1, -2.0f, 2.0f, -2.0f, 2.0f, 257);
        auto abs = std::make_shared<v0::Abs>(fq2);
        model_ref = std::make_shared<ov::Model>(OutputVector{abs}, ParameterVector{input});
    }

    manager.register_pass<ov::pass::FakeQuantizeEliminateSequential>();

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

// Test for FakeQuantizeEliminateSequential with out-of-range scenario from geekbench_ai model 011
// FQ1: in_low=-17.819, in_high=4.900, out_low=-17.819, out_high=4.900
// FQ2: in_low=-17.799, in_high=5.124, out_low=-17.799, out_high=5.124
// This case should NOT eliminate because FQ1 output is NOT within FQ2 input range:
// FQ1 out_low (-17.819) < FQ2 in_low (-17.799)
TEST_F(TransformationTestsF, FakeQuantizeEliminateSequential_out_of_range_callback_fails) {
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, Shape{1, 1083, 4});
        // FQ1 with slightly tighter range
        auto fq1 = make_fake_quantize(input, -17.819f, 4.900f, -17.819f, 4.900f, 256);
        // FQ2 with slightly looser range - note: in_low is higher (less negative) than fq1's out_low
        auto fq2 = make_fake_quantize(fq1, -17.799f, 5.124f, -17.799f, 5.124f, 256);
        auto abs = std::make_shared<v0::Abs>(fq2);
        model = std::make_shared<ov::Model>(OutputVector{abs}, ParameterVector{input});
    }
    {
        // Reference: transformation should NOT apply, so both FQs remain
        auto input = std::make_shared<v0::Parameter>(element::f32, Shape{1, 1083, 4});
        auto fq1 = make_fake_quantize(input, -17.819f, 4.900f, -17.819f, 4.900f, 256);
        auto fq2 = make_fake_quantize(fq1, -17.799f, 5.124f, -17.799f, 5.124f, 256);
        auto abs = std::make_shared<v0::Abs>(fq2);
        model_ref = std::make_shared<ov::Model>(OutputVector{abs}, ParameterVector{input});
    }

    manager.register_pass<ov::pass::FakeQuantizeEliminateSequential>();

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

// FQ2 separated from FQ1 by a Reshape: FQ2 is identity and grids align, so FQ2 is eliminated while
// the Reshape is preserved.
TEST_F(TransformationTestsF, eliminate_sequential_fake_quantize_with_reshape) {
    auto target_shape = v0::Constant::create(element::i64, Shape{3}, {1, 3, 256});
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 16, 16});
        auto fq1 = make_fake_quantize(input, -1.0f, 1.0f, -1.0f, 1.0f, 256);
        auto reshape = std::make_shared<v1::Reshape>(fq1, target_shape, false);
        auto fq2 = make_fake_quantize(reshape, -2.0f, 2.0f, -2.0f, 2.0f, 1021);
        auto abs = std::make_shared<v0::Abs>(fq2);
        model = std::make_shared<ov::Model>(OutputVector{abs}, ParameterVector{input});
    }
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 16, 16});
        auto fq1 = make_fake_quantize(input, -1.0f, 1.0f, -1.0f, 1.0f, 256);
        auto reshape = std::make_shared<v1::Reshape>(fq1, target_shape, false);
        auto abs = std::make_shared<v0::Abs>(reshape);
        model_ref = std::make_shared<ov::Model>(OutputVector{abs}, ParameterVector{input});
    }

    manager.register_pass<ov::pass::FakeQuantizeEliminateSequential>();

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

// FQ2 separated from FQ1 by a Transpose: FQ2 is not identity, so FQ1 and FQ2 are merged into a single
// FQ placed before the Transpose.
TEST_F(TransformationTestsF, merge_sequential_fake_quantize_with_transpose) {
    auto order = v0::Constant::create(element::i64, Shape{4}, {0, 1, 3, 2});
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 16, 16});
        auto fq1 = make_fake_quantize(input, -2.0f, 2.0f, -1.0f, 1.0f, 256);
        auto transpose = std::make_shared<v1::Transpose>(fq1, order);
        auto fq2 = make_fake_quantize(transpose, -1.0f, 1.0f, -1.0f, 0.0f, 256);
        auto abs = std::make_shared<v0::Abs>(fq2);
        model = std::make_shared<ov::Model>(OutputVector{abs}, ParameterVector{input});
    }
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 16, 16});
        auto merged_fq = make_fake_quantize(input, -2.0f, 2.0f, -1.0f, 0.0f, 256);
        auto transpose = std::make_shared<v1::Transpose>(merged_fq, order);
        auto abs = std::make_shared<v0::Abs>(transpose);
        model_ref = std::make_shared<ov::Model>(OutputVector{abs}, ParameterVector{input});
    }

    manager.register_pass<ov::pass::FakeQuantizeEliminateSequential>();

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

// FQ2 separated from FQ1 by a Reshape followed by a Transpose: the whole value-preserving chain is
// traversed and FQ2 (identity) is eliminated while the chain is preserved.
TEST_F(TransformationTestsF, eliminate_sequential_fake_quantize_with_reshape_transpose_chain) {
    auto target_shape = v0::Constant::create(element::i64, Shape{3}, {1, 3, 256});
    auto order = v0::Constant::create(element::i64, Shape{3}, {0, 2, 1});
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 16, 16});
        auto fq1 = make_fake_quantize(input, -1.0f, 1.0f, -1.0f, 1.0f, 256);
        auto reshape = std::make_shared<v1::Reshape>(fq1, target_shape, false);
        auto transpose = std::make_shared<v1::Transpose>(reshape, order);
        auto fq2 = make_fake_quantize(transpose, -2.0f, 2.0f, -2.0f, 2.0f, 1021);
        auto abs = std::make_shared<v0::Abs>(fq2);
        model = std::make_shared<ov::Model>(OutputVector{abs}, ParameterVector{input});
    }
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 16, 16});
        auto fq1 = make_fake_quantize(input, -1.0f, 1.0f, -1.0f, 1.0f, 256);
        auto reshape = std::make_shared<v1::Reshape>(fq1, target_shape, false);
        auto transpose = std::make_shared<v1::Transpose>(reshape, order);
        auto abs = std::make_shared<v0::Abs>(transpose);
        model_ref = std::make_shared<ov::Model>(OutputVector{abs}, ParameterVector{input});
    }

    manager.register_pass<ov::pass::FakeQuantizeEliminateSequential>();

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

// FQ2 separated from FQ1 by a Squeeze followed by an Unsqueeze: the chain is traversed and FQ1/FQ2
// are merged into a single FQ placed before the chain.
TEST_F(TransformationTestsF, merge_sequential_fake_quantize_with_squeeze_unsqueeze) {
    auto squeeze_axes = v0::Constant::create(element::i64, Shape{1}, {0});
    auto unsqueeze_axes = v0::Constant::create(element::i64, Shape{1}, {0});
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 16, 16});
        auto fq1 = make_fake_quantize(input, -2.0f, 2.0f, -1.0f, 1.0f, 256);
        auto squeeze = std::make_shared<v0::Squeeze>(fq1, squeeze_axes);
        auto unsqueeze = std::make_shared<v0::Unsqueeze>(squeeze, unsqueeze_axes);
        auto fq2 = make_fake_quantize(unsqueeze, -1.0f, 1.0f, -1.0f, 0.0f, 256);
        auto abs = std::make_shared<v0::Abs>(fq2);
        model = std::make_shared<ov::Model>(OutputVector{abs}, ParameterVector{input});
    }
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 16, 16});
        auto merged_fq = make_fake_quantize(input, -2.0f, 2.0f, -1.0f, 0.0f, 256);
        auto squeeze = std::make_shared<v0::Squeeze>(merged_fq, squeeze_axes);
        auto unsqueeze = std::make_shared<v0::Unsqueeze>(squeeze, unsqueeze_axes);
        auto abs = std::make_shared<v0::Abs>(unsqueeze);
        model_ref = std::make_shared<ov::Model>(OutputVector{abs}, ParameterVector{input});
    }

    manager.register_pass<ov::pass::FakeQuantizeEliminateSequential>();

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

// The intermediate Reshape feeds more than one consumer, so bypassing it would change the graph for
// the other consumer: the transformation must not fire.
TEST_F(TransformationTestsF, do_not_fold_when_intermediate_op_has_multiple_consumers) {
    auto target_shape = v0::Constant::create(element::i64, Shape{3}, {1, 3, 256});
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 16, 16});
        auto fq1 = make_fake_quantize(input, -1.0f, 1.0f, -1.0f, 1.0f, 256);
        auto reshape = std::make_shared<v1::Reshape>(fq1, target_shape, false);
        auto fq2 = make_fake_quantize(reshape, -2.0f, 2.0f, -2.0f, 2.0f, 1021);
        auto abs1 = std::make_shared<v0::Abs>(fq2);
        auto abs2 = std::make_shared<v0::Abs>(reshape);
        model = std::make_shared<ov::Model>(OutputVector{abs1, abs2}, ParameterVector{input});
    }
    {
        auto input = std::make_shared<v0::Parameter>(element::f32, Shape{1, 3, 16, 16});
        auto fq1 = make_fake_quantize(input, -1.0f, 1.0f, -1.0f, 1.0f, 256);
        auto reshape = std::make_shared<v1::Reshape>(fq1, target_shape, false);
        auto fq2 = make_fake_quantize(reshape, -2.0f, 2.0f, -2.0f, 2.0f, 1021);
        auto abs1 = std::make_shared<v0::Abs>(fq2);
        auto abs2 = std::make_shared<v0::Abs>(reshape);
        model_ref = std::make_shared<ov::Model>(OutputVector{abs1, abs2}, ParameterVector{input});
    }

    manager.register_pass<ov::pass::FakeQuantizeEliminateSequential>();

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}
