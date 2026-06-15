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
#include "openvino/pass/manager.hpp"

using namespace ov;
using namespace std;

namespace v0 = ov::op::v0;

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
