// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <map>
#include <string>

#include <mixed_affinity_functions.hpp>
#include "ngraph_transformations/mixed_affinity.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"

using MixedAffinityBuilder =
    std::pair<std::function<std::shared_ptr<MixedAffinityFunctionBase>(const std::vector<ov::PartialShape>& shapes)>,
              std::string>;
using MixedAffinityParams = typename std::tuple<
        std::vector<ov::PartialShape>, // Input shapes
        MixedAffinityBuilder>;         // builder

class MixedAffinityTests : public testing::WithParamInterface<MixedAffinityParams>, public TransformationTestsF {
public:
    MixedAffinityTests() : TransformationTestsF() {
        comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
        comparator.enable(FunctionsComparator::CmpValues::NAMES);
        // RT info propagation check was covered by other mixed affinity tests
        comparator.disable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
        disable_rt_info_check();
    }

    static std::string getTestCaseName(testing::TestParamInfo<MixedAffinityParams> obj) {
        std::vector<ov::PartialShape> shapes;
        MixedAffinityBuilder builder;
        std::tie(shapes, builder) = obj.param;

        std::ostringstream result;
        result << "IS=";
        for (const auto& elem : shapes) {
            result << elem << ",";
        }
        result << "builder=" << builder.second;
        return result.str();
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<ov::intel_cpu::MixedAffinity>();
    }
};

TEST_P(MixedAffinityTests, CompareFunctions) {
    std::vector<ov::PartialShape> shapes;
    MixedAffinityBuilder builder;
    std::tie(shapes, builder) = this->GetParam();

    model = builder.first(shapes)->getOriginal();
    model_ref = builder.first(shapes)->getReference();
}

namespace {
std::vector<std::vector<ov::PartialShape>> one_input_shapes = {
    {{4, 3, 70, 70}},
};

std::vector<MixedAffinityBuilder> one_input_builders = {
    {[](const std::vector<ov::PartialShape>& shapes){ return std::make_shared<ConvWithBiasFunction>(shapes); }, "ConvWithBiasFunction"},
    {[](const std::vector<ov::PartialShape>& shapes){ return std::make_shared<TwoConvWithS2BFunction>(shapes); }, "TwoConvWithS2BFunction"},
    {[](const std::vector<ov::PartialShape>& shapes){ return std::make_shared<ConvWithReshapeFunction>(shapes); }, "ConvWithReshapeFunction"},
    {[](const std::vector<ov::PartialShape>& shapes){ return std::make_shared<Int8ConvWithDqSubFunction>(shapes); }, "Int8ConvWithDqSubFunction"},
    {[](const std::vector<ov::PartialShape>& shapes){ return std::make_shared<ConvWithTransposeFunction>(shapes); }, "ConvWithTransposeFunction"},
    {[](const std::vector<ov::PartialShape>& shapes){ return std::make_shared<ConvolutionsAndSplitFunction>(shapes); }, "ConvolutionsAndSplitFunction"},
    {[](const std::vector<ov::PartialShape>& shapes){ return std::make_shared<ConvWithSplitAndResultFunction>(shapes); }, "ConvWithSplitAndResultFunction"},
};

INSTANTIATE_TEST_SUITE_P(TransformationTests_1Input, MixedAffinityTests,
                         ::testing::Combine(
                                 ::testing::ValuesIn(one_input_shapes),
                                 ::testing::ValuesIn(one_input_builders)),
                         MixedAffinityTests::getTestCaseName);


std::vector<std::vector<ov::PartialShape>> two_inputs_shapes = {
    {{8, 3, 70, 70}, {8, 3, 70, 70}},
    {{8, 3, 70, 70}, {1, 3, 70, 70}},
    {{1, 3, 70, 70}, {8, 3, 70, 70}},
};

std::vector<MixedAffinityBuilder> two_input_builders = {
    {[](const std::vector<ov::PartialShape>& shapes){ return std::make_shared<TwoConvAndAddFunction>(shapes); }, "TwoConvAndAddFunction"},
    {[](const std::vector<ov::PartialShape>& shapes){ return std::make_shared<ConvWithConcatFunction>(shapes); }, "ConvWithConcatFunction"},
    {[](const std::vector<ov::PartialShape>& shapes){ return std::make_shared<ConvAndAddWithParameterFunction>(shapes); }, "ConvAndAddWithParameterFunction"},
    {[](const std::vector<ov::PartialShape>& shapes){ return std::make_shared<ConvWithTransposeAndAddFunction>(shapes); }, "ConvWithTransposeAndAddFunction"},
};

INSTANTIATE_TEST_SUITE_P(TransformationTests_1Inputs, MixedAffinityTests,
                         ::testing::Combine(
                                 ::testing::ValuesIn(two_inputs_shapes),
                                 ::testing::ValuesIn(two_input_builders)),
                         MixedAffinityTests::getTestCaseName);


std::vector<ov::PartialShape> conv_with_param_weights_shapes = {{8, 3, 56, 56}, {3, 1, 1, 3, 3}};
MixedAffinityBuilder conv_with_param_weights_builder = {
    [](const std::vector<ov::PartialShape>& shapes){ return std::make_shared<GrConvWithParamFunction>(shapes); }, "GrConvWithParamFunction"
};

INSTANTIATE_TEST_SUITE_P(TransformationTests_conv_with_param_weights, MixedAffinityTests,
                         ::testing::Combine(
                                 ::testing::Values(conv_with_param_weights_shapes),
                                 ::testing::Values(conv_with_param_weights_builder)),
                         MixedAffinityTests::getTestCaseName);
}  // namespace
