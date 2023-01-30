// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include <shared_test_classes/base/ov_subgraph.hpp>
#include <ngraph_functions/builders.hpp>
#include <common_test_utils/common_utils.hpp>
#include <common_test_utils/ov_tensor_utils.hpp>
#include <mixed_affinity_functions.hpp>

namespace ov {
namespace test {
namespace mixed_affinity {
using namespace ov::test;
using namespace ngraph::helpers;

class MixedAffinityTest : public testing::WithParamInterface<MixedAffinityParams>, virtual public SubgraphBaseTest {
public:
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
        rel_threshold = 1e-4f;
        std::vector<ov::PartialShape> shapes;
        MixedAffinityBuilder builder;
        std::tie(shapes, builder) = this->GetParam();

        targetDevice = CommonTestUtils::DEVICE_CPU;
        init_input_shapes(static_partial_shapes_to_test_representation(shapes));
        function = builder.first(shapes)->getOriginal();
    }
};

TEST_P(MixedAffinityTest, CompareWithRefs) {
    run();
}

std::vector<std::vector<ov::PartialShape>> one_input_shapes = {
    {{8, 3, 70, 70}},
};

std::vector<MixedAffinityBuilder> one_input_builders = {
    {[](const std::vector<ov::PartialShape>& shapes){ return std::make_shared<ConvWithLRNFunction>(shapes); }, "ConvWithLRN"},
    {[](const std::vector<ov::PartialShape>& shapes){ return std::make_shared<ConvWithBiasFunction>(shapes); }, "ConvWithBias"},
    {[](const std::vector<ov::PartialShape>& shapes){ return std::make_shared<TwoConvWithS2BFunction>(shapes); }, "TwoConvWithS2B"},
    {[](const std::vector<ov::PartialShape>& shapes){ return std::make_shared<ConvWithReshapeFunction>(shapes); }, "ConvWithReshape"},
    {[](const std::vector<ov::PartialShape>& shapes){ return std::make_shared<ConvWithTransposeFunction>(shapes); }, "ConvWithTranspose"},
    {[](const std::vector<ov::PartialShape>& shapes){ return std::make_shared<ConvolutionsAndSplitFunction>(shapes); }, "ConvolutionsAndSplit"},
    {[](const std::vector<ov::PartialShape>& shapes){ return std::make_shared<ConvWithSplitAndResultFunction>(shapes); }, "ConvWithSplitAndResult"},
};

INSTANTIATE_TEST_SUITE_P(smoke_MixedAffinity_1input, MixedAffinityTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(one_input_shapes),
                                 ::testing::ValuesIn(one_input_builders)),
                         MixedAffinityTest::getTestCaseName);

std::vector<std::vector<ov::PartialShape>> two_inputs_shapes = {
    {{8, 3, 56, 56}, {8, 3, 56, 56}},
    {{8, 3, 56, 56}, {1, 3, 56, 56}},
    {{1, 3, 56, 56}, {8, 3, 56, 56}},
};

std::vector<MixedAffinityBuilder> two_input_builders = {
    {[](const std::vector<ov::PartialShape>& shapes){ return std::make_shared<TwoConvAndAddFunction>(shapes); }, "TwoConvAndAdd"},
    {[](const std::vector<ov::PartialShape>& shapes){ return std::make_shared<ConvWithConcatFunction>(shapes); }, "ConvWithConcat"},
    {[](const std::vector<ov::PartialShape>& shapes){ return std::make_shared<ConvWithTransposeAddFunction>(shapes); }, "ConvWithTransposeAdd"},
    {[](const std::vector<ov::PartialShape>& shapes){ return std::make_shared<ConvAndAddWithParameterFunction>(shapes); }, "ConvAndAddWithParameter"},
    {[](const std::vector<ov::PartialShape>& shapes){ return std::make_shared<ConvWithTransposeAndAddFunction>(shapes); }, "ConvWithTransposeAndAdd"},
};

INSTANTIATE_TEST_SUITE_P(smoke_MixedAffinity_2inputs, MixedAffinityTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(two_inputs_shapes),
                                 ::testing::ValuesIn(two_input_builders)),
                         MixedAffinityTest::getTestCaseName);

std::vector<ov::PartialShape> conv_with_param_weights_shapes = {{8, 3, 56, 56}, {3, 1, 1, 3, 3}};
MixedAffinityBuilder conv_with_param_weights_builder = {
    [](const std::vector<ov::PartialShape>& shapes){ return std::make_shared<GrConvWithParamFunction>(shapes); }, "GrConvWithParam"
};

INSTANTIATE_TEST_SUITE_P(smoke_MixedAffinity_conv_with_param_weights, MixedAffinityTest,
                         ::testing::Combine(
                                 ::testing::Values(conv_with_param_weights_shapes),
                                 ::testing::Values(conv_with_param_weights_builder)),
                         MixedAffinityTest::getTestCaseName);
}  // namespace mixed_affinity
}  // namespace test
}  // namespace ov
