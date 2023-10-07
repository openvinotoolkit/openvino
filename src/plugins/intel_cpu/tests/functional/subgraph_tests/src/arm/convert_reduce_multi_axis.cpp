// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <debug.h>
#include <shared_test_classes/base/ov_subgraph.hpp>
#include <ov_models/builders.hpp>
#include "common_test_utils/common_utils.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "functional_test_utils/skip_tests_config.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"

#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/convolution_params.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;
using namespace ngraph;
using namespace ngraph::helpers;

namespace CPUSubgraphTestsDefinitions {

typedef std::tuple<
        std::vector<int>,               // Axis to reduce order
        ngraph::helpers::ReductionType, // Reduce operation type
        std::vector<InputShape>         // Input shapes
> reduceConvertCPUTestParamsSet;

class reduceTransformationCPUTest: public testing::WithParamInterface<reduceConvertCPUTestParamsSet>,
                                            virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<reduceConvertCPUTestParamsSet> obj) {
        std::vector<InputShape> inputShapes;
        std::vector<int> axes;
        ReductionType reductionType;
        std::tie(axes, reductionType, inputShapes) = obj.param;

        std::ostringstream result;
        result << "type=" << reductionType << "_";
        result << "IS=(";
        for (const auto& shape : inputShapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << ")_axes=" << ov::test::utils::vec2str(axes) << "_";
        return result.str();
    }

protected:
    int numberOfExpectedReduce;
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        std::vector<int> axes;
        bool keepDims = true;
        std::vector<InputShape> inputShapes;
        std::tie(axes, reductionType, inputShapes) = this->GetParam();
        numberOfExpectedReduce = axes.size();

        init_input_shapes(inputShapes);

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(ngraph::element::f32, shape));
        }
        auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
        std::vector<size_t> shapeAxes;
        shapeAxes.push_back(axes.size());
        auto reductionAxesNode = std::dynamic_pointer_cast<ngraph::Node>(
                std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::i64, ngraph::Shape(shapeAxes), axes));

        const auto reduce = ngraph::builder::makeReduce(paramOuts[0], reductionAxesNode, keepDims, reductionType);
        function = makeNgraphFunction(ElementType::f32, params, reduce, "Reduce");
    }
private:
    ngraph::helpers::ReductionType reductionType;
};

TEST_P(reduceTransformationCPUTest, CompareWithRefs) {
    run();
    CheckNumberOfNodesWithType(compiledModel, "Reduce", numberOfExpectedReduce);
}

namespace {
std::vector<std::vector<ov::test::InputShape>> inputShapes = {
    {{{}, {{2, 19, 2, 9}}}}
};
const std::vector<ReductionType> reductionTypes = {
        ReductionType::Min,
        ReductionType::Max,
        ReductionType::Sum,
        ReductionType::Prod
};
const std::vector<std::vector<int>> axes = {
        {0, 1},
        {0, 2},
        {0, 3},
        {1, 2},
        {1, 3},
        {2, 3},
        {0, 1, 3},
        {0, 2, 3},
        {1, 2, 3},
        {0, 1, 2, 3}
};

const auto reduceTransformationParams = ::testing::Combine(::testing::ValuesIn(axes),
                                                           ::testing::ValuesIn(reductionTypes),
                                                           ::testing::ValuesIn(inputShapes));

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvToConvTransformationTest, reduceTransformationCPUTest,
                         reduceTransformationParams, reduceTransformationCPUTest::getTestCaseName);

} // namespace
} // namespace CPUSubgraphTestsDefinitions
