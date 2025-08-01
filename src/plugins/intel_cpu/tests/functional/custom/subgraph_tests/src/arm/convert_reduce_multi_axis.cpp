// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/node_builders/reduce.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

typedef std::tuple<std::vector<int>,                // Axis to reduce order
                   ov::test::utils::ReductionType,  // Reduce operation type
                   std::vector<InputShape>          // Input shapes
                   >
    reduceConvertCPUTestParamsSet;

class reduceTransformationCPUTest: public testing::WithParamInterface<reduceConvertCPUTestParamsSet>,
                                            virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<reduceConvertCPUTestParamsSet> obj) {
        const auto& [axes, reductionType, inputShapes] = obj.param;
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
        bool keepDims = true;
        const auto& [axes, _reductionType, inputShapes] = this->GetParam();
        reductionType = _reductionType;
        numberOfExpectedReduce = axes.size();

        init_input_shapes(inputShapes);

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape));
        }
        std::vector<size_t> shapeAxes;
        shapeAxes.push_back(axes.size());
        auto reductionAxesNode = std::dynamic_pointer_cast<ov::Node>(
            std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64, ov::Shape(shapeAxes), axes));

        const auto reduce = utils::make_reduce(params[0], reductionAxesNode, keepDims, reductionType);
        function = makeNgraphFunction(ElementType::f32, params, reduce, "Reduce");
    }
private:
    utils::ReductionType reductionType;
};

TEST_P(reduceTransformationCPUTest, CompareWithRefs) {
    run();
    CheckNumberOfNodesWithType(compiledModel, "Reduce", numberOfExpectedReduce);
}

namespace {
std::vector<std::vector<ov::test::InputShape>> inputShapes = {
    {{{}, {{2, 19, 2, 9}}}}
};
const std::vector<utils::ReductionType> reductionTypes = {
        utils::ReductionType::Min,
        utils::ReductionType::Max,
        utils::ReductionType::Sum,
        utils::ReductionType::Prod
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

}  // namespace
}  // namespace test
}  // namespace ov
