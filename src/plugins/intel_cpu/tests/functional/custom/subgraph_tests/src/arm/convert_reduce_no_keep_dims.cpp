// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/node_builders/reduce.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/general_utils.h"

using namespace CPUTestUtils;
using namespace ov::intel_cpu;

namespace ov {
namespace test {

typedef std::tuple<int,                             // Axe to reduce order
                   ov::test::utils::ReductionType,  // Reduce operation type
                   std::vector<InputShape>          // Input shapes
                   >
    ReduceConvertCPUTestParamsSet;

class ReduceNoKeepDimsTransformationCPUTest: public testing::WithParamInterface<ReduceConvertCPUTestParamsSet>,
                                             virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ReduceConvertCPUTestParamsSet> obj) {
        std::vector<InputShape> inputShapes;
        int axe;
        utils::ReductionType reductionType;
        std::tie(axe, reductionType, inputShapes) = obj.param;

        std::ostringstream result;
        result << "type=" << reductionType << "_";
        result << "IS=(";
        for (const auto& shape : inputShapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << ")_axe=" << axe << "_";
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        int axe;
        bool keepDims = false;
        std::vector<InputShape> inputShapes;
        std::tie(axe, reductionType, inputShapes) = this->GetParam();
        ov::element::Type_t dataType = one_of(reductionType, utils::ReductionType::LogicalAnd, utils::ReductionType::LogicalOr) ?
                                            ov::element::boolean : ov::element::f32;

        init_input_shapes(inputShapes);
        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(dataType, shape));
        }
        auto reductionAxesNode = std::dynamic_pointer_cast<ov::Node>(
            std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64, ov::Shape{}, axe));

        const auto reduce = utils::make_reduce(params[0], reductionAxesNode, keepDims, reductionType);
        function = makeNgraphFunction(dataType, params, reduce, "Reduce");
    }
private:
    utils::ReductionType reductionType;
};

TEST_P(ReduceNoKeepDimsTransformationCPUTest, CompareWithRefs) {
    run();
    CheckNumberOfNodesWithType(compiledModel, "Reshape", 1); //Squeeze turns into Reshape
    CheckNumberOfNodesWithType(compiledModel, "Reduce", 1);

    std::shared_ptr<const ov::Model> function = compiledModel.get_runtime_model();
    for (const auto& node : function->get_ops()) {
        if (ov::is_type<ov::op::util::LogicalReductionKeepDims>(node)) {
            auto reduce_node = std::dynamic_pointer_cast<ov::op::util::LogicalReductionKeepDims>(node);
            EXPECT_TRUE(reduce_node->get_keep_dims());
            break;
        } else if (ov::is_type<ov::op::util::ArithmeticReductionKeepDims>(node)) {
            auto reduce_node = std::dynamic_pointer_cast<ov::op::util::ArithmeticReductionKeepDims>(node);
            EXPECT_TRUE(reduce_node->get_keep_dims());
            break;
        }
    }
}

namespace {
std::vector<std::vector<ov::test::InputShape>> inputShapes = {
    {{{}, {{2, 19, 2, 9}}}}
};
const std::vector<utils::ReductionType> reductionTypes = {
        utils::ReductionType::Min,
        utils::ReductionType::Max,
        utils::ReductionType::Sum,
        utils::ReductionType::Prod,
        utils::ReductionType::LogicalAnd,
        utils::ReductionType::LogicalOr
};

const std::vector<int> axes = {0, 1, 2, 3};

const auto reduceTransformationParams = ::testing::Combine(::testing::ValuesIn(axes),
                                                                     ::testing::ValuesIn(reductionTypes),
                                                                     ::testing::ValuesIn(inputShapes));

INSTANTIATE_TEST_SUITE_P(smoke_ReduceNoKeepDimsTransformationTest, ReduceNoKeepDimsTransformationCPUTest,
                         reduceTransformationParams, ReduceNoKeepDimsTransformationCPUTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
