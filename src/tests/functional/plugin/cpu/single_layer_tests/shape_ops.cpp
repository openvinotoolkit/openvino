// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>

using namespace ngraph;
using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

enum class shapeNodeType {
    Reshape,
    Squeeze,
    Unsqueeze
};

std::ostream& operator<<(std::ostream & os, shapeNodeType type) {
    switch (type) {
        case shapeNodeType::Reshape:
            os << "Reshape";
            break;
        case shapeNodeType::Squeeze:
            os << "Squeeze";
            break;
        case shapeNodeType::Unsqueeze:
            os << "Unsqueeze";
            break;
    }
    return os;
}

struct inputDescription {
    InputShape inputShape;
    std::vector<std::vector<int>> data;
};

using shapeOpsParams = std::tuple<
    inputDescription,                  // input shapes
    ngraph::helpers::InputLayerType,   // second input type
    shapeNodeType,                     // node type
    Precision,                         // precision
    bool>;                             // special zero

class ShapeOpsCPUTest : public testing::WithParamInterface<shapeOpsParams>, virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<shapeOpsParams> obj) {
        inputDescription inpDesc;
        ngraph::helpers::InputLayerType secondType;
        shapeNodeType nodeType;
        Precision prc;
        bool specialZero;
        std::tie(inpDesc, secondType, nodeType, prc, specialZero) = obj.param;

        std::ostringstream result;
        result << nodeType << "_";
        result << "IS=";
        result  << CommonTestUtils::partialShape2str({inpDesc.inputShape.first}) << "_";
        result << "TS=(";
        for (const auto& shape : inpDesc.inputShape.second) {
            result << CommonTestUtils::vec2str(shape) << "_";
        }
        result << ")_";
        result << "DATA=";
        for (const auto& data : inpDesc.data) {
            result << "[" << CommonTestUtils::vec2str(data) << "]_";
        }
        result << "PRC=" << prc << "_";
        result << "specialZero=" << specialZero;

        return result.str();
    }

protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (int i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::runtime::Tensor tensor;
            if (i == 1) {
                tensor = ov::runtime::Tensor{ov::element::i32, targetInputStaticShapes[i]};
                auto inputData = tensor.data<ov::element_type_traits<ov::element::i32>::value_type>();
                for (size_t j = 0lu; j < data[idx].size(); ++j) {
                    inputData[j] =  data[idx][j];
                }
            } else {
                if (funcInput.get_element_type().is_real()) {
                    tensor = utils::create_and_fill_tensor(
                            funcInput.get_element_type(), targetInputStaticShapes[i], 10, 0, 1000);
                } else {
                    tensor = utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
                }
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

    void SetUp() override {
        idx = 0;
        targetDevice = CommonTestUtils::DEVICE_CPU;

        inputDescription inpDesc;
        ngraph::helpers::InputLayerType secondType;
        shapeNodeType nodeType;
        Precision prc;
        bool specialZero;
        std::tie(inpDesc, secondType, nodeType, prc, specialZero) = this->GetParam();

        selectedType = std::string("unknown_") + prc.name();

        data = inpDesc.data;


        std::vector<InputShape> inputShapes =
                {inpDesc.inputShape, InputShape{{static_cast<ov::Dimension::value_type>(inpDesc.data[0].size())},
                                                std::vector<Shape>(inpDesc.inputShape.second.size(), {inpDesc.data[0].size()})}};
        init_input_shapes(inputShapes);

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(prc);
        const auto secondInPrc = ngraph::element::Type_t::i32;
        auto inputs = ngraph::builder::makeDynamicParams(ngPrc, {inputDynamicShapes.front()});
        auto dataInput = inputs.front();
        dataInput->set_friendly_name("param_1");
        std::shared_ptr<ngraph::Node> secondaryInput;
        if (secondType == ngraph::helpers::InputLayerType::PARAMETER) {
            secondaryInput = ngraph::builder::makeDynamicParams(secondInPrc, {inputDynamicShapes.back()}).front();
            secondaryInput->set_friendly_name("param_2");
            inputs.push_back(std::dynamic_pointer_cast<ngraph::opset3::Parameter>(secondaryInput));
        } else {
            secondaryInput = ngraph::builder::makeConstant(secondInPrc, {inpDesc.data[0].size()}, inpDesc.data[0]);
        }

        std::shared_ptr<ngraph::Node> shapeOps;
        switch (nodeType) {
            case shapeNodeType::Reshape: {
                shapeOps = std::make_shared<ngraph::opset1::Reshape>(dataInput, secondaryInput, specialZero);
                break;
            }
            case shapeNodeType::Squeeze: {
                shapeOps = std::make_shared<ngraph::opset1::Squeeze>(dataInput, secondaryInput);
                break;
            }
            case shapeNodeType::Unsqueeze: {
                shapeOps = std::make_shared<ngraph::opset1::Unsqueeze>(dataInput, secondaryInput);
                break;
            }
        }

        function = makeNgraphFunction(ngPrc, inputs, shapeOps, "ShapeOpsCPUTest");
    }

private:
    std::vector<std::vector<int>> data;
    size_t idx;
};

TEST_P(ShapeOpsCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();
    CheckPluginRelatedResults(compiledModel, "Reshape");
}

namespace reshapeTest {

inputDescription noBounds{{{-1, -1, -1, -1},
                           {ngraph::Shape{2, 5, 7, 3}, ngraph::Shape{10, 6, 10, 5}, ngraph::Shape{10, 6, 10, 5}, ngraph::Shape{1, 2, 5, 5}}},
                           {std::vector<int>{1, -1, 0}, std::vector<int>{-1, 60, 2}, std::vector<int>{10, 30, 10}, std::vector<int>{5, 10, -1}}};

const auto params = ::testing::Combine(::testing::Values(noBounds),
                                       ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                                       ::testing::Values(shapeNodeType::Reshape),
                                       ::testing::Values(Precision::FP32),
                                       ::testing::Values(true));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_dynamic, ShapeOpsCPUTest, params, ShapeOpsCPUTest::getTestCaseName);

inputDescription noBounds_const{{{{1, 10}, {2, 6}, {1, 15}, {3, 11}},
                                 {ngraph::Shape{2, 5, 7, 3}, ngraph::Shape{10, 6, 10, 5}, ngraph::Shape{1, 2, 5, 5}}},
                                 {std::vector<int>{2, -1, 0}}};

const auto params_const = ::testing::Combine(::testing::Values(noBounds_const),
                                             ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
                                             ::testing::Values(shapeNodeType::Reshape),
                                             ::testing::Values(Precision::FP32),
                                             ::testing::Values(true));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_dynamic_const, ShapeOpsCPUTest, params_const, ShapeOpsCPUTest::getTestCaseName);

inputDescription shape_dynBatch{{{{1, 10}, 5, 7, 3},
                                 {ngraph::Shape{2, 5, 7, 3}, ngraph::Shape{10, 5, 7, 3}, ngraph::Shape{1, 5, 7, 3}}},
                                 {std::vector<int>{-1, 15, 7}}};

const auto params_dynBatch = ::testing::Combine(::testing::Values(shape_dynBatch),
                                                ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
                                                ::testing::Values(shapeNodeType::Reshape),
                                                ::testing::Values(Precision::FP32),
                                                ::testing::Values(true));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_dynBatch, ShapeOpsCPUTest, params_dynBatch, ShapeOpsCPUTest::getTestCaseName);

} // namespace reshapeTest

namespace squeezeTest {

inputDescription noBounds{{{-1, -1, -1, -1, -1, -1},
                           {
                                ngraph::Shape{2, 5, 1, 7, 3, 1},
                                ngraph::Shape{10, 1, 1, 6, 10, 5},
                                ngraph::Shape{10, 6, 10, 5, 1, 1},
                                ngraph::Shape{1, 1, 5, 1, 5}
                           }},
                           {std::vector<int>{2, 5}, std::vector<int>{1, 2}, std::vector<int>{4, 5}, std::vector<int>{0, 1}}};

const auto params = ::testing::Combine(::testing::Values(noBounds),
                                       ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                                       ::testing::Values(shapeNodeType::Squeeze),
                                       ::testing::Values(Precision::FP32),
                                       ::testing::Values(true));

// at this momemnt squeze produce dynamic output rank, if second input is not constant
// enable after CPU plug-in will support dynamic rank
// INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_dynamic, ShapeOpsCPUTest, params, ShapeOpsCPUTest::getTestCaseName);

inputDescription noBounds_const{{{{1, 10}, {1, 15}, {2, 6}, {1, 15}, {3, 11}, {1, 15}},
                                 {ngraph::Shape{2, 1, 5, 7, 3, 1}, ngraph::Shape{10, 1, 6, 10, 5, 1}, ngraph::Shape{1, 1, 2, 5, 5, 1}}},
                                 {std::vector<int>{1, 5}}};

const auto params_const = ::testing::Combine(::testing::Values(noBounds_const),
                                             ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
                                             ::testing::Values(shapeNodeType::Squeeze),
                                             ::testing::Values(Precision::FP32),
                                             ::testing::Values(true));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_dynamic_const, ShapeOpsCPUTest, params_const, ShapeOpsCPUTest::getTestCaseName);

} // namespace squeezeTest

namespace unsqueezeTest {

inputDescription noBounds{{{-1, -1, -1, -1},
                           {ngraph::Shape{2, 5, 7, 3}, ngraph::Shape{10, 6, 10, 5}, ngraph::Shape{10, 6, 10, 5}, ngraph::Shape{5, 1, 5}}},
                           {std::vector<int>{2, 5}, std::vector<int>{1, 2}, std::vector<int>{4, 5}, std::vector<int>{0, 1}}};

const auto params = ::testing::Combine(::testing::Values(noBounds),
                                       ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                                       ::testing::Values(shapeNodeType::Unsqueeze),
                                       ::testing::Values(Precision::FP32),
                                       ::testing::Values(true));

// at this momemnt unsqueze produce dynamic output rank, if second input is not constant
// enable after CPU plug-in will support dynamic rank
// INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_dynamic, ShapeOpsCPUTest, params, ShapeOpsCPUTest::getTestCaseName);

inputDescription noBounds_const{{{{1, 10}, {1, 15}, {2, 20}, {3, 7}},
                                 {ngraph::Shape{2, 5, 7, 3}, ngraph::Shape{10, 6, 10, 5}, ngraph::Shape{1, 2, 5, 5}}},
                                 {std::vector<int>{1, 3}}};

const auto params_const = ::testing::Combine(::testing::Values(noBounds_const),
                                             ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
                                             ::testing::Values(shapeNodeType::Unsqueeze),
                                             ::testing::Values(Precision::FP32),
                                             ::testing::Values(true));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_dynamic_const, ShapeOpsCPUTest, params_const, ShapeOpsCPUTest::getTestCaseName);

} // namespace unsqueezeTest

} // namespace CPULayerTestsDefinitions