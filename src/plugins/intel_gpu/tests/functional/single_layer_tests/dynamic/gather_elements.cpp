// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ie_precision.hpp"
#include "ov_models/builders.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include <string>

using namespace ov::test;
using namespace ngraph;
using namespace InferenceEngine;
using namespace ngraph::helpers;

namespace GPULayerTestsDefinitions  {

using GatherElementsParams = std::tuple<
        std::vector<InputShape>,           // Dynamic shape + Target static shapes
        int,                               // Axis
        ElementType,                       // Data precision
        ElementType,                       // Indices precision
        TargetDevice                       // Device name
>;

class GatherElementsGPUTest : public testing::WithParamInterface<GatherElementsParams>,
                              virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GatherElementsParams>& obj) {
        std::vector<InputShape> shapes;
        ElementType dPrecision, iPrecision;
        int axis;
        std::string device;
        std::tie(shapes, axis, dPrecision, iPrecision, device) = obj.param;

        std::ostringstream result;
        result << "IS=(";
        for (const auto& shape : shapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << ")_TS=(";
        for (const auto& shape : shapes) {
            for (const auto& item : shape.second) {
                result << ov::test::utils::vec2str(item) << "_";
            }
        }
        result << "Ax=" << axis << "_";
        result << "DP=" << dPrecision << "_";
        result << "IP=" << iPrecision << "_";
        result << "device=" << device;

        return result.str();
    }

    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;

            tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], 15, 0, 32768);

            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

protected:
    void SetUp() override {
        std::vector<InputShape> shapes;
        ElementType dPrecision, iPrecision;
        int axis;
        std::tie(shapes, axis, dPrecision, iPrecision, targetDevice) = this->GetParam();
        init_input_shapes(shapes);

        ngraph::ParameterVector params = {
            std::make_shared<ngraph::opset1::Parameter>(dPrecision, inputDynamicShapes[0]),
            std::make_shared<ngraph::opset1::Parameter>(iPrecision, inputDynamicShapes[1]),
        };

        auto gather = std::make_shared<ngraph::op::v6::GatherElements>(params[0], params[1], axis);

        ngraph::ResultVector results{std::make_shared<ngraph::opset4::Result>(gather)};
        function = std::make_shared<ngraph::Function>(results, params, "GatherElements");
    }
};

TEST_P(GatherElementsGPUTest, CompareWithRefs) {
    run();
}

namespace {

const std::vector<std::vector<InputShape>> inDynamicShapeParams = {
    {{{-1, -1, -1, -1}, {{2, 3, 5, 7}, {3, 4, 6, 8}}},
     {{-1, -1, -1, -1}, {{2, 3, 9, 7}, {3, 4, 4, 8}}}},
    {{{{1, 10}, {1, 10}, {1, 10}, {1, 10}}, {{3, 4, 6, 8}, {2, 3, 5, 7}}},
     {{{1, 10}, {1, 10}, {1, 10}, {1, 10}}, {{3, 4, 4, 8}, {2, 3, 9, 7}}}}
};

INSTANTIATE_TEST_SUITE_P(smoke_set1, GatherElementsGPUTest,
                ::testing::Combine(
                    ::testing::ValuesIn(inDynamicShapeParams),                // shape
                    ::testing::ValuesIn(std::vector<int>({2, -2})),           // Axis
                    ::testing::ValuesIn(std::vector<ElementType>({ElementType::f16, ElementType::f32})),
                    ::testing::Values(ElementType::i32),
                    ::testing::Values(ov::test::utils::DEVICE_GPU)),
        GatherElementsGPUTest::getTestCaseName);

} // namespace
} // namespace GPULayerTestsDefinitions
