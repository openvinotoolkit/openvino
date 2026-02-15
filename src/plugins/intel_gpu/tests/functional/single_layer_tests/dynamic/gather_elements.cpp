// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/gather_elements.hpp"

namespace {
using ov::test::InputShape;

using GatherElementsParams = std::tuple<
        std::vector<InputShape>,           // Dynamic shape + Target static shapes
        int,                               // Axis
        ov::element::Type,                 // Data type
        ov::element::Type,                 // Indices type
        std::string>;                      // Device name

class GatherElementsGPUTest : public testing::WithParamInterface<GatherElementsParams>,
                              virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GatherElementsParams>& obj) {
        const auto& [shapes, axis, data_type, indices_type, device] = obj.param;

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
        result << "DP=" << data_type << "_";
        result << "IP=" << indices_type << "_";
        result << "device=" << device;

        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;

            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = 0;
            in_data.range = 15;
            in_data.resolution = 32768;
            tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);

            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

protected:
    void SetUp() override {
        const auto& [shapes, axis, data_type, indices_type, _targetDevice] = this->GetParam();
        targetDevice = _targetDevice;
        init_input_shapes(shapes);

        ov::ParameterVector params = {
            std::make_shared<ov::op::v0::Parameter>(data_type, inputDynamicShapes[0]),
            std::make_shared<ov::op::v0::Parameter>(indices_type, inputDynamicShapes[1]),
        };

        auto gather = std::make_shared<ov::op::v6::GatherElements>(params[0], params[1], axis);

        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(gather)};
        function = std::make_shared<ov::Model>(results, params, "GatherElements");
    }
};

TEST_P(GatherElementsGPUTest, Inference) {
    run();
}

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
                    ::testing::ValuesIn(std::vector<ov::element::Type>({ov::element::f16, ov::element::f32})),
                    ::testing::Values(ov::element::i32),
                    ::testing::Values(ov::test::utils::DEVICE_GPU)),
        GatherElementsGPUTest::getTestCaseName);

} // namespace
