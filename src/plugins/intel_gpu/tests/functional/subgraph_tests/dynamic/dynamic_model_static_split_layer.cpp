// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/node_builders/eltwise.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/variadic_split.hpp"

namespace {
using ov::test::InputShape;

typedef std::tuple<
        std::vector<InputShape>,           // input shapes
        ov::element::Type,                 // Model type
        std::string                       // Device name
> DynamicModelStaticSplitLayerGPUTestParamsSet;

const std::vector<ov::element::Type> model_types = {
    ov::element::f16
};

class DynamicModelStaticSplitLayerGPUTest : public testing::WithParamInterface<DynamicModelStaticSplitLayerGPUTestParamsSet>,
                                            virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<DynamicModelStaticSplitLayerGPUTestParamsSet>& obj) {
        DynamicModelStaticSplitLayerGPUTestParamsSet basicParamsSet = obj.param;
        std::ostringstream result;

        const auto& [inputShapes, model_type, targetDevice] = basicParamsSet;
        result << "IS=";
        for (const auto& shape : inputShapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
            for (const auto& actual_shape : shape.second) {
                result << ov::test::utils::partialShape2str({actual_shape}) << "_";
            }
        }
        result << "NetType=" << model_type << "_";
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = 0;
            in_data.range = 80;
            in_data.resolution = 8;
            tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

    void SetUp() override {
        DynamicModelStaticSplitLayerGPUTestParamsSet basicParamsSet = this->GetParam();

        const auto& [inputShapes, model_type, _targetDevice] = basicParamsSet;
        targetDevice = _targetDevice;

        init_input_shapes(inputShapes);
        const auto inShapSplit = inputDynamicShapes[0];
        const auto inShapeElt = inputDynamicShapes[1];
        ov::ParameterVector params;
        for (auto&& shape : {inShapSplit, inShapeElt})
            params.push_back(std::make_shared<ov::op::v0::Parameter>(model_type, shape));

        auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
        axis->set_friendly_name("axis");

        auto split_sizes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 1});
        split_sizes->set_friendly_name("split_sizes");

        auto variadicSplitOp = std::make_shared<ov::op::v1::VariadicSplit>(params[0], axis, split_sizes);
        variadicSplitOp->set_friendly_name("variadicSplit");

        auto addOp = ov::test::utils::make_eltwise(params[1], variadicSplitOp->output(1), ov::test::utils::EltwiseTypes::ADD);
        addOp->set_friendly_name("add");

        ov::ResultVector results = {std::make_shared<ov::op::v0::Result>(addOp)};
        function = std::make_shared<ov::Model>(results, params, "eltwise_add_out");
    }
};

TEST_P(DynamicModelStaticSplitLayerGPUTest, Inference) {
    run();
}

const std::vector<std::vector<ov::test::InputShape>> dynInputShapes = {
    {
        // Input for static VariadicSplit
        {{2}, {{2}, {2}}},
        // Input for dynamic Add
        {{-1, -1}, {{5, 10}, {2, 3}}},
    },
    {
        // Input for static VariadicSplit
        {{2, 5}, {{2, 5}, {2, 5}}},
        // Input for dynamic Add
        {{-1, -1, -1}, {{1, 5, 1}, {1, 1, 1}}},
    },
};


const auto testParams_smoke = ::testing::Combine(::testing::ValuesIn(dynInputShapes),
                                                   ::testing::ValuesIn(model_types), // netprec
                                                   ::testing::Values(ov::test::utils::DEVICE_GPU));

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_model_static_split, DynamicModelStaticSplitLayerGPUTest,
                         testParams_smoke, DynamicModelStaticSplitLayerGPUTest::getTestCaseName);
} // namespace
