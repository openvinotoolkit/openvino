// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/node_builders/convolution.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convolution.hpp"

namespace {
using ov::test::InputShape;

typedef std::tuple<
        std::vector<InputShape>,    // input shapes
        ov::element::Type,          // Model type
        std::string                 // Device name
> concatInPlaceDynamicBatchGPUTestParamsSet;

class ConcatInPlaceDynamicBatchGPUTest : public testing::WithParamInterface<concatInPlaceDynamicBatchGPUTestParamsSet>,
                                        virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<concatInPlaceDynamicBatchGPUTestParamsSet>& obj) {
        concatInPlaceDynamicBatchGPUTestParamsSet basicParamsSet = obj.param;
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
    void infer() override {
        if (!inferRequest) {
            inferRequest = compiledModel.create_infer_request();
        }
        for (const auto& input : inputs) {
            inferRequest.set_tensor(input.first, input.second);
        }
        inferRequest.infer();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = -1;
            in_data.range = 2;
            in_data.resolution = 32;
            tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

    void SetUp() override {
        concatInPlaceDynamicBatchGPUTestParamsSet basicParamsSet = this->GetParam();

        const auto& [inputShapes, model_type, _targetDevice] = basicParamsSet;
        targetDevice = _targetDevice;

        configuration.insert(ov::hint::inference_precision(ov::element::f16));
        abs_threshold = 0.05f;

        init_input_shapes(inputShapes);

        ov::ParameterVector inputParams;
        for (auto&& shape : inputDynamicShapes)
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(model_type, shape));

        const auto inChannels = inputShapes[0].first[1].get_length();
        const ov::Shape weightShape = {2, static_cast<size_t>(inChannels), 1, 1};

        auto makeConv = [&](const ov::element::Type &model_type, const ov::Output<ov::Node> &in, int32_t seed) {
            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = -1;
            in_data.range = 2;
            in_data.resolution = 32;
            in_data.seed = seed;
            auto tensor = ov::test::utils::create_and_fill_tensor(model_type, weightShape, in_data);
            auto constantWeightOp = std::make_shared<ov::op::v0::Constant>(tensor);
            return ov::test::utils::make_convolution(in, constantWeightOp, model_type, {1, 1}, {1, 1}, {0, 0}, {0, 0},
                                                     {1, 1}, ov::op::PadType::EXPLICIT, 2);
        };

        auto convolutionOp1 = makeConv(model_type, inputParams[0], 1);
        convolutionOp1->set_friendly_name("convolution1");

        auto convolutionOp2 = makeConv(model_type, inputParams[0], 2);
        convolutionOp2->set_friendly_name("convolution2");

        const auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector({convolutionOp1, convolutionOp2}), 1);

        auto makeFunction = [](ov::ParameterVector &params, const std::shared_ptr<ov::Node> &lastNode) {
            ov::ResultVector results;

            for (size_t i = 0; i < lastNode->get_output_size(); i++)
                results.push_back(std::make_shared<ov::op::v0::Result>(lastNode->output(i)));

            return std::make_shared<ov::Model>(results, params, "Concat");
        };
        function = makeFunction(inputParams, concat);
    }
};

TEST_P(ConcatInPlaceDynamicBatchGPUTest, Inference) {
    run();
}

const std::vector<std::vector<InputShape>> dynInputShapes = {
    {
        {{-1, 4, 4, 4}, {{1, 4, 4, 4}}},
    },
    {
        {{-1, 4, 4, 4}, {{8, 4, 4, 4}, {1, 4, 4, 4}, {8, 4, 4, 4}}},
    },
    {
        {{-1, 4, 4, 4}, {{1, 4, 4, 4}, {1, 4, 4, 4}, {1, 4, 4, 4}}},
    },
    {
        {{-1, 4, 4, 4}, {{1, 4, 4, 4}, {8, 4, 4, 4}, {3, 4, 4, 4}, {1, 4, 4, 4}}},
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_concat_in_place_dynamic_batch, ConcatInPlaceDynamicBatchGPUTest,
                        ::testing::Combine(::testing::ValuesIn(dynInputShapes),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        ConcatInPlaceDynamicBatchGPUTest::getTestCaseName);

}  // namespace
