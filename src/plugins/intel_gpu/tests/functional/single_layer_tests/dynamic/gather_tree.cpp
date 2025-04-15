// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_enums.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/gather_tree.hpp"

namespace {
using ov::test::InputShape;

typedef std::tuple<
    InputShape,                         // Input tensors shape
    ov::test::utils::InputLayerType,    // Secondary input type
    ov::element::Type,                  // Model type
    std::string                         // Device name
> GatherTreeGPUTestParams;

class GatherTreeLayerGPUTest : public testing::WithParamInterface<GatherTreeGPUTestParams>,
                               virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GatherTreeGPUTestParams> &obj) {
        InputShape inputShape;
        ov::element::Type_t model_type;
        ov::test::utils::InputLayerType secondaryInputType;
        std::string targetName;

        std::tie(inputShape, secondaryInputType, model_type, targetName) = obj.param;

        std::ostringstream result;
        result << "IS=" << ov::test::utils::partialShape2str({inputShape.first}) << "_";
        result << "TS=";
        for (const auto& item : inputShape.second) {
            result << ov::test::utils::vec2str(item) << "_";
        }
        result << "secondaryInputType=" << secondaryInputType << "_";
        result << "netPRC=" << model_type << "_";
        result << "trgDev=" << targetName;

        return result.str();
    }

protected:
    void SetUp() override {
        InputShape inputShape;
        ov::element::Type model_type;
        ov::test::utils::InputLayerType secondaryInputType;

        std::tie(inputShape, secondaryInputType, model_type, targetDevice) = this->GetParam();
        InputShape parentShape{inputShape};
        InputShape::first_type maxSeqLenFirst;
        if (inputShape.first.is_dynamic()) {
            maxSeqLenFirst = {inputShape.first[1]};
        }
        InputShape::second_type maxSeqLenSecond;
        maxSeqLenSecond.reserve(inputShape.second.size());
        for (const auto& item : inputShape.second) {
            maxSeqLenSecond.emplace_back(std::initializer_list<size_t>{item[1]});
        }
        InputShape maxSeqLenShape{std::move(maxSeqLenFirst), std::move(maxSeqLenSecond)};

        init_input_shapes({inputShape, parentShape, maxSeqLenShape});

        // initialization of scalar input as it cannot be done properly in init_input_shapes
        inputDynamicShapes.push_back({});
        for (auto& shape : targetStaticShapes) {
            shape.push_back({});
        }

        std::shared_ptr<ov::Node> inp2;
        std::shared_ptr<ov::Node> inp3;
        std::shared_ptr<ov::Node> inp4;

        ov::ParameterVector paramsIn{std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[0])};
        if (ov::test::utils::InputLayerType::PARAMETER == secondaryInputType) {
            auto param2 = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[1]);
            auto param3 = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[2]);
            auto param4 = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[3]);
            inp2 = param2;
            inp3 = param3;
            inp4 = param4;

            paramsIn.push_back(param2);
            paramsIn.push_back(param3);
            paramsIn.push_back(param4);
        } else if (ov::test::utils::InputLayerType::CONSTANT == secondaryInputType) {
            auto maxBeamIndex = inputShape.second.front().at(2) - 1;

            auto inp2_tensor = ov::test::utils::create_and_fill_tensor(model_type, inputShape.second.front(), maxBeamIndex);
            inp2 = std::make_shared<ov::op::v0::Constant>(inp2_tensor);
            auto inp3_tensor = ov::test::utils::create_and_fill_tensor(model_type, ov::Shape{inputShape.second.front().at(1)}, maxBeamIndex);
            inp3 = std::make_shared<ov::op::v0::Constant>(inp3_tensor);
            auto inp4_tensor = ov::test::utils::create_and_fill_tensor(model_type, ov::Shape{}, maxBeamIndex);
            inp4 = std::make_shared<ov::op::v0::Constant>(inp4_tensor);
        } else {
            throw std::runtime_error("Unsupported inputType");
        }

        auto operationResult = std::make_shared<ov::op::v1::GatherTree>(paramsIn.front(), inp2, inp3, inp4);

        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(operationResult)};
        function = std::make_shared<ov::Model>(results, paramsIn, "GatherTree");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto maxBeamIndex = targetInputStaticShapes.front().at(2) - 1;
        const auto& funcInputs = function->inputs();

        for (size_t i = 0; i < funcInputs.size(); ++i) {
            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = (i == 2 || i == 3) ? maxBeamIndex / 2 : 0;
            in_data.range = maxBeamIndex;
            auto tensor =
                ov::test::utils::create_and_fill_tensor(funcInputs[i].get_element_type(), targetInputStaticShapes[i], in_data);
            inputs.insert({funcInputs[i].get_node_shared_ptr(), tensor});
        }
    }
};

TEST_P(GatherTreeLayerGPUTest, Inference) {
    run();
}

const std::vector<ov::element::Type> model_types = {
    ov::element::f32,
    ov::element::i32
};

const std::vector<InputShape> inputDynamicShapesParameter = {
    {
        {-1, 1, -1}, {{7, 1, 10}, {8, 1, 20}}
    },
    {
        {-1, 1, {5, 10}}, {{2, 1, 7}, {5, 1, 8}}
    },
    {
        {-1, {1, 5}, 10}, {{20, 1, 10}, {17, 2, 10}}
    },
    {
        {-1, -1, -1}, {{20, 20, 15}, {30, 30, 10}}
    }
};

const std::vector<InputShape> inputDynamicShapesConstant = {
    {
        {-1, 1, -1}, {{7, 1, 10}}
    },
    {
        {-1, 1, {5, 10}}, {{2, 1, 7}}
    },
    {
        {-1, {1, 5}, 10}, {{20, 1, 10}}
    },
    {
        {-1, -1, -1}, {{20, 20, 15}}
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_gathertree_parameter_compareWithRefs_dynamic, GatherTreeLayerGPUTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(inputDynamicShapesParameter),
                            ::testing::Values(ov::test::utils::InputLayerType::PARAMETER),
                            ::testing::ValuesIn(model_types),
                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        GatherTreeLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_gathertree_constant_compareWithRefs_dynamic, GatherTreeLayerGPUTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(inputDynamicShapesConstant),
                            ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                            ::testing::ValuesIn(model_types),
                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        GatherTreeLayerGPUTest::getTestCaseName);

} // namespace
