// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "common_test_utils/test_constants.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_enums.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {
using ov::test::InputShape;

struct StridedSliceParams {
    std::vector<int64_t> begin;
    std::vector<int64_t> end;
    std::vector<int64_t> stride;
    std::vector<int64_t> beginMask;
    std::vector<int64_t> endMask;
    std::vector<int64_t> newAxisMask;
    std::vector<int64_t> shrinkAxisMask;
    std::vector<int64_t> ellipsisAxisMask;
};

typedef std::tuple<
        InputShape,                                     // Input shapes
        StridedSliceParams,
        ov::element::Type,                              // Element type
        std::vector<ov::test::utils::InputLayerType>,   // begin/end/stride input type
        std::map<std::string, std::string>              // Additional network configuration
> StridedSliceLayerParamSet;

class DynamicShapeHugeRangeGPUTest : public testing::WithParamInterface<StridedSliceLayerParamSet>,
                                     virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<StridedSliceLayerParamSet>& obj) {
        InputShape shapes;
        StridedSliceParams params;
        ov::element::Type model_type;
        std::vector<ov::test::utils::InputLayerType> restInputType;
        std::map<std::string, std::string> additionalConfig;
        std::tie(shapes, params, model_type, restInputType, additionalConfig) = obj.param;

        std::ostringstream results;
        results << "IS=" << ov::test::utils::partialShape2str({shapes.first}) << "_";
        results << "TS=";
        for (const auto& item : shapes.second) {
            results << ov::test::utils::vec2str(item) << "_";
        }
        results << "modelType=" << model_type << "_";
        results << "begin=" << ov::test::utils::vec2str(params.begin) << "_";
        results << "end=" << ov::test::utils::vec2str(params.end) << "_";
        results << "stride=" << ov::test::utils::vec2str(params.stride) << "_";
        results << "begin_m=" << ov::test::utils::vec2str(params.beginMask) << "_";
        results << "end_m=" << ov::test::utils::vec2str(params.endMask) << "_";
        results << "new_axis_m=" << (params.newAxisMask.empty() ? "def" : ov::test::utils::vec2str(params.newAxisMask)) << "_";
        results << "shrink_m=" << (params.shrinkAxisMask.empty() ? "def" : ov::test::utils::vec2str(params.shrinkAxisMask)) << "_";
        results << "ellipsis_m=" << (params.ellipsisAxisMask.empty() ? "def" : ov::test::utils::vec2str(params.ellipsisAxisMask)) << "_";
        results << "beginType=" << restInputType[0] << "_";
        results << "endType=" << restInputType[1] << "_";
        results << "strideType=" << restInputType[2] << "_";
        results << "config=(";
        for (const auto& configEntry : additionalConfig) {
            results << configEntry.first << ", " << configEntry.second << ":";
        }
        results << ")";

        return results.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        ov::Tensor tensor;

        // input0: data
        int32_t idx = 0;
        tensor = ov::test::utils::create_and_fill_tensor(funcInputs[idx].get_element_type(), targetInputStaticShapes[idx]);
        inputs.insert({funcInputs[idx].get_node_shared_ptr(), tensor});

        // input1: begin
        if (restInputType[0] == ov::test::utils::InputLayerType::PARAMETER) {
            idx += 1;
            tensor = ov::Tensor(funcInputs[idx].get_element_type(), targetInputStaticShapes[idx]);
            auto *dataPtr = tensor.data<float>();
            for (size_t i = 0; i < begin.size(); i++) {
                dataPtr[i] = static_cast<float>(begin[i]);
            }
            inputs.insert({funcInputs[idx].get_node_shared_ptr(), tensor});
        }

        // input2: end
        if (restInputType[1] == ov::test::utils::InputLayerType::PARAMETER) {
            idx += 1;
            tensor = ov::Tensor(funcInputs[idx].get_element_type(), targetInputStaticShapes[idx]);
            auto *dataPtr = tensor.data<float>();
            for (size_t i = 0; i < end.size(); i++) {
                dataPtr[i] = static_cast<float>(end[i]);
            }
            inputs.insert({funcInputs[idx].get_node_shared_ptr(), tensor});
        }

        // input3: stride
        if (restInputType[2] == ov::test::utils::InputLayerType::PARAMETER) {
            idx += 1;
            tensor = ov::Tensor(funcInputs[idx].get_element_type(), targetInputStaticShapes[idx]);
            auto *dataPtr = tensor.data<float>();
            for (size_t i = 0; i < stride.size(); i++) {
                dataPtr[i] = static_cast<float>(stride[i]);
            }
            inputs.insert({funcInputs[idx].get_node_shared_ptr(), tensor});
        }

        inferRequestNum++;
    }

protected:
    std::vector<int64_t> begin;
    std::vector<int64_t> end;
    std::vector<int64_t> stride;
    std::vector<ov::test::utils::InputLayerType> restInputType;
    size_t inferRequestNum = 0;
    bool exception;

    void SetUp() override {
        InputShape shapes;
        StridedSliceParams ssParams;
        std::map<std::string, std::string> additionalConfig;
        std::tie(shapes, ssParams, inType, restInputType, additionalConfig) = this->GetParam();

        begin = ssParams.begin;
        end = ssParams.end;
        stride = ssParams.stride;

        targetDevice = ov::test::utils::DEVICE_GPU;

        std::vector<InputShape> inputShapes;
        inputShapes.push_back(shapes);
        if (restInputType[0] == ov::test::utils::InputLayerType::PARAMETER)
            inputShapes.push_back(InputShape({static_cast<int64_t>(begin.size())}, std::vector<ov::Shape>(shapes.second.size(), {begin.size()})));
        if (restInputType[1] == ov::test::utils::InputLayerType::PARAMETER)
            inputShapes.push_back(InputShape({static_cast<int64_t>(end.size())}, std::vector<ov::Shape>(shapes.second.size(), {end.size()})));
        if (restInputType[2] == ov::test::utils::InputLayerType::PARAMETER)
            inputShapes.push_back(InputShape({static_cast<int64_t>(stride.size())}, std::vector<ov::Shape>(shapes.second.size(), {stride.size()})));

        init_input_shapes(inputShapes);

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.front())};

        std::shared_ptr<ov::Node> beginInput, endInput, strideInput;
        if (restInputType[0] == ov::test::utils::InputLayerType::PARAMETER) {
            auto beginNode = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{begin.size()});
            params.push_back(beginNode);
            beginInput = beginNode;
        } else {
            beginInput = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{begin.size()}, begin);
        }

        if (restInputType[1] == ov::test::utils::InputLayerType::PARAMETER) {
            auto endNode = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{end.size()});
            params.push_back(endNode);
            endInput = endNode;
        } else {
            endInput = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{end.size()}, end);
        }

        if (restInputType[2] == ov::test::utils::InputLayerType::PARAMETER) {
            auto strideNode = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{stride.size()});
            params.push_back(strideNode);
            strideInput = strideNode;
        } else {
            strideInput = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{stride.size()}, stride);
        }

        auto stridedSliceOp = std::make_shared<ov::op::v1::StridedSlice>(params[0], beginInput, endInput, strideInput, ssParams.beginMask, ssParams.endMask,
                                                                 ssParams.newAxisMask, ssParams.shrinkAxisMask, ssParams.ellipsisAxisMask);

        auto shapeOfOp = std::make_shared<ov::op::v3::ShapeOf>(stridedSliceOp, ov::element::i32);

        ov::ResultVector results;
        for (size_t i = 0; i < shapeOfOp->get_output_size(); i++) {
            results.push_back(std::make_shared<ov::op::v0::Result>(shapeOfOp->output(i)));
        }

        function = std::make_shared<ov::Model>(results, params, "result");

        set_callback_exception([this](const std::exception& exp) {
            exception = true;
        });
    }
};

TEST_P(DynamicShapeHugeRangeGPUTest, Inference) {
    run();
}

std::map<std::string, std::string> emptyAdditionalConfig;

const std::vector<ov::element::Type> model_types = {
        ov::element::f32
};

const std::vector<std::vector<ov::test::utils::InputLayerType>> restInputTypes = {
    {ov::test::utils::InputLayerType::CONSTANT, ov::test::utils::InputLayerType::CONSTANT, ov::test::utils::InputLayerType::CONSTANT},
    {ov::test::utils::InputLayerType::PARAMETER, ov::test::utils::InputLayerType::PARAMETER, ov::test::utils::InputLayerType::PARAMETER},
    {ov::test::utils::InputLayerType::PARAMETER, ov::test::utils::InputLayerType::CONSTANT, ov::test::utils::InputLayerType::CONSTANT},
    {ov::test::utils::InputLayerType::CONSTANT, ov::test::utils::InputLayerType::PARAMETER, ov::test::utils::InputLayerType::CONSTANT},
    {ov::test::utils::InputLayerType::CONSTANT, ov::test::utils::InputLayerType::CONSTANT, ov::test::utils::InputLayerType::PARAMETER},
    {ov::test::utils::InputLayerType::CONSTANT, ov::test::utils::InputLayerType::PARAMETER, ov::test::utils::InputLayerType::PARAMETER},
    {ov::test::utils::InputLayerType::PARAMETER, ov::test::utils::InputLayerType::CONSTANT, ov::test::utils::InputLayerType::PARAMETER},
    {ov::test::utils::InputLayerType::PARAMETER, ov::test::utils::InputLayerType::PARAMETER, ov::test::utils::InputLayerType::CONSTANT},
};

const std::vector<InputShape> inputShapesDynamic2D_excessive_uppper_boundary = {
        {{{0, 1000}, {0, 364000000}, 4},
         {{640, 640, 4}}},
};

const std::vector<StridedSliceParams> paramsPlain2D_excessive_uppper_boundary = {
        StridedSliceParams{ { 0, 1 }, { 0, 2147483647 }, { 1, 1 }, { 1, 0 }, { 1, 0 },  { },  { },  { } },
};

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Dynamic_2D_excessive_uppper_boundary, DynamicShapeHugeRangeGPUTest,
                         ::testing::Combine(
                             ::testing::ValuesIn(inputShapesDynamic2D_excessive_uppper_boundary),
                             ::testing::ValuesIn(paramsPlain2D_excessive_uppper_boundary),
                             ::testing::ValuesIn(model_types),
                             ::testing::Values(restInputTypes[0]),
                             ::testing::Values(emptyAdditionalConfig)),
                         DynamicShapeHugeRangeGPUTest::getTestCaseName);
} // namespace
