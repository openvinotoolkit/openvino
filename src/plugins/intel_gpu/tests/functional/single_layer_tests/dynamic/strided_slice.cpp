// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_enums.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/strided_slice.hpp"

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
        std::vector<ov::test::utils::InputLayerType>   // begin/end/stride input type
> StridedSliceLayerParamSet;

class StridedSliceLayerGPUTest : public testing::WithParamInterface<StridedSliceLayerParamSet>,
                                 virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<StridedSliceLayerParamSet>& obj) {
        InputShape shapes;
        StridedSliceParams params;
        ov::element::Type model_type;
        std::vector<ov::test::utils::InputLayerType> rest_input_type;
        std::string targetDevice;
        std::tie(shapes, params, model_type, rest_input_type) = obj.param;

        std::ostringstream results;
        results << "IS=" << ov::test::utils::partialShape2str({shapes.first}) << "_";
        results << "TS=";
        for (const auto& item : shapes.second) {
            results << ov::test::utils::vec2str(item) << "_";
        }
        results << "netPRC=" << model_type << "_";
        results << "begin=" << ov::test::utils::vec2str(params.begin) << "_";
        results << "end=" << ov::test::utils::vec2str(params.end) << "_";
        results << "stride=" << ov::test::utils::vec2str(params.stride) << "_";
        results << "begin_m=" << ov::test::utils::vec2str(params.beginMask) << "_";
        results << "end_m=" << ov::test::utils::vec2str(params.endMask) << "_";
        results << "new_axis_m=" << (params.newAxisMask.empty() ? "def" : ov::test::utils::vec2str(params.newAxisMask)) << "_";
        results << "shrink_m=" << (params.shrinkAxisMask.empty() ? "def" : ov::test::utils::vec2str(params.shrinkAxisMask)) << "_";
        results << "ellipsis_m=" << (params.ellipsisAxisMask.empty() ? "def" : ov::test::utils::vec2str(params.ellipsisAxisMask)) << "_";
        results << "beginType=" << rest_input_type[0] << "_";
        results << "endType=" << rest_input_type[1] << "_";
        results << "strideType=" << rest_input_type[2];

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
        if (rest_input_type[0] == ov::test::utils::InputLayerType::PARAMETER) {
            idx += 1;
            tensor = ov::Tensor(funcInputs[idx].get_element_type(), targetInputStaticShapes[idx]);
            auto *dataPtr = tensor.data<float>();
            for (size_t i = 0; i < begin.size(); i++) {
                dataPtr[i] = static_cast<float>(begin[i]);
            }
            inputs.insert({funcInputs[idx].get_node_shared_ptr(), tensor});
        }

        // input2: end
        if (rest_input_type[1] == ov::test::utils::InputLayerType::PARAMETER) {
            idx += 1;
            tensor = ov::Tensor(funcInputs[idx].get_element_type(), targetInputStaticShapes[idx]);
            auto *dataPtr = tensor.data<float>();
            for (size_t i = 0; i < end.size(); i++) {
                dataPtr[i] = static_cast<float>(end[i]);
            }
            inputs.insert({funcInputs[idx].get_node_shared_ptr(), tensor});
        }

        // input3: stride
        if (rest_input_type[2] == ov::test::utils::InputLayerType::PARAMETER) {
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
    std::vector<ov::test::utils::InputLayerType> rest_input_type;
    size_t inferRequestNum = 0;

    void SetUp() override {
        InputShape shapes;
        StridedSliceParams ssParams;
        std::tie(shapes, ssParams, inType, rest_input_type) = this->GetParam();

        begin = ssParams.begin;
        end = ssParams.end;
        stride = ssParams.stride;

        targetDevice = ov::test::utils::DEVICE_GPU;

        std::vector<InputShape> inputShapes;
        inputShapes.push_back(shapes);
        if (rest_input_type[0] == ov::test::utils::InputLayerType::PARAMETER)
            inputShapes.push_back(InputShape({static_cast<int64_t>(begin.size())}, std::vector<ov::Shape>(shapes.second.size(), {begin.size()})));
        if (rest_input_type[1] == ov::test::utils::InputLayerType::PARAMETER)
            inputShapes.push_back(InputShape({static_cast<int64_t>(end.size())}, std::vector<ov::Shape>(shapes.second.size(), {end.size()})));
        if (rest_input_type[2] == ov::test::utils::InputLayerType::PARAMETER)
            inputShapes.push_back(InputShape({static_cast<int64_t>(stride.size())}, std::vector<ov::Shape>(shapes.second.size(), {stride.size()})));

        init_input_shapes(inputShapes);

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.front())};

        std::shared_ptr<ov::Node> beginInput, endInput, strideInput;
        if (rest_input_type[0] == ov::test::utils::InputLayerType::PARAMETER) {
            auto beginNode = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{begin.size()});
            params.push_back(beginNode);
            beginInput = beginNode;
        } else {
            beginInput = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{begin.size()}, begin);
        }

        if (rest_input_type[1] == ov::test::utils::InputLayerType::PARAMETER) {
            auto endNode = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{end.size()});
            params.push_back(endNode);
            endInput = endNode;
        } else {
            endInput = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{end.size()}, end);
        }

        if (rest_input_type[2] == ov::test::utils::InputLayerType::PARAMETER) {
            auto strideNode = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{stride.size()});
            params.push_back(strideNode);
            strideInput = strideNode;
        } else {
            strideInput = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{stride.size()}, stride);
        }

        auto ss = std::make_shared<ov::op::v1::StridedSlice>(params[0], beginInput, endInput, strideInput, ssParams.beginMask, ssParams.endMask,
                                                                 ssParams.newAxisMask, ssParams.shrinkAxisMask, ssParams.ellipsisAxisMask);

        ov::ResultVector results;
        for (size_t i = 0; i < ss->get_output_size(); i++) {
            results.push_back(std::make_shared<ov::op::v0::Result>(ss->output(i)));
        }

        function = std::make_shared<ov::Model>(results, params, "StridedSlice");
    }
};

class StridedSliceLayerReshapeGPUTest : virtual public StridedSliceLayerGPUTest {
public:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        ov::Tensor tensor;

        // input0: data
        int32_t idx = 0;
        tensor = ov::test::utils::create_and_fill_tensor(funcInputs[idx].get_element_type(), targetInputStaticShapes[idx]);
        inputs.insert({funcInputs[idx].get_node_shared_ptr(), tensor});

        // input1: begin
        if (rest_input_type[0] == ov::test::utils::InputLayerType::PARAMETER) {
            idx += 1;
            tensor = ov::Tensor(funcInputs[idx].get_element_type(), targetInputStaticShapes[idx]);
            auto *dataPtr = tensor.data<float>();
            for (size_t i = 0; i < begin.size(); i++) {
                dataPtr[i] = static_cast<float>(begin[i]);
            }
            inputs.insert({funcInputs[idx].get_node_shared_ptr(), tensor});
        }

        // input2: end
        if (rest_input_type[1] == ov::test::utils::InputLayerType::PARAMETER) {
            idx += 1;
            tensor = ov::Tensor(funcInputs[idx].get_element_type(), targetInputStaticShapes[idx]);
            auto *dataPtr = tensor.data<float>();
            for (size_t i = 0; i < end.size(); i++) {
                dataPtr[i] = static_cast<float>(end[i]);
            }
            inputs.insert({funcInputs[idx].get_node_shared_ptr(), tensor});
        }

        // input3: stride
        if (rest_input_type[2] == ov::test::utils::InputLayerType::PARAMETER) {
            idx += 1;
            tensor = ov::Tensor(funcInputs[idx].get_element_type(), targetInputStaticShapes[idx]);
            auto *dataPtr = tensor.data<float>();
            for (size_t i = 0; i < stride.size(); i++) {
                dataPtr[i] = static_cast<float>(stride[i]);
            }
            inputs.insert({funcInputs[idx].get_node_shared_ptr(), tensor});
        }

        // data for input_1d
        idx += 1;
        tensor = ov::test::utils::create_and_fill_tensor(funcInputs[idx].get_element_type(), targetInputStaticShapes[idx]);
        inputs.insert({funcInputs[idx].get_node_shared_ptr(), tensor});

        inferRequestNum++;
    }

protected:
    std::vector<int64_t> begin;
    std::vector<int64_t> end;
    std::vector<int64_t> stride;
    std::vector<ov::test::utils::InputLayerType> rest_input_type;
    size_t inferRequestNum = 0;

    void SetUp() override {
        InputShape shapes;
        StridedSliceParams ssParams;
        std::tie(shapes, ssParams, inType, rest_input_type) = this->GetParam();

        begin = ssParams.begin;
        end = ssParams.end;
        stride = ssParams.stride;

        targetDevice = ov::test::utils::DEVICE_GPU;

        std::vector<InputShape> inputShapes;

        ov::PartialShape input_1d_shape{1024};

        inputShapes.push_back(shapes);
        if (rest_input_type[0] == ov::test::utils::InputLayerType::PARAMETER)
            inputShapes.push_back(InputShape({static_cast<int64_t>(begin.size())}, std::vector<ov::Shape>(shapes.second.size(), {begin.size()})));
        if (rest_input_type[1] == ov::test::utils::InputLayerType::PARAMETER)
            inputShapes.push_back(InputShape({static_cast<int64_t>(end.size())}, std::vector<ov::Shape>(shapes.second.size(), {end.size()})));
        if (rest_input_type[2] == ov::test::utils::InputLayerType::PARAMETER)
            inputShapes.push_back(InputShape({static_cast<int64_t>(stride.size())}, std::vector<ov::Shape>(shapes.second.size(), {stride.size()})));

        inputShapes.push_back(InputShape(input_1d_shape, {input_1d_shape.to_shape()}));

        init_input_shapes(inputShapes);

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.front())};

        std::shared_ptr<ov::Node> beginInput, endInput, strideInput;
        if (rest_input_type[0] == ov::test::utils::InputLayerType::PARAMETER) {
            auto beginNode = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{begin.size()});
            params.push_back(beginNode);
            beginInput = beginNode;
        } else {
            beginInput = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{begin.size()}, begin);
        }

        if (rest_input_type[1] == ov::test::utils::InputLayerType::PARAMETER) {
            auto endNode = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{end.size()});
            params.push_back(endNode);
            endInput = endNode;
        } else {
            endInput = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{end.size()}, end);
        }

        if (rest_input_type[2] == ov::test::utils::InputLayerType::PARAMETER) {
            auto strideNode = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{stride.size()});
            params.push_back(strideNode);
            strideInput = strideNode;
        } else {
            strideInput = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{stride.size()}, stride);
        }

        auto input_1d = std::make_shared<ov::op::v0::Parameter>(inType, input_1d_shape.to_shape());
        params.push_back(input_1d);

        auto ss = std::make_shared<ov::op::v1::StridedSlice>(input_1d, beginInput, endInput, strideInput, ssParams.beginMask, ssParams.endMask,
                                                                 ssParams.newAxisMask, ssParams.shrinkAxisMask, ssParams.ellipsisAxisMask);
        auto mul = std::make_shared<ov::op::v1::Multiply>(params[0], ss);

        ov::ResultVector results;
        for (size_t i = 0; i < mul->get_output_size(); i++) {
            results.push_back(std::make_shared<ov::op::v0::Result>(mul->output(i)));
        }

        function = std::make_shared<ov::Model>(results, params, "StridedSlice");
    }
};

TEST_P(StridedSliceLayerGPUTest, Inference) {
    run();
}

TEST_P(StridedSliceLayerReshapeGPUTest, Inference) {
    run();
}

const std::vector<ov::element::Type> model_types = {
        ov::element::f32
};

const std::vector<std::vector<ov::test::utils::InputLayerType>> rest_input_types = {
    {ov::test::utils::InputLayerType::CONSTANT, ov::test::utils::InputLayerType::CONSTANT, ov::test::utils::InputLayerType::CONSTANT},
    {ov::test::utils::InputLayerType::PARAMETER, ov::test::utils::InputLayerType::PARAMETER, ov::test::utils::InputLayerType::PARAMETER},
    {ov::test::utils::InputLayerType::PARAMETER, ov::test::utils::InputLayerType::CONSTANT, ov::test::utils::InputLayerType::CONSTANT},
    {ov::test::utils::InputLayerType::CONSTANT, ov::test::utils::InputLayerType::PARAMETER, ov::test::utils::InputLayerType::CONSTANT},
    {ov::test::utils::InputLayerType::CONSTANT, ov::test::utils::InputLayerType::CONSTANT, ov::test::utils::InputLayerType::PARAMETER},
    {ov::test::utils::InputLayerType::CONSTANT, ov::test::utils::InputLayerType::PARAMETER, ov::test::utils::InputLayerType::PARAMETER},
    {ov::test::utils::InputLayerType::PARAMETER, ov::test::utils::InputLayerType::CONSTANT, ov::test::utils::InputLayerType::PARAMETER},
    {ov::test::utils::InputLayerType::PARAMETER, ov::test::utils::InputLayerType::PARAMETER, ov::test::utils::InputLayerType::CONSTANT},
};

const std::vector<InputShape> inputShapesDynamic2D = {
        {{-1, -1},
         {{32, 20}, {16, 16}, {24, 16}}},

        {{-1, 16},
         {{16, 16}, {20, 16}, {32, 16}}},
};

const std::vector<StridedSliceParams> paramsPlain2D = {
        StridedSliceParams{ { 0, 10 }, { 16, 16 }, { 1, 1 }, { 0, 0 }, { 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 2, 5 }, { 16, 16 }, { 1, 2 }, { 0, 1 }, { 1, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0 }, { 16, 16 }, { 2, 1 }, { 0, 0 }, { 1, 0 },  { },  { },  { } },
};

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Plain_Static_2D, StridedSliceLayerGPUTest,
                         ::testing::Combine(
                             ::testing::ValuesIn(ov::test::static_shapes_to_test_representation({{32, 20}})),
                             ::testing::ValuesIn(paramsPlain2D),
                             ::testing::ValuesIn(model_types),
                             ::testing::Values(rest_input_types[0])),
                         StridedSliceLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Plain_Dynamic_2D, StridedSliceLayerGPUTest,
                         ::testing::Combine(
                             ::testing::ValuesIn(inputShapesDynamic2D),
                             ::testing::ValuesIn(paramsPlain2D),
                             ::testing::ValuesIn(model_types),
                             ::testing::ValuesIn(rest_input_types)),
                         StridedSliceLayerGPUTest::getTestCaseName);

const std::vector<StridedSliceParams> testCasesCommon4D = {
        StridedSliceParams{ { 0, 2, 5, 4 }, { 1, 4, 28, 27 }, { 1, 1, 1, 1 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 10, 20 }, { 1, 5, 28, 26 }, { 1, 1, 1, 2 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 0, 20 }, { 1, 2, 30, 30 }, { 1, 1, 2, 1 }, { 0, 0, 0, 1 }, { 0, 1, 0, 1 },  { },  { },  { } },
        StridedSliceParams{ { 0, 1, 2, 10 }, { 1, 5, 32, 18 }, { 1, 1, 1, 2 }, { 0, 0, 1, 0 }, { 0, 0, 0, 1 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 2, 10 }, { 1, 8, 32, 18 }, { 1, 2, 1, 2 },  { 0, 0, 1, 0 }, { 0, 0, 0, 1 },  { },  { },  { } },
        StridedSliceParams{ { 2, 10 }, { 32, 18 }, { 1, 2 },  { 1, 0 }, { 0, 1 },  { },  { },  { 1, 0 } },
};

const std::vector<InputShape> inputShapesDynamic4D = {
        {{-1, -1, -1, -1},
         {{ 1, 5, 32, 32 }, { 2, 5, 32, 32 }, { 1, 5, 64, 64 }}},

        {{1, 64, -1, -1},
         {{ 1, 64, 16, 32 }, { 1, 64, 32, 64 }, { 1, 64, 64, 64 }}},

        {{1, -1, 16, 32},
        {{ 1, 16, 16, 32 }, { 1, 32, 16, 32 }, { 1, 64, 16, 32 }}},
};

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Common_Dynamic_4D, StridedSliceLayerGPUTest,
                         ::testing::Combine(
                             ::testing::ValuesIn(inputShapesDynamic4D),
                             ::testing::ValuesIn(testCasesCommon4D),
                             ::testing::ValuesIn(model_types),
                             ::testing::ValuesIn(rest_input_types)),
                         StridedSliceLayerGPUTest::getTestCaseName);


const std::vector<StridedSliceParams> testCasesCommon5D = {
        StridedSliceParams{ { 0, 2, 5, 4 }, { 1, 4, 28, 27 }, { 1, 1, 1, 1 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 10, 20 }, { 1, 5, 28, 26 }, { 1, 1, 1, 2 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 0, 20 }, { 1, 2, 30, 30 }, { 1, 1, 2, 1 }, { 0, 0, 0, 1 }, { 0, 1, 0, 1 },  { },  { },  { } },
        StridedSliceParams{ { 0, 1, 2, 10 }, { 1, 5, 32, 18 }, { 1, 1, 1, 2 }, { 0, 0, 1, 0 }, { 0, 0, 0, 1 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 2, 10 }, { 1, 8, 32, 18 }, { 1, 2, 1, 2 },  { 0, 0, 1, 0 }, { 0, 0, 0, 1 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 2 }, { 1, 8, 32 }, { 1, 2, 1 },  { 1, 0, 1 }, { 0, 0, 0 },  { },  { },  { 0, 1, 0} },
};

const std::vector<InputShape> inputShapesDynamic5D = {
        {{-1, -1, -1, -1, -1},
         {{ 1, 5, 32, 32, 32 }, { 2, 5, 32, 32, 32 }, { 1, 5, 64, 64, 64 }}},

        {{1, 64, -1, -1, -1},
         {{ 1, 64, 1, 16, 32 }, { 1, 64, 1, 32, 64 }, { 1, 64, 1, 64, 64 }}},

        {{1, -1, 16, 32, -1},
        {{ 1, 16, 16, 32, 1 }, { 1, 32, 16, 32, 1 }, { 1, 64, 16, 32, 1 }}},
};

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Common_Dynamic_5D, StridedSliceLayerGPUTest,
                         ::testing::Combine(
                             ::testing::ValuesIn(inputShapesDynamic5D),
                             ::testing::ValuesIn(testCasesCommon5D),
                             ::testing::ValuesIn(model_types),
                             ::testing::ValuesIn(rest_input_types)),
                         StridedSliceLayerGPUTest::getTestCaseName);


const std::vector<StridedSliceParams> testCasesCommon6D = {
        StridedSliceParams{ { 0, 2, 5, 4 }, { 1, 4, 28, 27 }, { 1, 1, 1, 1 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 10, 20 }, { 1, 5, 28, 26 }, { 1, 1, 1, 2 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 },  { },  { },  { } },
        StridedSliceParams{ { 0, 0, 0 }, { 0, 0, 0 }, { 1, 1, 1 }, { 0, 1, 0 }, { 0, 1, 0 },  { },  { },  { 0, 0, 1 } },
};

const std::vector<InputShape> inputShapesDynamic6D = {
        {{-1, -1, -1, -1, -1, -1},
         {{ 1, 5, 5, 32, 32, 32 }, { 2, 5, 7, 32, 32, 64 }, { 1, 3, 5, 64, 64, 64 }}},

        {{1, -1, 16, 32, -1, -1},
        {{ 1, 16, 16, 32, 1, 32 }, { 1, 32, 16, 32, 32, 64 }, { 1, 64, 16, 32, 32, 64 }}},
};

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Common_Dynamic_6D, StridedSliceLayerGPUTest,
                         ::testing::Combine(
                             ::testing::ValuesIn(inputShapesDynamic6D),
                             ::testing::ValuesIn(testCasesCommon6D),
                             ::testing::ValuesIn(model_types),
                             ::testing::ValuesIn(rest_input_types)),
                         StridedSliceLayerGPUTest::getTestCaseName);
} // namespace

const std::vector<InputShape> inputShapesDynamic3D = {
        {{-1, -1, -1},
         {{ 1, 1024, 2 }}},
};
const std::vector<StridedSliceParams> paramsPlain3D = {
        StridedSliceParams{ { 0, 0, 0 }, { 0, 0, 0 }, { 1, 1, 1 }, { 0, 1, 0 }, { 0, 1, 0 },  { 1, 0, 1 },  { },  { } },
};

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_new_axis_3D, StridedSliceLayerReshapeGPUTest,
                         ::testing::Combine(
                             ::testing::ValuesIn(inputShapesDynamic3D),
                             ::testing::ValuesIn(paramsPlain3D),
                             ::testing::ValuesIn(model_types),
                             ::testing::Values(rest_input_types[0])),
                         StridedSliceLayerGPUTest::getTestCaseName);