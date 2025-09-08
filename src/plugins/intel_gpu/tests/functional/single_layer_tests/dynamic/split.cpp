// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_enums.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/op/split.hpp"

namespace {
using ov::test::InputShape;

typedef std::tuple<
        size_t,                    // Num splits
        int64_t,                   // Axis
        ov::element::Type,         // Model type
        InputShape,                // Input shapes
        std::vector<size_t>        // Used outputs indices
> splitDynamicGPUTestParams;

class SplitLayerGPUDynamicTest : public testing::WithParamInterface<splitDynamicGPUTestParams>,
                          virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<splitDynamicGPUTestParams> obj) {
        std::ostringstream result;

        const auto& [num_splits, axis, model_type, input_shape, out_indices] = obj.param;

        result << "IS=";
        result << ov::test::utils::partialShape2str({input_shape.first}) << "_";
        result << "TS=";
        for (const auto& shape : input_shape.second) {
            result << ov::test::utils::vec2str(shape) << "_";
        }
        result << "num_splits=" << num_splits << "_";
        result << "axis=" << axis << "_";
        if (!out_indices.empty()) {
            result << "out_indices" << ov::test::utils::vec2str(out_indices) << "_";
        }
        result << "netPRC=" << model_type << "_";
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        const auto& [num_splits, axis, model_type, input_shape, _out_indices] = this->GetParam();
        auto out_indices = _out_indices;
        if (out_indices.empty()) {
            for (size_t i = 0; i < num_splits; ++i) {
                out_indices.push_back(i);
            }
        }
        init_input_shapes({input_shape});
        ov::ParameterVector dyn_params{std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[0])};
        auto split_axis_op = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64, ov::Shape{}, std::vector<int64_t>{axis});
        auto split = std::make_shared<ov::op::v1::Split>(dyn_params[0], split_axis_op, num_splits);

        ov::ResultVector results;
        for (size_t i = 0; i < out_indices.size(); i++) {
            results.push_back(std::make_shared<ov::op::v0::Result>(split->output(out_indices[i])));
        }
        function = std::make_shared<ov::Model>(results, dyn_params, "split");
    }
};

TEST_P(SplitLayerGPUDynamicTest, Inference) {
    run();
}

const std::vector<InputShape> input_shapes4d = {
        {
            {-1, -1, -1, -1}, {{1, 4, 5, 7}, {3, 8, 5, 9}, {5, 16, 1, 8}}
        }
};

const std::vector<InputShape> input_shapes5d = {
        {
            {-1, -1, -1, -1, -1}, {{10, 20, 30, 40, 10}, {5, 18, 3, 10, 10}, {3, 10, 6, 2, 4}}
        }
};

const std::vector<InputShape> input_shapes6d = {
        {
            {-1, -1, -1, -1, -1, -1}, {{10, 32, 3, 4, 12, 24}, {5, 2, 3, 1, 32, 12}, {3, 1, 6, 2, 4, 18}}
        }
};

INSTANTIATE_TEST_SUITE_P(smoke_SplitsCheck4Dr, SplitLayerGPUDynamicTest,
                        ::testing::Combine(
                                ::testing::Values(2),                                       // nSplits
                                ::testing::Values(1),                                       // axes
                                ::testing::Values(ov::element::f16),                         // netPrec
                                ::testing::ValuesIn(input_shapes4d),                         // inShapes
                                ::testing::Values(std::vector<size_t>({}))),                // out_indices
                        SplitLayerGPUDynamicTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SplitsCheck5D, SplitLayerGPUDynamicTest,
                        ::testing::Combine(
                                ::testing::Values(3),                                       // nSplits
                                ::testing::Values(2),                                       // axes
                                ::testing::Values(ov::element::f32),                         // netPrec
                                ::testing::ValuesIn(input_shapes5d),                         // inShapes
                                ::testing::Values(std::vector<size_t>({}))),                // out_indices
                        SplitLayerGPUDynamicTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SplitsCheck6D, SplitLayerGPUDynamicTest,
                        ::testing::Combine(
                                ::testing::Values(4),                                       // nSplits
                                ::testing::Values(4),                                       // axes
                                ::testing::Values(ov::element::i8),                         // netPrec
                                ::testing::ValuesIn(input_shapes6d),                         // inShapes
                                ::testing::Values(std::vector<size_t>({}))),                // out_indices
                        SplitLayerGPUDynamicTest::getTestCaseName);

typedef std::tuple<
        int64_t,                            // Axis
        std::vector<int32_t>,               // SplitLength
        ov::element::Type,                  // Model type
        InputShape,                         // Input shapes
        ov::test::utils::InputLayerType     // input type of split_length
> varSplitDynamicGPUTestParams;

class VariadicSplitLayerGPUDynamicTest : public testing::WithParamInterface<varSplitDynamicGPUTestParams>,
                                         virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<varSplitDynamicGPUTestParams> obj) {
        std::ostringstream result;

        const auto& [axis, split_length, model_type, input_shape, inputType] = obj.param;

        result << "IS=";
        result << ov::test::utils::partialShape2str({input_shape.first}) << "_";
        result << "TS=";
        for (const auto& shape : input_shape.second) {
            result << ov::test::utils::vec2str(shape) << "_";
        }
        result << "SplitLen=" << ov::test::utils::vec2str(split_length) << "_";
        result << "axis=" << axis << "_";
        result << "netPRC=" << model_type << "_";
        result << "restInputType=" << inputType << "_";
        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            if (i == 1) {
                tensor = ov::Tensor(ov::element::i64, targetInputStaticShapes[i]);
                auto *dataPtr = tensor.data<ov::element_type_traits<ov::element::i64>::value_type>();
                for (size_t i = 0; i < split_length_vec.size(); i++) {
                    dataPtr[i] = split_length_vec[i];
                }
            } else {
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
        inferRequestNum++;
    }

protected:
    std::vector<int32_t> split_length_vec;
    size_t inferRequestNum = 0;
    ov::element::Type model_type;

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        const auto& [axis, split_length, _model_type, input_shape, inputType] = this->GetParam();
        model_type = _model_type;

        split_length_vec = split_length;

        std::vector<InputShape> input_shapes;
        input_shapes.push_back(input_shape);
        if (inputType == ov::test::utils::InputLayerType::PARAMETER) {
            input_shapes.push_back(InputShape({static_cast<int64_t>(split_length.size())},
                                  std::vector<ov::Shape>(input_shape.second.size(), {split_length.size()})));
        }
        init_input_shapes(input_shapes);

        ov::ParameterVector dyn_params{std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[0])};

        auto splitAxisOp = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{static_cast<int64_t>(axis)});

        std::shared_ptr<ov::Node> split_lengthOp;
        if (inputType == ov::test::utils::InputLayerType::PARAMETER) {
            auto split_lengthNode = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{split_length.size()});
            dyn_params.push_back(split_lengthNode);
            split_lengthOp = split_lengthNode;
        } else {
            split_lengthOp = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{split_length.size()}, split_length);
        }

        auto varSplit = std::make_shared<ov::op::v1::VariadicSplit>(dyn_params[0], splitAxisOp, split_lengthOp);
        ov::ResultVector results;
        for (size_t i = 0; i < split_length.size(); i++) {
            results.push_back(std::make_shared<ov::op::v0::Result>(varSplit->output(i)));
        }
        function = std::make_shared<ov::Model>(results, dyn_params, "varSplit");
    }
};

TEST_P(VariadicSplitLayerGPUDynamicTest, Inference) {
    run();
}

const std::vector<ov::test::utils::InputLayerType> restInputTypes = {
    ov::test::utils::InputLayerType::CONSTANT,
    ov::test::utils::InputLayerType::PARAMETER
};

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplitsCheck4D, VariadicSplitLayerGPUDynamicTest,
                        ::testing::Combine(
                                ::testing::Values(1),                                       // axes
                                ::testing::Values(std::vector<int32_t>{2, 1, -1}),          // split_length
                                ::testing::Values(ov::element::f16),                        // netPrec
                                ::testing::ValuesIn(input_shapes4d),                         // inShapes
                                ::testing::ValuesIn(restInputTypes)),                       // input type of split_length
                        VariadicSplitLayerGPUDynamicTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplitsCheck5D, VariadicSplitLayerGPUDynamicTest,
                        ::testing::Combine(
                                ::testing::Values(2),                                       // axes
                                ::testing::Values(std::vector<int32_t>{2, -1}),             // split_length
                                ::testing::Values(ov::element::f32),                        // netPrec
                                ::testing::ValuesIn(input_shapes5d),                         // inShapes
                                ::testing::ValuesIn(restInputTypes)),                       // input type of split_length
                        VariadicSplitLayerGPUDynamicTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplitsCheck6D, VariadicSplitLayerGPUDynamicTest,
                        ::testing::Combine(
                                ::testing::Values(5),                                       // nSplits
                                ::testing::Values(std::vector<int32_t>{2, 3, 2, -1}),       // split_length
                                ::testing::Values(ov::element::i8),                         // netPrec
                                ::testing::ValuesIn(input_shapes6d),                         // inShapes
                                ::testing::ValuesIn(restInputTypes)),                       // input type of split_length
                        VariadicSplitLayerGPUDynamicTest::getTestCaseName);


const std::vector<InputShape> input_shapes4d_static = {
        {
            {5, 16, 10, 8}, {{5, 16, 10, 8}, }
        }
};

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplitsCheck4D_static_input_dyn_output, VariadicSplitLayerGPUDynamicTest,
                        ::testing::Combine(
                                ::testing::Values(1),                                       // axes
                                ::testing::Values(std::vector<int32_t>{2, 1, -1}),          // split_length
                                ::testing::Values(ov::element::f16),                        // netPrec
                                ::testing::ValuesIn(input_shapes4d_static),                         // inShapes
                                ::testing::ValuesIn(restInputTypes)),                       // input type of split_length
                        VariadicSplitLayerGPUDynamicTest::getTestCaseName);

} // namespace
