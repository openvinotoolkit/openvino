// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/convert.hpp"

namespace {

using ov::test::InputShape;

struct ShapeParams {
    ShapeParams() = default;
    ShapeParams(InputShape data_shape, ov::Shape weights_shape)
        : data_shape(std::move(data_shape)),
          weights_shape(std::move(weights_shape)) {}

    InputShape data_shape;
    ov::Shape weights_shape;
};

using MatmulConversionSharedParams = std::tuple<ShapeParams,  // input shapes
                                          ov::element::Type,  // input precision
                                          bool,   // transpose on input0(fc_input_a) of matmul_0
                                          bool,   // transpose on input1(fc_input_b) of matmul_0
                                          bool,   // transpose on input0(fc_input_a) of matmul_1
                                          bool>;  // transpose on input1(fc_input_b) of matmul_1

using MatmulConversionParams = std::tuple<std::vector<InputShape>,  // input shapes
                                                     ov::element::Type,  // input precision
                                                     bool,  // transpose on input0(fc_input_a) of matmul_0
                                                     bool,  // transpose on input1(fc_input_b) of matmul_0
                                                     bool,  // transpose on input0(fc_input_a) of matmul_1
                                                     bool>;  // transpose on input1(fc_input_b) of matmul_1

/*
 *
 *                      const                Input0
 *                        |                   /
 *                  Convert_0(FP16)     Convert_1(FP16)
 *                     |  |       \         /
 *                   MatMul_0       MatMul_1
 */
class MatmulConversionsSameParent : public testing::WithParamInterface<MatmulConversionSharedParams>,
                     virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MatmulConversionSharedParams> obj) {
        ShapeParams shape_params;
        ov::element::Type input_precision;
        bool matmul0_tran_0;
        bool matmul0_tran_1;
        bool matmul1_tran_0;
        bool matmul1_tran_1;

        std::tie(shape_params,
                 input_precision,
                 matmul0_tran_0,
                 matmul0_tran_1,
                 matmul1_tran_0,
                 matmul1_tran_1) = obj.param;

        std::ostringstream result;
        result << "IS=(";
        result << ov::test::utils::partialShape2str({shape_params.data_shape.first}) << "_";
        for (const auto& shape : shape_params.data_shape.second) {
            result << ov::test::utils::partialShape2str({shape}) << "_";
        }
        result << "_" << "weights_shape=" << shape_params.weights_shape << "_";
        result << "input_precision=" << input_precision;
        result << "_" << "matmul0_tran_a=" << matmul0_tran_0 << "_";
        result << "_" << "matmul0_tran_b=" << matmul0_tran_1 << "_";
        result << "_" << "matmul1_tran_a=" << matmul1_tran_0 << "_";
        result << "_" << "matmul1_tran_b=" << matmul1_tran_1 << "_";
        return result.str();
    }

protected:
    std::shared_ptr<ov::Model> init_subgraph(const ov::PartialShape& data_shape,
                                             const ov::Shape& weights_shape,
                                             const ov::element::Type input_precision,
                                             bool matmul0_tran_0 = false,
                                             bool matmul0_tran_1 = false,
                                             bool matmul1_tran_0 = false,
                                             bool matmul1_tran_1 = false) {
        auto weights_tensor = ov::test::utils::create_and_fill_tensor(ov::element::i8, weights_shape);
        auto weights = std::make_shared<ov::op::v0::Constant>(weights_tensor);

        auto input0 = std::make_shared<ov::op::v0::Parameter>(input_precision, data_shape);
        auto convert_0 = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f16);
        auto convert_1 = std::make_shared<ov::op::v0::Convert>(input0, ov::element::f16);

        auto matmul_0 = std::make_shared<ov::op::v0::MatMul>(convert_0, convert_0, matmul0_tran_0, matmul0_tran_1);
        auto matmul_1 = std::make_shared<ov::op::v0::MatMul>(convert_0, convert_1, matmul1_tran_0, matmul1_tran_1);

        ov::ParameterVector params{input0};
        return std::make_shared<ov::Model>(ov::NodeVector{matmul_1}, params, "MatmulConversionsSameParent");
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        ShapeParams shape_params;
        ov::element::Type input_precision;
        bool matmul0_tran_0;
        bool matmul0_tran_1;
        bool matmul1_tran_0;
        bool matmul1_tran_1;

        std::tie(shape_params,
                 input_precision,
                 matmul0_tran_0,
                 matmul0_tran_1,
                 matmul1_tran_0,
                 matmul1_tran_1) = GetParam();

        init_input_shapes({shape_params.data_shape, {{}, {{shape_params.weights_shape}}}});

        inType = outType = input_precision;

        function = init_subgraph(inputDynamicShapes[0],
                                 shape_params.weights_shape,
                                 input_precision,
                                 matmul0_tran_0,
                                 matmul0_tran_1,
                                 matmul1_tran_0,
                                 matmul1_tran_1);
    }
};

TEST_P(MatmulConversionsSameParent, Inference) {
    run();
}

const std::vector<ShapeParams> test_input_shapes = {
    {{{1, 128, 64}, {{1, 128, 64}}}, {1, 16, 128}},
};

const std::vector<ShapeParams> test_input_shapes_transposed = {
    {{{1, 64, 128}, {{1, 64, 128}}}, {1, 16, 128}},
};

const std::vector<ov::element::Type> test_input_precisions = {ov::element::f32,
                                                              ov::element::i8};

INSTANTIATE_TEST_SUITE_P(MatmulConversionsSameParent_1,
                         MatmulConversionsSameParent,
                         ::testing::Combine(::testing::ValuesIn(test_input_shapes),
                                            ::testing::ValuesIn(test_input_precisions),
                                            ::testing::Values(true),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(false)),
                         MatmulConversionsSameParent::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(MatmulConversionsSameParent_2,
                         MatmulConversionsSameParent,
                         ::testing::Combine(::testing::ValuesIn(test_input_shapes),
                                            ::testing::ValuesIn(test_input_precisions),
                                            ::testing::Values(false),
                                            ::testing::Values(true),
                                            ::testing::Values(false),
                                            ::testing::Values(false)),
                         MatmulConversionsSameParent::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(MatmulConversionsSameParent_transposed,
                         MatmulConversionsSameParent,
                         ::testing::Combine(::testing::ValuesIn(test_input_shapes_transposed),
                                            ::testing::ValuesIn(test_input_precisions),
                                            ::testing::Values(true),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(true)),
                         MatmulConversionsSameParent::getTestCaseName);

/*
 *
 *    Input2               Input0                Input1
 *       |                    |                    /
 *    Convert_2(FP16)     Convert_0(FP16)      Convert_1(FP16)
 *            \            /        \            /
 *           (in0)      (in1)      (in0)      (in1)
 *              \        /            \        /
 *               MatMul_0              MatMul_1
 */
class MatmulConversions : public testing::WithParamInterface<MatmulConversionParams>,
                     virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MatmulConversionParams> obj) {
        std::vector<InputShape> input_shapes;
        ov::element::Type input_precision;
        bool matmul0_tran_0;
        bool matmul0_tran_1;
        bool matmul1_tran_0;
        bool matmul1_tran_1;

        std::tie(input_shapes,
                 input_precision,
                 matmul0_tran_0,
                 matmul0_tran_1,
                 matmul1_tran_0,
                 matmul1_tran_1) = obj.param;

        std::ostringstream result;
        result << "IS=(";
        for (const auto& shape : input_shapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << ")_TS=";
        for (const auto& shape : input_shapes) {
            result << "(";
            if (!shape.second.empty()) {
                auto itr = shape.second.begin();
                do {
                    result << ov::test::utils::vec2str(*itr);
                } while (++itr != shape.second.end() && result << "_");
            }
            result << ")_";
        }
        result << "input_precision=" << input_precision;
        result << "_" << "matmul0_tran_a=" << matmul0_tran_0 << "_";
        result << "_" << "matmul0_tran_b=" << matmul0_tran_1 << "_";
        result << "_" << "matmul1_tran_a=" << matmul1_tran_0 << "_";
        result << "_" << "matmul1_tran_b=" << matmul1_tran_1 << "_";
        return result.str();
    }

protected:
    std::shared_ptr<ov::Model> init_subgraph(std::vector<ov::PartialShape>& input_shapes,
                                             const ov::element::Type input_precision,
                                             bool matmul0_tran_0,
                                             bool matmul0_tran_1,
                                             bool matmul1_tran_0,
                                             bool matmul1_tran_1) {
        auto input0 = std::make_shared<ov::op::v0::Parameter>(input_precision, input_shapes[0]);
        auto input1 = std::make_shared<ov::op::v0::Parameter>(input_precision, input_shapes[1]);
        auto input2 = std::make_shared<ov::op::v0::Parameter>(input_precision, input_shapes[2]);
        auto convert_0 = std::make_shared<ov::op::v0::Convert>(input0, ov::element::f16);
        auto convert_1 = std::make_shared<ov::op::v0::Convert>(input1, ov::element::f16);
        auto convert_2 = std::make_shared<ov::op::v0::Convert>(input2, ov::element::f16);

        auto matmul_1 = std::make_shared<ov::op::v0::MatMul>(convert_0, convert_1, matmul1_tran_0, matmul1_tran_1);
        auto matmul_0 = std::make_shared<ov::op::v0::MatMul>(convert_2, convert_0, matmul0_tran_0, matmul0_tran_1);

        return std::make_shared<ov::Model>(ov::NodeVector{matmul_1}, ov::ParameterVector{input0, input1}, "MatmulConversions");
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        std::vector<InputShape> input_shapes;
        ov::element::Type input_precision;
        bool matmul0_tran_0;
        bool matmul0_tran_1;
        bool matmul1_tran_0;
        bool matmul1_tran_1;

        std::tie(input_shapes,
                 input_precision,
                 matmul0_tran_0,
                 matmul0_tran_1,
                 matmul1_tran_0,
                 matmul1_tran_1) = GetParam();

        init_input_shapes(input_shapes);

        inType = outType = input_precision;
        function = init_subgraph(inputDynamicShapes,
                                 input_precision,
                                 matmul0_tran_0,
                                 matmul0_tran_1,
                                 matmul1_tran_0,
                                 matmul1_tran_1);
    }
};

TEST_P(MatmulConversions, Inference) {
    run();
}

const std::vector<std::vector<InputShape>> test_input_shapes_basic_1 = {
    {{{1, 16, 128}, {{1, 16, 128}}}, {{1, 128, 64}, {{1, 128, 64}}}, {{1, 32, 16}, {{1, 32, 16}}}},
};

const std::vector<std::vector<InputShape>> test_input_shapes_basic_2 = {
    {{{1, 16, 128}, {{1, 16, 128}}}, {{1, 128, 64}, {{1, 128, 64}}}, {{1, 16, 128}, {{1, 16, 128}}}},
};

INSTANTIATE_TEST_SUITE_P(MatmulConversions_basic_1,
                         MatmulConversions,
                         ::testing::Combine(::testing::ValuesIn(test_input_shapes_basic_1),
                                            ::testing::ValuesIn(test_input_precisions),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(false)),
                         MatmulConversions::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(MatmulConversions_basic_2,
                         MatmulConversions,
                         ::testing::Combine(::testing::ValuesIn(test_input_shapes_basic_2),
                                            ::testing::ValuesIn(test_input_precisions),
                                            ::testing::Values(true),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(false)),
                         MatmulConversions::getTestCaseName);


/*
 *
 *    Input2               Input0               Input1
 *       |                   |                    /
 *    Convert_2(FP16)    Convert_0(FP16)   Convert_1(FP16)
 *           \            /         \           /
 *           (in0)     (in1)       (in1)     (in0)
 *             \        /             \       /
 *              MatMul_0               MatMul_1
 */
class MatmulConversionsOtherTypeSibling : public MatmulConversions,
            virtual public ov::test::SubgraphBaseTest {
public:
protected:
    std::shared_ptr<ov::Model> init_subgraph(std::vector<ov::PartialShape>& input_shapes,
                                             const ov::element::Type input_precision,
                                             bool matmul0_tran_0,
                                             bool matmul0_tran_1,
                                             bool matmul1_tran_0,
                                             bool matmul1_tran_1) {
        auto input0 = std::make_shared<ov::op::v0::Parameter>(input_precision, input_shapes[0]);
        auto input1 = std::make_shared<ov::op::v0::Parameter>(input_precision, input_shapes[1]);
        auto input2 = std::make_shared<ov::op::v0::Parameter>(input_precision, input_shapes[2]);
        auto convert_0 = std::make_shared<ov::op::v0::Convert>(input0, ov::element::f16);
        auto convert_1 = std::make_shared<ov::op::v0::Convert>(input1, ov::element::f16);
        auto convert_2 = std::make_shared<ov::op::v0::Convert>(input2, ov::element::f16);

        auto matmul_1 = std::make_shared<ov::op::v0::MatMul>(convert_1, convert_0, matmul1_tran_0, matmul1_tran_1);
        auto matmul_0 = std::make_shared<ov::op::v0::MatMul>(convert_2, convert_0, matmul0_tran_0, matmul0_tran_1);

        return std::make_shared<ov::Model>(ov::NodeVector{matmul_1}, ov::ParameterVector{input0, input1}, "MatmulConversionsOtherTypeSibling");
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        std::vector<InputShape> input_shapes;
        ov::element::Type input_precision;
        bool matmul0_tran_0;
        bool matmul0_tran_1;
        bool matmul1_tran_0;
        bool matmul1_tran_1;

        std::tie(input_shapes,
                 input_precision,
                 matmul0_tran_0,
                 matmul0_tran_1,
                 matmul1_tran_0,
                 matmul1_tran_1) = GetParam();

        init_input_shapes(input_shapes);

        inType = outType = input_precision;
        function = init_subgraph(inputDynamicShapes,
                                 input_precision,
                                 matmul0_tran_0,
                                 matmul0_tran_1,
                                 matmul1_tran_0,
                                 matmul1_tran_1);
    }
};

TEST_P(MatmulConversionsOtherTypeSibling, Inference) {
    run();
}

const std::vector<std::vector<InputShape>> test_input_shapes_basic_3 = {
    {{{1, 16, 128}, {{1, 16, 128}}}, {{1, 16, 64}, {{1, 16, 64}}}, {{1, 32, 16}, {{1, 32, 16}}}},
};

const std::vector<std::vector<InputShape>> test_input_shapes_basic_4 = {
    {{{1, 16, 128}, {{1, 16, 128}}}, {{1, 16, 64}, {{1, 16, 64}}}, {{1, 16, 128}, {{1, 16, 128}}}},
};

INSTANTIATE_TEST_SUITE_P(MatmulConversionsOtherTypeSibling_basic_1,
                         MatmulConversionsOtherTypeSibling,
                         ::testing::Combine(::testing::ValuesIn(test_input_shapes_basic_3),
                                            ::testing::ValuesIn(test_input_precisions),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(true),
                                            ::testing::Values(false)),
                         MatmulConversionsOtherTypeSibling::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(MatmulConversionsOtherTypeSibling_basic_2,
                         MatmulConversionsOtherTypeSibling,
                         ::testing::Combine(::testing::ValuesIn(test_input_shapes_basic_4),
                                            ::testing::ValuesIn(test_input_precisions),
                                            ::testing::Values(true),
                                            ::testing::Values(false),
                                            ::testing::Values(true),
                                            ::testing::Values(false)),
                         MatmulConversionsOtherTypeSibling::getTestCaseName);

} // namespace
