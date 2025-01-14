// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"

namespace {
using ov::test::InputShape;

struct ShapeParams {
    ShapeParams() = default;
    ShapeParams(std::vector<InputShape> input_shape, std::vector<ov::Shape> weights_shapes)
        : input_shape(std::move(input_shape)),
          weights_shapes(std::move(weights_shapes)) {}

    std::vector<InputShape> input_shape;
    std::vector<ov::Shape> weights_shapes;
};

using ActivationsScalingParams = std::tuple<ShapeParams,             // input shapes
                                            ov::element::Type>;      // input precision

class ActivationsScaling : public testing::WithParamInterface<ActivationsScalingParams>,
                             virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ActivationsScalingParams> obj) {
        ShapeParams shape_params;
        ov::element::Type input_precision;

        std::tie(shape_params, input_precision) = obj.param;

        std::ostringstream result;
        result << "IS=(";
        for (const auto& shape : shape_params.input_shape) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << ")_TS=";
        for (const auto& shape : shape_params.input_shape) {
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
        return result.str();
    }

protected:
    //                input
    //               /     \_
    //         MatMul      MutMul
    //            |           |
    //           Add         Add
    //            |           |
    //         Variadic    Reshape
    //          Split         |
    //         /     \    Transpose
    //       Add      \       |
    //       /         \      |
    //   Reshape    Reshape   |
    //      |          |      |
    // Transpose   Transpose /
    //    |    \   /        /
    //    |     Add        /
    //    |    /   \      /
    //   Concat     Concat
    //         \   /
    //          Add
    //           |
    //        Output
    std::shared_ptr<ov::Model> init_subgraph(const std::vector<ov::PartialShape>& input_shapes,
                                             const std::vector<ov::Shape>& weights_shapes,
                                             const ov::element::Type input_precision) {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(input_precision, input_shapes[0])};

        const auto weights_tensor0 = ov::test::utils::create_and_fill_tensor(
            input_precision, weights_shapes[0], ov::test::utils::InputGenerateData(-1, 2, 1357, 1));
        auto weight0 = std::make_shared<ov::op::v0::Constant>(weights_tensor0);

        const auto weights_tensor1 = ov::test::utils::create_and_fill_tensor(
            input_precision, weights_shapes[1], ov::test::utils::InputGenerateData(-1, 2, 1357, 1));
        auto weight1 = std::make_shared<ov::op::v0::Constant>(weights_tensor1);

        auto matmul0 = std::make_shared<ov::op::v0::MatMul>(params[0], weight0, false, false);
        auto matmul1 = std::make_shared<ov::op::v0::MatMul>(params[0], weight1, false, false);

        std::vector<ov::Shape> bias_shape = {{weights_shapes[0][1]}, {weights_shapes[1][1]}};
        const auto bias_tensor0 = ov::test::utils::create_and_fill_tensor(
            input_precision, bias_shape[0], ov::test::utils::InputGenerateData(-1, 2, 1357, 1));
        const auto bias_tensor1 = ov::test::utils::create_and_fill_tensor(
            input_precision, bias_shape[1], ov::test::utils::InputGenerateData(-1, 2, 1357, 1));

        auto bias0 = std::make_shared<ov::op::v0::Constant>(bias_tensor0);
        auto bias1 = std::make_shared<ov::op::v0::Constant>(bias_tensor1);

        auto add0 = std::make_shared<ov::op::v1::Add>(matmul0, bias0);
        auto add1 = std::make_shared<ov::op::v1::Add>(matmul1, bias1);

        int32_t last_size0 = static_cast<int32_t>(weights_shapes[0][1]);
        auto axis0 = ov::op::v0::Constant::create(ov::element::i32, {1}, {2});
        auto split_lengths0 = ov::op::v0::Constant::create(ov::element::i32, {2}, {last_size0/2, last_size0/2});
        auto variadic_split0 = std::make_shared<ov::op::v1::VariadicSplit>(add0, axis0, split_lengths0);

        ov::Shape bias_shape00 = {weights_shapes[0][1]/2};
        const auto bias_tensor00 = ov::test::utils::create_and_fill_tensor(
            input_precision, bias_shape00, ov::test::utils::InputGenerateData(-1, 2, 1357, 1));
        auto bias00 = std::make_shared<ov::op::v0::Constant>(bias_tensor00);
        auto add00 = std::make_shared<ov::op::v1::Add>(variadic_split0->output(0), bias00);

        const std::vector<int32_t> reshape_target_shape = {0, -1, 16, (last_size0/32)};
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32,
                                                          {reshape_target_shape.size()},
                                                          reshape_target_shape);

        auto reshape0 = std::make_shared<ov::op::v1::Reshape>(add00, reshape_const, true);
        auto reshape1 = std::make_shared<ov::op::v1::Reshape>(variadic_split0->output(1), reshape_const, true);
        auto reshape2 = std::make_shared<ov::op::v1::Reshape>(add1, reshape_const, true);

        const std::vector<int32_t> transpose_order = {0, 2, 1, 3};
        auto transpose_const = ov::op::v0::Constant::create(ov::element::i32,
                                                            {transpose_order.size()},
                                                            transpose_order);

        auto transpose0 = std::make_shared<ov::op::v1::Transpose>(reshape0, transpose_const);
        auto transpose1 = std::make_shared<ov::op::v1::Transpose>(reshape1, transpose_const);
        auto transpose2 = std::make_shared<ov::op::v1::Transpose>(reshape2, transpose_const);

        auto add12 = std::make_shared<ov::op::v1::Add>(transpose1, transpose2);

        auto concat0 = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{transpose0, add12}, 0);

        auto concat1 = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{add12, transpose2}, 0);

        auto add = std::make_shared<ov::op::v1::Add>(concat0, concat1);

        return std::make_shared<ov::Model>(ov::NodeVector{add}, params, "ActivationsScaling");
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        ShapeParams shape_params;
        ov::element::Type input_precision;

        std::tie(shape_params, input_precision) = GetParam();

        init_input_shapes(shape_params.input_shape);

        inType = outType = input_precision;

        if (input_precision == ov::element::f16) {
            abs_threshold = 1.0f;
        } else {
            abs_threshold = 1e-4f;
        }

        function = init_subgraph(inputDynamicShapes, shape_params.weights_shapes, input_precision);
    }

    void validate() override {
        ov::test::SubgraphBaseTest::validate();

        ov::element::Type input_precision = function->input(0).get_element_type();
        auto runtime_model = compiledModel.get_runtime_model();

        for (auto& op : runtime_model->get_ordered_ops()) {
            ASSERT_EQ(op->output(0).get_element_type(), input_precision)
                    << "expected output precision is " << input_precision << " , while actual is " << op->output(0).get_element_type();
        }
    }
};

TEST_P(ActivationsScaling, Inference) {
    core->set_property(targetDevice, ov::hint::activations_scale_factor(4.3));
    run();
}

TEST_P(ActivationsScaling, Inference_cached) {
    std::stringstream ss;
    ss << "gpu_model_cache_" << std::hash<std::string>{}(
          std::string(::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name()) +
          std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    std::string cacheDirName = ss.str();
    {
        ov::test::utils::removeFilesWithExt(cacheDirName, "blob");
        ov::test::utils::removeFilesWithExt(cacheDirName, "cl_cache");
        ov::test::utils::removeDir(cacheDirName);
        core->set_property(ov::cache_dir(cacheDirName));
        compile_model();
    }
    {
        run();
        ov::test::utils::removeFilesWithExt(cacheDirName, "blob");
        ov::test::utils::removeFilesWithExt(cacheDirName, "cl_cache");
        ov::test::utils::removeDir(cacheDirName);
    }
}

const std::vector<ov::element::Type> input_precisions = {ov::element::f16};

const std::vector<ShapeParams> input_shapes = {
    {{{{-1, 32, 128}, {{1, 32, 128}}}}, {{128, 128}, {128, 64}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_ActivationsScaling_basic,
                         ActivationsScaling,
                         ::testing::Combine(::testing::ValuesIn(input_shapes),
                                            ::testing::ValuesIn(input_precisions)),
                         ActivationsScaling::getTestCaseName);
} // namespace
