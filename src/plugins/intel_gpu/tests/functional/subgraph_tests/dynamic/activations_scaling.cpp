// Copyright (C) 2018-2026 Intel Corporation
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
#include "openvino/op/convolution.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/variadic_split.hpp"

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
    static std::string getTestCaseName(const testing::TestParamInfo<ActivationsScalingParams>& obj) {
        const auto& [shape_params, input_precision] = obj.param;

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

        return std::make_shared<ov::Model>(ov::OutputVector{add}, params, "ActivationsScaling");
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        const auto& [shape_params, input_precision] = GetParam();

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
    std::string cacheDirName = ov::test::utils::generateTestFilePrefix() + "_gpu_model_cache";
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

class ActivationsScalingFeatureAxisRMS : public ov::test::SubgraphBaseTest {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        init_input_shapes({{{-1, 3, -1, -1}, {{1, 3, 16, 16}}}});
        inType = outType = ov::element::f32;

        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[0]);
        auto weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{8, 3, 3, 3}, {1.f});
        auto convolution = std::make_shared<ov::op::v1::Convolution>(input,
                                                                     weights,
                                                                     ov::Strides{1, 1},
                                                                     ov::CoordinateDiff{1, 1},
                                                                     ov::CoordinateDiff{1, 1},
                                                                     ov::Strides{1, 1});

        auto power = std::make_shared<ov::op::v1::Power>(
            convolution,
            ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {2.f}));
        auto mean = std::make_shared<ov::op::v1::ReduceMean>(
            power,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}),
            true);
        auto add = std::make_shared<ov::op::v1::Add>(
            mean,
            ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1e-6f}));
        auto sqrt = std::make_shared<ov::op::v0::Sqrt>(add);
        auto reciprocal = std::make_shared<ov::op::v1::Divide>(
            ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.f}),
            sqrt);
        auto normalized = std::make_shared<ov::op::v1::Multiply>(convolution, reciprocal);

        function = std::make_shared<ov::Model>(ov::OutputVector{normalized}, ov::ParameterVector{input});
    }

    void generate_inputs(const std::vector<ov::Shape>& target_input_static_shapes) override {
        inputs.clear();
        const auto& function_inputs = function->inputs();
        ov::Tensor tensor(function_inputs[0].get_element_type(), target_input_static_shapes[0]);
        auto* data = tensor.data<float>();
        for (size_t i = 0; i < tensor.get_size(); ++i) {
            data[i] = 10000.f;
        }
        inputs.insert({function_inputs[0].get_node_shared_ptr(), tensor});
    }
};

TEST_F(ActivationsScalingFeatureAxisRMS, Inference) {
    core->set_property(targetDevice, ov::hint::activations_scale_factor(8.f));
    run();
}

// FakeQuantize layers kept by the INT8 LPT run (Q/K quantization in front of SDPA, as in INT8 hybrid quantized
// diffusion transformers) must keep their intervals when activations scaling runs LPT again. Decomposing them with
// the f16 DataPrecision interval overflows f16 in the standalone quantize primitive and SDPA returns NaN.
//
//                      input
//                        |
//         +--------------+--------------+
//         |              |              |
//      MatMul         MatMul         MatMul
//         |              |              |
//      Reshape        Reshape        Reshape
//         |              |              |
//       RoPE           RoPE             |
//         |              |              |
//   FakeQuantize   FakeQuantize         |
//         |              |              |
//     Transpose      Transpose      Transpose
//         | Q            | K            | V
//         +--------------+--------------+
//                        |
//                      SDPA
//                        |
//                    Transpose
//                        |
//                     Reshape
//                        |
//                     MatMul
class ActivationsScalingFakeQuantizeSDPA : public ov::test::SubgraphBaseTest {
protected:
    static constexpr int64_t num_heads = 2;
    static constexpr int64_t head_size = 64;
    static constexpr int64_t hidden_size = num_heads * head_size;

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        init_input_shapes({{{-1, -1, hidden_size}, {{1, 32, hidden_size}}},
                           {{1, -1, 1, head_size}, {{1, 32, 1, head_size}}},
                           {{1, -1, 1, head_size}, {{1, 32, 1, head_size}}}});
        inType = outType = ov::element::f32;
        abs_threshold = 0.01f;
        // Activations scaling applies to f16 inference only, while core_configuration() selects f32 for f32 parameters.
        configuration.insert(ov::hint::inference_precision(ov::element::f16));

        ov::ParameterVector params;
        for (const auto& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape));
        }
        const auto& input = params[0];
        const auto& cos = params[1];
        const auto& sin = params[2];

        int seed = 1;
        auto linear = [&](const ov::Output<ov::Node>& data) {
            const auto weights =
                ov::test::utils::create_and_fill_tensor_real_distribution(ov::element::f32, ov::Shape{hidden_size, hidden_size}, -0.1f, 0.1f, seed++);
            return std::make_shared<ov::op::v0::MatMul>(data, std::make_shared<ov::op::v0::Constant>(weights), false, true);
        };
        auto split_heads = [](const ov::Output<ov::Node>& data) {
            const auto shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 0, num_heads, head_size});
            return std::make_shared<ov::op::v1::Reshape>(data, shape, true);
        };
        auto slice_last_axis = [](const ov::Output<ov::Node>& data, int64_t start, int64_t stop) {
            return std::make_shared<ov::op::v8::Slice>(data,
                                                       ov::op::v0::Constant::create(ov::element::i64, {1}, {start}),
                                                       ov::op::v0::Constant::create(ov::element::i64, {1}, {stop}),
                                                       ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                                       ov::op::v0::Constant::create(ov::element::i64, {1}, {3}));
        };
        auto rope = [&](const ov::Output<ov::Node>& data) {
            const auto second_half = slice_last_axis(data, head_size / 2, std::numeric_limits<int64_t>::max());
            const auto negated = std::make_shared<ov::op::v1::Multiply>(second_half, ov::op::v0::Constant::create(ov::element::f32, {}, {-1.f}));
            const auto rotated = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{negated, slice_last_axis(data, 0, head_size / 2)}, -1);
            return std::make_shared<ov::op::v1::Add>(std::make_shared<ov::op::v1::Multiply>(data, cos), std::make_shared<ov::op::v1::Multiply>(rotated, sin));
        };
        auto fake_quantize = [](const ov::Output<ov::Node>& data) {
            const auto low = ov::op::v0::Constant::create(ov::element::f32, {}, {-4.f});
            const auto high = ov::op::v0::Constant::create(ov::element::f32, {}, {4.f});
            return std::make_shared<ov::op::v0::FakeQuantize>(data, low, high, low, high, 256);
        };
        const auto order = ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3});

        const auto q = std::make_shared<ov::op::v1::Transpose>(fake_quantize(rope(split_heads(linear(input)))), order);
        const auto k = std::make_shared<ov::op::v1::Transpose>(fake_quantize(rope(split_heads(linear(input)))), order);
        const auto v = std::make_shared<ov::op::v1::Transpose>(split_heads(linear(input)), order);
        const auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, false);

        const auto merged = std::make_shared<ov::op::v1::Reshape>(std::make_shared<ov::op::v1::Transpose>(sdpa, order),
                                                                  ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{0, 0, hidden_size}),
                                                                  true);
        function = std::make_shared<ov::Model>(ov::OutputVector{linear(merged)}, params, "ActivationsScalingFakeQuantizeSDPA");
    }

    void generate_inputs(const std::vector<ov::Shape>& target_input_static_shapes) override {
        inputs.clear();
        const auto& model_inputs = function->inputs();
        for (size_t i = 0; i < model_inputs.size(); ++i) {
            const auto tensor =
                ov::test::utils::create_and_fill_tensor_real_distribution(ov::element::f32, target_input_static_shapes[i], -1.f, 1.f, static_cast<int>(i) + 1);
            inputs.insert({model_inputs[i].get_node_shared_ptr(), tensor});
        }
    }
};

TEST_F(ActivationsScalingFakeQuantizeSDPA, smoke_Inference) {
    core->set_property(targetDevice, ov::hint::activations_scale_factor(8.f));
    run();
}
} // namespace
