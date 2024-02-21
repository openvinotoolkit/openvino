// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "transformations/rt_info/decompression.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"

namespace {
using ov::test::InputShape;

/*
 *                        Subtract_const(U8/NF4/U4/I4)
 *                             /
 *    Weights(U8/NF4/U4/I4)  Convert(F32)
 *       |                 /
 *    Convert(F32)   Reshape(optional)
 *            \        /       Multiply_const(F32)
 *            Subtract(optional)     /
 *                  \       Reshape(optional)
 *                   \       /
 *    Indices(I32)    Multiply
 *            \     /
 *             Gather
 */

struct ShapeParams {
    ShapeParams() = default;
    ShapeParams(ov::Shape data_shape, InputShape indices_shape,
                int axis, int64_t batch_dims, int decompression_group_size = -1)
        : data_shape(std::move(data_shape)),
          indices_shape(std::move(indices_shape)),
          axis(axis),
          batch_dims(batch_dims),
          decompression_group_size(decompression_group_size) {}

    ov::Shape data_shape;
    InputShape indices_shape;
    int axis;
    int64_t batch_dims;
    // Decompression group size. If the value is equal to -1, ordinary decompression is used
    int decompression_group_size;
};

using GatherWeightsDecompressionParams = std::tuple<ShapeParams,              // input shapes
                                                    ov::element::Type,        // data type
                                                    ov::element::Type,        // output type
                                                    bool,                     // decompression subtract
                                                    bool,                     // reshape on decompression constants
                                                    bool>;                    // per-tensor zero-point

class GatherWeightsDecompression : public testing::WithParamInterface<GatherWeightsDecompressionParams>,
                                   virtual public ov::test::SubgraphBaseTest {
public:
    static std::string get_test_case_name(testing::TestParamInfo<GatherWeightsDecompressionParams> obj) {
        ShapeParams shape_params;
        ov::element::Type data_precision;
        ov::element::Type output_precision;
        bool decompression_sub;
        bool reshape_on_decompression;
        bool per_tensor_zp;

        std::tie(shape_params,
                 data_precision,
                 output_precision,
                 decompression_sub,
                 reshape_on_decompression,
                 per_tensor_zp) = obj.param;

        std::ostringstream result;
        result << "data_shape=" << shape_params.data_shape << "_";
        result << "indices_shape=";
        result << ov::test::utils::partialShape2str({shape_params.indices_shape.first}) << "_";
        for (const auto& actual_shape : shape_params.indices_shape.second) {
            result << ov::test::utils::partialShape2str({actual_shape}) << "_";
        }
        result << "group_size=" << shape_params.decompression_group_size << "_";
        result << "data_precision=" << data_precision << "_";
        result << "output_precision=" << output_precision << "_";
        result << "decompression_subtract=" << decompression_sub << "_";
        result << "reshape_on_decompression=" << reshape_on_decompression << "_";
        result << "per_tensor_zp=" << per_tensor_zp;

        return result.str();
    }

protected:
    std::shared_ptr<ov::Model> init_subgraph(const ov::Shape& data_shape,
                                             const ov::PartialShape& indices_shape,
                                             const int axis,
                                             const int64_t batch_dims,
                                             const int group_size,
                                             const ov::element::Type data_precision,
                                             const ov::element::Type output_precision,
                                             const bool add_subtract,
                                             const bool reshape_on_decompression,
                                             const bool per_tensor_zp) {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::i32, indices_shape)};
        auto axis_const = ov::op::v0::Constant::create(ov::element::i32, {1}, {axis});
        const auto data_subgraph = init_compressed_weights_subgraph(data_shape,
                                                                    group_size,
                                                                    data_precision,
                                                                    output_precision,
                                                                    add_subtract,
                                                                    reshape_on_decompression,
                                                                    per_tensor_zp);

        auto gather = std::make_shared<ov::op::v8::Gather>(data_subgraph, params[0], axis_const, batch_dims);
        return std::make_shared<ov::Model>(ov::NodeVector{gather}, params, "GatherDataDecompression");
    }

    std::shared_ptr<ov::Node> init_compressed_weights_subgraph(const ov::Shape& data_shape,
                                                               const int group_size,
                                                               const ov::element::Type data_precision,
                                                               const ov::element::Type output_precision,
                                                               const bool add_subtract,
                                                               const bool reshape_on_decompression_constant,
                                                               const bool per_tensor_zp) {
        const bool group_decompression = group_size != -1;
        // Weights has shape [I, D], where
        // I - index
        // D - data
        // In case of group decompression, data dimension is split into 2: I -> [N, G], where
        // N - number of groups
        // G - group size
        auto original_data_shape = data_shape;
        if (group_decompression) {
            OPENVINO_ASSERT(data_shape[1] % group_size == 0,
                            "The last data dimension (",
                            data_shape[1],
                            ") must be divisible by decompression group size (",
                            group_size,
                            ").");
            auto data_idx = data_shape.size() - 1;
            original_data_shape[data_idx] = data_shape[1] / group_size;
            original_data_shape.insert(original_data_shape.begin() + data_idx + 1, group_size);
        }
        auto weights_tensor = ov::test::utils::create_and_fill_tensor(data_precision, original_data_shape);
        auto weights = std::make_shared<ov::op::v0::Constant>(weights_tensor);
        weights->set_friendly_name("Compressed_weighs");
        auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights, output_precision);

        std::shared_ptr<ov::Node> mul_parent = weights_convert;

        // Decompression constants shape:
        // Ordinary decompression: [I, 1]
        // Group decompression: [I, N, 1]
        ov::Shape scaleshift_target_shape{data_shape[0]};
        scaleshift_target_shape.insert(scaleshift_target_shape.end(), group_decompression ? data_shape[1] / group_size : 1);
        if (group_decompression) {
            auto data_idx = scaleshift_target_shape.size() - 1;
            scaleshift_target_shape.insert(scaleshift_target_shape.begin() + data_idx + 1, 1);
        }

        auto scaleshift_const_shape = scaleshift_target_shape;
        if (reshape_on_decompression_constant)
            scaleshift_const_shape.erase(std::remove(scaleshift_const_shape.begin(), scaleshift_const_shape.end(), 1), scaleshift_const_shape.end());
        if (add_subtract) {
            auto shift_tensor_shape = per_tensor_zp ? ov::Shape{1} : scaleshift_const_shape;
            auto shift_tensor = ov::test::utils::create_and_fill_tensor(data_precision, shift_tensor_shape);
            if (per_tensor_zp && data_precision.bitwidth() == 4) {
                static_cast<uint8_t*>(shift_tensor.data())[0] = 0x88;
            }
            auto shift_const = std::make_shared<ov::op::v0::Constant>(shift_tensor);
            std::shared_ptr<ov::Node> shift_convert = std::make_shared<ov::op::v0::Convert>(shift_const, output_precision);
            if (reshape_on_decompression_constant && !per_tensor_zp) {
                auto shift_reshape_const = ov::op::v0::Constant::create(ov::element::i32, {scaleshift_target_shape.size()}, scaleshift_target_shape);
                auto shift_reshape = std::make_shared<ov::op::v1::Reshape>(shift_convert, shift_reshape_const, false);
                shift_convert = shift_reshape;
            }
            mul_parent = std::make_shared<ov::op::v1::Subtract>(weights_convert, shift_convert);
        }

        ov::test::utils::InputGenerateData in_data;
        in_data.start_from = -0.5;
        in_data.range = 1;
        in_data.resolution = 30000;
        auto scale_tensor = ov::test::utils::create_and_fill_tensor(output_precision, scaleshift_const_shape, in_data);
        for (size_t i = 0; i < scale_tensor.get_size(); i++) {
            if (output_precision == ov::element::f16)
                scale_tensor.data<ov::float16>()[i] /= ov::float16(16.f);
            else if (output_precision == ov::element::f32)
                scale_tensor.data<float>()[i] /= 16.f;
        }
        std::shared_ptr<ov::Node> scale_const = std::make_shared<ov::op::v0::Constant>(scale_tensor);
        if (reshape_on_decompression_constant) {
            auto scale_reshape_const = ov::op::v0::Constant::create(ov::element::i32, {scaleshift_target_shape.size()}, scaleshift_target_shape);
            auto scale_reshape = std::make_shared<ov::op::v1::Reshape>(scale_const, scale_reshape_const, false);
            scale_const = scale_reshape;
        }
        std::shared_ptr<ov::Node> last_node = std::make_shared<ov::op::v1::Multiply>(mul_parent, scale_const);

        if (group_decompression) {
            auto reshape_target_shape = std::vector<int>{static_cast<int>(data_shape[0]), -1};
            auto target_shape_node = ov::op::v0::Constant::create(ov::element::i32, {reshape_target_shape.size()}, reshape_target_shape);
            last_node = std::make_shared<ov::op::v1::Reshape>(last_node, target_shape_node, false);
        }
        return last_node;
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        ShapeParams shape_params;
        ov::element::Type data_precision;
        ov::element::Type output_precision;
        bool decompression_sub;
        bool reshape_on_decompression;
        bool per_tensor_zp;

        std::tie(shape_params,
                 data_precision,
                 output_precision,
                 decompression_sub,
                 reshape_on_decompression,
                 per_tensor_zp) = GetParam();

        init_input_shapes({shape_params.indices_shape, {{}, {{shape_params.data_shape}}}});

        inType = ov::element::i32;
        outType = output_precision;

        function = init_subgraph(shape_params.data_shape,
                                 inputDynamicShapes[0],
                                 shape_params.axis,
                                 shape_params.batch_dims,
                                 shape_params.decompression_group_size,
                                 data_precision,
                                 output_precision,
                                 decompression_sub,
                                 reshape_on_decompression,
                                 per_tensor_zp);

        if (output_precision == ov::element::f16) {
            abs_threshold = 1.0f;
        } else {
            abs_threshold = 1e-4f;
        }
    }

    void generate_inputs(const std::vector<ov::Shape>& target_input_static_shapes) override {
          inputs.clear();
          const auto& model_inputs = function->inputs();
          for (size_t i = 0; i < model_inputs.size(); ++i) {
                const auto& model_input = model_inputs[i];
                ov::test::utils::InputGenerateData in_data;
                in_data.start_from = -1;
                in_data.range = 2;
                in_data.resolution = 10000;
                ov::Tensor tensor = ov::test::utils::create_and_fill_tensor(model_input.get_element_type(), target_input_static_shapes[i], in_data);
                inputs.insert({model_input.get_node_shared_ptr(), tensor});
          }
    }

    void check_results() {
        const auto& test_param = GetParam();
        ov::element::Type weights_precision = std::get<1>(test_param);
        for (const auto& n : compiledModel.get_runtime_model()->get_ordered_ops()) {
            if (n->get_friendly_name() == "Compressed_weights") {
                ASSERT_EQ(n->get_output_element_type(0), weights_precision);
            }
        }
    }
};

TEST_P(GatherWeightsDecompression, Inference) {
    run();
    check_results();
}

const std::vector<ov::element::Type> output_precisions = {ov::element::f32, ov::element::f16};
const std::vector<ov::element::Type> weights_precisions = {ov::element::u8, ov::element::i8, ov::element::u4, ov::element::i4};
const std::vector<ShapeParams> input_shapes_basic = {
    {{2, 5}, {{-1, -1}, {{2, 3}}}, 1, 1},
    {{15, 32}, {{-1, -1}, {{2, 3}}}, 1, 0, 16},
    {{2, 5}, {{}, {{2, 3}}}, 1, -1},
};
const std::vector<bool> add_decompression_sub = {true, false};
const std::vector<bool> reshape_on_decompression = {true, false};
const std::vector<bool> per_tensor_zp = {true, false};

INSTANTIATE_TEST_SUITE_P(smoke_GatherCompressedWeights_basic,
                         GatherWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_basic),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(output_precisions),
                                            ::testing::ValuesIn(add_decompression_sub),
                                            ::testing::ValuesIn(reshape_on_decompression),
                                            ::testing::ValuesIn(per_tensor_zp)),
                         GatherWeightsDecompression::get_test_case_name);
} // namespace
