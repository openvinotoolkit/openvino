// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/gather_weights_decompression.hpp"
#include "ov_ops/gather_compressed.hpp"

namespace ov {
namespace test {

std::string GatherWeightsDecompression::get_test_case_name(
    testing::TestParamInfo<GatherWeightsDecompressionParams> obj) {
    std::string target_device;
    GWDShapeParams shape_params;
    ov::element::Type data_precision;
    ov::element::Type output_precision;
    bool decompression_sub;
    bool reshape_on_decompression;
    bool per_tensor_zp;

    std::tie(target_device,
             shape_params,
             data_precision,
             output_precision,
             decompression_sub,
             reshape_on_decompression,
             per_tensor_zp) = obj.param;

    std::ostringstream result;
    result << "target_device=" << target_device << "_";
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

std::shared_ptr<ov::Model> GatherWeightsDecompression::init_subgraph(const ov::Shape& data_shape,
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
    gather->set_friendly_name("gather_node");
    return std::make_shared<ov::Model>(ov::NodeVector{gather}, params, "GatherDataDecompression");
}
std::shared_ptr<ov::Node> GatherWeightsDecompression::init_compressed_weights_subgraph(
    const ov::Shape& data_shape,
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
    ov::test::utils::InputGenerateData generate_data;
    if (data_precision.is_signed())
        generate_data.start_from = -5;
    auto weights_tensor = ov::test::utils::create_and_fill_tensor(data_precision, original_data_shape, generate_data);
    auto weights = std::make_shared<ov::op::v0::Constant>(weights_tensor);
    weights->set_friendly_name("Compressed_weights");
    auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights, output_precision);

    std::shared_ptr<ov::Node> mul_parent = weights_convert;

    // Decompression constants shape:
    // Ordinary decompression: [I, 1]
    // Group decompression: [I, N, 1]
    ov::Shape scaleshift_target_shape{data_shape[0]};
    scaleshift_target_shape.insert(scaleshift_target_shape.end(), group_decompression ? data_shape[1] / group_size : 1);
    if (group_decompression || scaleshift_target_shape.size() < original_data_shape.size()) {
        auto data_idx = scaleshift_target_shape.size() - 1;
        scaleshift_target_shape.insert(scaleshift_target_shape.begin() + data_idx + 1, 1);
    }

    auto scaleshift_const_shape = scaleshift_target_shape;
    if (reshape_on_decompression_constant)
        scaleshift_const_shape.erase(std::remove(scaleshift_const_shape.begin(), scaleshift_const_shape.end(), 1),
                                     scaleshift_const_shape.end());
    if (add_subtract) {
        auto shift_tensor_shape = per_tensor_zp ? ov::Shape{1} : scaleshift_const_shape;
        auto shift_tensor = ov::test::utils::create_and_fill_tensor(data_precision, shift_tensor_shape);
        if (per_tensor_zp && data_precision.bitwidth() == 4) {
            static_cast<uint8_t*>(shift_tensor.data())[0] = 0x88;
        }
        auto shift_const = std::make_shared<ov::op::v0::Constant>(shift_tensor);
        std::shared_ptr<ov::Node> shift_convert = std::make_shared<ov::op::v0::Convert>(shift_const, output_precision);
        if (reshape_on_decompression_constant && !per_tensor_zp) {
            auto shift_reshape_const = ov::op::v0::Constant::create(ov::element::i32,
                                                                    {scaleshift_target_shape.size()},
                                                                    scaleshift_target_shape);
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
        auto scale_reshape_const =
            ov::op::v0::Constant::create(ov::element::i32, {scaleshift_target_shape.size()}, scaleshift_target_shape);
        auto scale_reshape = std::make_shared<ov::op::v1::Reshape>(scale_const, scale_reshape_const, false);
        scale_const = scale_reshape;
    }
    std::shared_ptr<ov::Node> last_node = std::make_shared<ov::op::v1::Multiply>(mul_parent, scale_const);

    if (group_decompression) {
        auto reshape_target_shape = std::vector<int>{static_cast<int>(data_shape[0]), -1};
        auto target_shape_node =
            ov::op::v0::Constant::create(ov::element::i32, {reshape_target_shape.size()}, reshape_target_shape);
        last_node = std::make_shared<ov::op::v1::Reshape>(last_node, target_shape_node, false);
    }
    return last_node;
}
void GatherWeightsDecompression::generate_inputs(const std::vector<ov::Shape>& target_input_static_shapes) {
    inputs.clear();
    const auto& model_inputs = function->inputs();
    for (size_t i = 0; i < model_inputs.size(); ++i) {
        const auto& model_input = model_inputs[i];
        ov::test::utils::InputGenerateData in_data;
        in_data.start_from = -1;
        in_data.range = 2;
        in_data.resolution = 10000;
        ov::Tensor tensor = ov::test::utils::create_and_fill_tensor(model_input.get_element_type(),
                                                                    target_input_static_shapes[i],
                                                                    in_data);
        inputs.insert({model_input.get_node_shared_ptr(), tensor});
    }
}
void GatherWeightsDecompression::check_results() {
    const auto& test_param = GetParam();
    ov::element::Type weights_precision = std::get<2>(test_param);
    size_t num_exec_ops = 0;

    for (const auto& n : compiledModel.get_runtime_model()->get_ordered_ops()) {
        if (n->get_friendly_name() == "Compressed_weights") {
            ASSERT_EQ(n->get_output_element_type(0), weights_precision);
        }
        if (n->get_input_size() > 0) {
            num_exec_ops += 1;
        }
    }

    EXPECT_LE(num_exec_ops, 3u);
}

void GatherWeightsDecompression::SetUp() {
    GWDShapeParams shape_params;
    ov::element::Type data_precision;
    ov::element::Type output_precision;
    bool decompression_sub;
    bool reshape_on_decompression;
    bool per_tensor_zp;

    std::tie(targetDevice,
             shape_params,
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

}  // namespace test
}  // namespace ov
