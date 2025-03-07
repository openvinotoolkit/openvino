// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/gather_weights_decompression.hpp"

#include "ov_ops/gather_compressed.hpp"
#include "shared_test_classes/subgraph/weights_decompression_builders.hpp"

namespace ov {
namespace test {

void GatherWeightsDecompressionBase::generate_inputs(const std::vector<ov::Shape>& target_input_static_shapes) {
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

void GatherWeightsDecompressionBase::check_results(const ov::element::Type& weights_precision, const size_t& num_ops_expect) {
    size_t num_exec_ops = 0;

    for (const auto& n : compiledModel.get_runtime_model()->get_ordered_ops()) {
        if (n->get_friendly_name() == "Compressed_weights") {
            ASSERT_EQ(n->get_output_element_type(0), weights_precision);
        }
        if (n->get_input_size() > 0) {
            num_exec_ops += 1;
        }
    }

    EXPECT_LE(num_exec_ops, num_ops_expect);
}

std::string GatherWeightsDecompression::get_test_case_name(
    testing::TestParamInfo<GatherWeightsDecompressionParams> obj) {
    std::string target_device;
    GatherDecompressionShapeParams shape_params;
    ov::element::Type data_precision;
    ov::element::Type output_precision;
    bool decompression_sub;
    bool reshape_on_decompression;
    bool per_tensor_zp;
    bool per_tensor_scale;

    std::tie(target_device,
             shape_params,
             data_precision,
             output_precision,
             decompression_sub,
             reshape_on_decompression,
             per_tensor_zp,
             per_tensor_scale) = obj.param;

    std::ostringstream result;
    result << "target_device=" << target_device << "_";
    result << shape_params << "_";
    result << "data_precision=" << data_precision << "_";
    result << "output_precision=" << output_precision << "_";
    result << "decompression_subtract=" << decompression_sub << "_";
    result << "reshape_on_decompression=" << reshape_on_decompression << "_";
    result << "per_tensor_zp=" << per_tensor_zp;
    result << "per_tensor_scale=" << per_tensor_scale;

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
                                                                     const bool per_tensor_zp,
                                                                     const bool per_tensor_scale) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::i32, indices_shape)};
    auto axis_const = ov::op::v0::Constant::create(ov::element::i32, {1}, {axis});
    const auto data_subgraph = initGatherDecompressionSubgraph(data_shape,
                                                               group_size,
                                                               data_precision,
                                                               output_precision,
                                                               add_subtract,
                                                               reshape_on_decompression,
                                                               per_tensor_zp,
                                                               per_tensor_scale);

    auto gather = std::make_shared<ov::op::v8::Gather>(data_subgraph, params[0], axis_const, batch_dims);
    gather->set_friendly_name("gather_node");
    return std::make_shared<ov::Model>(ov::NodeVector{gather}, params, "GatherDataDecompression");
}

void GatherWeightsDecompression::check_results() {
    const auto& test_param = GetParam();
    const ov::element::Type& weights_precision = std::get<2>(test_param);
    GatherWeightsDecompressionBase::check_results(weights_precision, 3u);
}

void GatherWeightsDecompression::SetUp() {
    GatherDecompressionShapeParams shape_params;
    ov::element::Type data_precision;
    ov::element::Type output_precision;
    bool decompression_sub;
    bool reshape_on_decompression;
    bool per_tensor_zp;
    bool per_tensor_scale;

    std::tie(targetDevice,
             shape_params,
             data_precision,
             output_precision,
             decompression_sub,
             reshape_on_decompression,
             per_tensor_zp,
             per_tensor_scale) = GetParam();

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
                             per_tensor_zp,
                             per_tensor_scale);

    if (output_precision == ov::element::f16) {
        abs_threshold = 1.0f;
    } else {
        abs_threshold = 1e-4f;
    }
}

// fp16/bf16 constant + convert(16bit to f32) + gather case
std::string GatherWeightsDecompressionWithoutScale::get_test_case_name(
    testing::TestParamInfo<GatherWeightsDecompressionWithoutScaleParams> obj) {
    std::string target_device;
    GatherDecompressionShapeParams shape_params;
    ov::element::Type data_precision;
    ov::element::Type output_precision;

    std::tie(target_device, shape_params, data_precision, output_precision) = obj.param;

    std::ostringstream result;
    result << "target_device=" << target_device << "_";
    result << shape_params << "_";
    result << "data_precision=" << data_precision << "_";
    result << "output_precision=" << output_precision;

    return result.str();
}

std::shared_ptr<ov::Model> GatherWeightsDecompressionWithoutScale::init_subgraph(
    const ov::Shape& data_shape,
    const ov::PartialShape& indices_shape,
    const int axis,
    const int64_t batch_dims,
    const ov::element::Type data_precision,
    const ov::element::Type output_precision) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::i32, indices_shape)};
    auto axis_const = ov::op::v0::Constant::create(ov::element::i32, {1}, {axis});
    ov::test::utils::InputGenerateData generator(-10, 20);
    auto weights_tensor = ov::test::utils::create_and_fill_tensor(data_precision, data_shape, generator);
    auto weights_const = std::make_shared<ov::op::v0::Constant>(weights_tensor);
    weights_const->set_friendly_name("Compressed_weights");
    auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
    auto gather = std::make_shared<ov::op::v8::Gather>(convert, params[0], axis_const, batch_dims);
    gather->set_friendly_name("gather_node");
    auto convert_to_output_precision = std::make_shared<ov::op::v0::Convert>(gather, output_precision);
    return std::make_shared<ov::Model>(ov::NodeVector{convert_to_output_precision}, params, "GatherDataDecompression");
}

void GatherWeightsDecompressionWithoutScale::SetUp() {
    GatherDecompressionShapeParams shape_params;
    ov::element::Type data_precision;
    ov::element::Type output_precision;

    std::tie(targetDevice, shape_params, data_precision, output_precision) = GetParam();

    init_input_shapes({shape_params.indices_shape, {{}, {{shape_params.data_shape}}}});

    inType = ov::element::i32;
    outType = output_precision;

    function = init_subgraph(shape_params.data_shape,
                             inputDynamicShapes[0],
                             shape_params.axis,
                             shape_params.batch_dims,
                             data_precision,
                             output_precision);
}

void GatherWeightsDecompressionWithoutScale::check_results() {
    const auto& test_param = GetParam();
    const ov::element::Type& weights_precision = std::get<2>(test_param);
    GatherWeightsDecompressionBase::check_results(weights_precision, 4u);
}

}  // namespace test
}  // namespace ov
