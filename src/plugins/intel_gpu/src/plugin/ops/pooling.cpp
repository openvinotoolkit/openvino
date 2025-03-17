// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/max_pool.hpp"
#include "openvino/op/avg_pool.hpp"

#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/primitives/pooling.hpp"

namespace ov::intel_gpu {

static void CreateAvgPoolOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::AvgPool>& op) {
    validate_inputs_count(op, {1});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    std::shared_ptr<cldnn::pooling> pooling_prim = nullptr;
    if (p.use_new_shape_infer()) {
        pooling_prim = std::make_shared<cldnn::pooling>(layerName,
                                                        inputs[0],
                                                        op->get_exclude_pad() ? cldnn::pooling_mode::average_no_padding
                                                                              : cldnn::pooling_mode::average,
                                                        op->get_kernel(),
                                                        op->get_strides(),
                                                        op->get_pads_begin(),
                                                        op->get_pads_end(),
                                                        op->get_auto_pad(),
                                                        op->get_rounding_type());
    } else {
        pooling_prim = std::make_shared<cldnn::pooling>(layerName,
                                                        inputs[0],
                                                        op->get_exclude_pad() ? cldnn::pooling_mode::average_no_padding
                                                                              : cldnn::pooling_mode::average,
                                                        op->get_kernel(),
                                                        op->get_strides(),
                                                        op->get_pads_begin(),
                                                        op->get_pads_end(),
                                                        tensor_from_dims(op->get_output_shape(0)),
                                                        cldnn::element_type_to_data_type(op->get_output_element_type(0)));
    }
    p.add_primitive(*op, pooling_prim);
}

static void CreateMaxPoolOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::MaxPool>& op) {
    validate_inputs_count(op, {1});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    std::shared_ptr<cldnn::pooling> pooling_prim = nullptr;
    if (p.use_new_shape_infer()) {
        pooling_prim = std::make_shared<cldnn::pooling>(layerName,
                                                        inputs[0],
                                                        cldnn::pooling_mode::max,
                                                        op->get_kernel(),
                                                        op->get_strides(),
                                                        op->get_pads_begin(),
                                                        op->get_pads_end(),
                                                        op->get_auto_pad(),
                                                        op->get_rounding_type());
    } else {
        pooling_prim = std::make_shared<cldnn::pooling>(layerName,
                                                        inputs[0],
                                                        cldnn::pooling_mode::max,
                                                        op->get_kernel(),
                                                        op->get_strides(),
                                                        op->get_pads_begin(),
                                                        op->get_pads_end(),
                                                        tensor_from_dims(op->get_output_shape(0)),
                                                        cldnn::element_type_to_data_type(op->get_output_element_type(0)));
    }
    p.add_primitive(*op, pooling_prim);
}

static void CreateMaxPoolOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v8::MaxPool>& op) {
    validate_inputs_count(op, {1});
    if (op->get_output_size() != 2) {
        OPENVINO_THROW("[GPU] v8:MaxPool requires 2 outputs");
    }
    auto inputs = p.GetInputInfo(op);
    const auto layer_type_name = layer_type_name_ID(op);
    const auto layerName = layer_type_name + ".out0";

    const auto mutable_precision = op->get_output_element_type(1);
    const auto output_shape = op->get_output_shape(1);
    cldnn::layout mutableLayout = cldnn::layout(cldnn::element_type_to_data_type(mutable_precision),
                                                cldnn::format::get_default_format(output_shape.size()),
                                                tensor_from_dims(output_shape));
    const auto shared_memory = p.get_engine().allocate_memory(mutableLayout);
    const cldnn::primitive_id maxpool_mutable_id_w = layer_type_name + "_md_write";
    auto indices_mutable_prim = cldnn::mutable_data(maxpool_mutable_id_w,
                                                          shared_memory);
    p.add_primitive(*op, indices_mutable_prim);
    inputs.push_back(cldnn::input_info(maxpool_mutable_id_w));

    auto poolPrim = cldnn::pooling(layerName,
                                   inputs[0],
                                   inputs.back(),
                                   op->get_kernel(),
                                   op->get_strides(),
                                   op->get_dilations(),
                                   op->get_pads_begin(),
                                   op->get_pads_end(),
                                   op->get_auto_pad(),
                                   op->get_rounding_type(),
                                   op->get_axis(),
                                   cldnn::element_type_to_data_type(op->get_index_element_type()),
                                   tensor_from_dims(op->get_output_shape(0)),
                                   cldnn::element_type_to_data_type(op->get_output_element_type(0)));
    p.add_primitive(*op, poolPrim);

    const cldnn::primitive_id maxpool_mutable_id_r = layer_type_name + ".out1";
    auto indices_mutable_id_r = cldnn::mutable_data(maxpool_mutable_id_r,
                                                    { cldnn::input_info(layerName) },
                                                    shared_memory);
    p.add_primitive(*op, indices_mutable_id_r);
}


REGISTER_FACTORY_IMPL(v1, MaxPool);
REGISTER_FACTORY_IMPL(v8, MaxPool);
REGISTER_FACTORY_IMPL(v1, AvgPool);

}  // namespace ov::intel_gpu
