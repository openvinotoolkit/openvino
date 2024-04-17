// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"

#include "openvino/op/max_pool.hpp"
#include "openvino/op/avg_pool.hpp"

#include "intel_gpu/primitives/pooling.hpp"

namespace ov {
namespace intel_gpu {

static void CreateAvgPoolOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::AvgPool>& op) {
    validate_inputs_count(op, {1});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto pooling_prim = std::make_shared<cldnn::pooling>(layerName,
                                                         inputs[0],
                                                         op->get_exclude_pad() ? cldnn::pooling_mode::average_no_padding
                                                                               : cldnn::pooling_mode::average,
                                                         op->get_kernel(),
                                                         op->get_strides(),
                                                         op->get_pads_begin(),
                                                         op->get_pads_end(),
                                                         op->get_auto_pad(),
                                                         op->get_rounding_type());
    p.add_primitive(*op, pooling_prim);
}

static void CreateMaxPoolOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::MaxPool>& op) {
    validate_inputs_count(op, {1});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto pooling_prim = std::make_shared<cldnn::pooling>(layerName,
                                                         inputs[0],
                                                         cldnn::pooling_mode::max,
                                                         op->get_kernel(),
                                                         op->get_strides(),
                                                         op->get_pads_begin(),
                                                         op->get_pads_end(),
                                                         op->get_auto_pad(),
                                                         op->get_rounding_type());
    p.add_primitive(*op, pooling_prim);
}

static void CreateMaxPoolOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v8::MaxPool>& op) {
    validate_inputs_count(op, {1});
    if (op->get_output_size() != 2) {
        OPENVINO_THROW("[GPU] v8:MaxPool requires 2 outputs");
    }
    auto inputs = p.GetInputInfo(op);
    auto poolPrim = cldnn::pooling(layer_type_name_ID(op),
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
                                   cldnn::element_type_to_data_type(op->get_output_element_type(0)));

    poolPrim.num_outputs = op->get_output_size();
    poolPrim.output_data_types = get_output_data_types(op);

    p.add_primitive(*op, poolPrim);
}


REGISTER_FACTORY_IMPL(v1, MaxPool);
REGISTER_FACTORY_IMPL(v8, MaxPool);
REGISTER_FACTORY_IMPL(v1, AvgPool);

}  // namespace intel_gpu
}  // namespace ov
