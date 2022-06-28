// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/adaptive_max_pool.hpp"

#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/primitives/adaptive_pooling.hpp"

namespace ov {
namespace runtime {
namespace intel_gpu {

static void CreateAdaptiveAvgPoolOp(Program& p, const std::shared_ptr<ngraph::op::v8::AdaptiveAvgPool>& op) {
    p.ValidateInputs(op, {2});

    const auto input_primitives = p.GetInputPrimitiveIDs(op);
    const auto layer_name = layer_type_name_ID(op);
    const auto op_friendly_name = op->get_friendly_name();

    const cldnn::adaptive_pooling poolPrim{layer_name,
                                           input_primitives[0],
                                           tensor_from_dims(op->get_output_shape(0)),
                                           op_friendly_name};
    p.AddPrimitive(poolPrim);
    p.AddPrimitiveToProfiler(poolPrim, op);
}

static void CreateAdaptiveMaxPoolOp(Program& p, const std::shared_ptr<ngraph::op::v8::AdaptiveMaxPool>& op) {
    p.ValidateInputs(op, {2});
    if (op->get_output_size() != 2) {
        IE_THROW() << "AdaptiveMaxPool requires 2 outputs";
    }

    auto input_primitives = p.GetInputPrimitiveIDs(op);
    const auto layer_type_name = layer_type_name_ID(op);
    const auto layer_name = layer_type_name + ".0";
    const auto op_friendly_name = op->get_friendly_name();

    const auto indices_precision = op->get_output_element_type(1);
    const auto indices_shape = op->get_output_shape(1);
    const cldnn::layout indices_layout{DataTypeFromPrecision(indices_precision),
                                       DefaultFormatForDims(indices_shape.size()),
                                       tensor_from_dims(indices_shape)};
    const auto indices_memory = p.GetEngine().allocate_memory(indices_layout);

    const cldnn::primitive_id indices_id_w = layer_type_name + "_md_write";
    const cldnn::mutable_data indices_mutable_prim_w{indices_id_w, indices_memory, op_friendly_name};
    p.primitiveIDs[indices_id_w] = indices_id_w;
    p.AddPrimitive(indices_mutable_prim_w);

    input_primitives.push_back(indices_id_w);

    const cldnn::adaptive_pooling poolPrim{layer_name,
                                           input_primitives[0],
                                           tensor_from_dims(op->get_output_shape(0)),
                                           input_primitives.back(),
                                           DataTypeFromPrecision(op->get_index_element_type()),
                                           op_friendly_name};
    p.AddPrimitive(poolPrim);

    const cldnn::primitive_id indices_id_r = layer_type_name + ".1";
    const cldnn::mutable_data indices_mutable_prim_r{indices_id_r, {layer_name}, indices_memory, op_friendly_name};
    p.primitiveIDs[indices_id_r] = indices_id_r;
    p.AddPrimitive(indices_mutable_prim_r);

    p.AddPrimitiveToProfiler(poolPrim, op);
}

REGISTER_FACTORY_IMPL(v8, AdaptiveAvgPool);
REGISTER_FACTORY_IMPL(v8, AdaptiveMaxPool);

}  // namespace intel_gpu
}  // namespace runtime
}  // namespace ov
