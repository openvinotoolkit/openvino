// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/select.hpp"

#include "intel_gpu/primitives/select.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/reshape.hpp"

namespace ov {
namespace intel_gpu {

static void CreateSelectOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::Select>& op) {
    validate_inputs_count(op, {3});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto output_pshape = op->get_output_partial_shape(0);
    auto output_rank = output_pshape.size();

    auto broadcast_type = op->get_auto_broadcast();

    if (broadcast_type.m_type != ov::op::AutoBroadcastType::NONE &&
        broadcast_type.m_type != ov::op::AutoBroadcastType::NUMPY) {
        OPENVINO_THROW("[GPU] Unsupported broadcast type (", broadcast_type.m_type, ") in layer " + op->get_friendly_name());
    }

    if (broadcast_type.m_type == ov::op::AutoBroadcastType::NUMPY) {
        // Preprocess inputs
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto input_pshape = op->get_input_partial_shape(i);

            if (input_pshape.is_static() && !p.use_new_shape_infer()) {
                auto input_shape = input_pshape.to_shape();
                auto input_rank = input_shape.size();

                // Add reorder if changing number of dimensions requires changing format
                auto targetFormat = cldnn::format::get_default_format(output_rank);

                if (targetFormat.value != cldnn::format::get_default_format(input_rank).value) {
                    auto reorderName = layerName + "_cldnn_in" + std::to_string(i) + "_reorder";
                    auto targetDatatype = cldnn::element_type_to_data_type(op->get_input_element_type(i));
                    auto reorderPrim = cldnn::reorder(reorderName,
                                                      inputs[i],
                                                      targetFormat,
                                                      targetDatatype);

                    p.add_primitive(*op, reorderPrim);

                    inputs[i] = cldnn::input_info(reorderName);
                }

                // Reshape input if they differ or select specific shape matches default one
                if (input_rank != output_rank || input_rank < 4) {
                    auto reshapeName = layerName + "_cldnn_in" + std::to_string(i) + "_reshape";

                    // Extend input dimensions to the same size as output dimensions by prepending ones
                    input_shape.insert(input_shape.begin(), output_rank - input_rank, 1ul);

                    auto targetShape = tensor_from_dims(input_shape);

                    auto reshapePrim = cldnn::reshape(reshapeName, inputs[i], targetShape);

                    p.add_primitive(*op, reshapePrim);

                    inputs[i] = cldnn::input_info(reshapeName);
                }
            }
        }
    }

    auto selectPrim = cldnn::select(layerName,
                                    inputs[0],
                                    inputs[1],
                                    inputs[2],
                                    broadcast_type);

    p.add_primitive(*op, selectPrim);
}

REGISTER_FACTORY_IMPL(v1, Select);

}  // namespace intel_gpu
}  // namespace ov
