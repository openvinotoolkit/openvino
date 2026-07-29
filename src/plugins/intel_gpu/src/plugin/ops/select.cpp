// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/select.hpp"

#include "intel_gpu/primitives/select.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/reshape.hpp"

namespace ov::intel_gpu {

namespace {

ov::Shape get_pdpd_aligned_shape(const ov::Shape& input_shape, size_t output_rank, int64_t axis) {
    ov::Shape padded_shape = input_shape;

    while (!padded_shape.empty() && padded_shape.back() == 1) {
        padded_shape.pop_back();
    }

    for (int64_t i = 0; i < axis && padded_shape.size() < output_rank; ++i) {
        padded_shape.insert(padded_shape.begin(), 1ul);
    }

    while (padded_shape.size() < output_rank) {
        padded_shape.push_back(1ul);
    }

    return padded_shape;
}

}  // namespace

static void CreateSelectOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::Select>& op) {
    validate_inputs_count(op, {3});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto output_pshape = op->get_output_partial_shape(0);
    auto output_rank = output_pshape.size();

    auto broadcast_type = op->get_auto_broadcast();

    if (broadcast_type.m_type != ov::op::AutoBroadcastType::NONE &&
        broadcast_type.m_type != ov::op::AutoBroadcastType::NUMPY &&
        broadcast_type.m_type != ov::op::AutoBroadcastType::PDPD) {
        OPENVINO_THROW("[GPU] Unsupported broadcast type (", broadcast_type.m_type, ") in layer " + op->get_friendly_name());
    }

    if (broadcast_type.m_type == ov::op::AutoBroadcastType::NUMPY ||
        broadcast_type.m_type == ov::op::AutoBroadcastType::PDPD) {
        // Preprocess inputs
        int64_t pdpd_axis = broadcast_type.m_axis;
        if (broadcast_type.m_type == ov::op::AutoBroadcastType::PDPD && pdpd_axis == -1) {
            auto input2_pshape = op->get_input_partial_shape(2);
            pdpd_axis = static_cast<int64_t>(output_rank) - static_cast<int64_t>(input2_pshape.size());
        }

        for (size_t i = 0; i < inputs.size(); ++i) {
            auto input_pshape = op->get_input_partial_shape(i);

            if (input_pshape.is_static() && !p.use_new_shape_infer()) {
                auto input_shape = input_pshape.to_shape();
                auto input_rank = input_shape.size();
                ov::Shape target_shape = input_shape;

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

                if (broadcast_type.m_type == ov::op::AutoBroadcastType::NUMPY) {
                    // Extend input dimensions to the same size as output dimensions by prepending ones
                    target_shape.insert(target_shape.begin(), output_rank - input_rank, 1ul);
                } else if (broadcast_type.m_type == ov::op::AutoBroadcastType::PDPD && i != 1) {
                    if (input_rank > output_rank || static_cast<size_t>(pdpd_axis) + input_rank > output_rank) {
                        OPENVINO_THROW("[GPU] Invalid PDPD broadcast axis (", pdpd_axis, ") for input ", i,
                                       " in layer " + op->get_friendly_name());
                    }
                    target_shape = get_pdpd_aligned_shape(input_shape, output_rank, pdpd_axis);
                }

                const bool pdpd_adjusts_shape =
                    (broadcast_type.m_type == ov::op::AutoBroadcastType::PDPD) && (target_shape != input_shape);
                const bool need_reshape = (input_rank != output_rank) || (input_rank < 4) || pdpd_adjusts_shape;

                // Reshape input if they differ or select specific shape matches default one
                if (need_reshape) {
                    auto reshapeName = layerName + "_cldnn_in" + std::to_string(i) + "_reshape";
                    auto targetShape = tensor_from_dims(target_shape);

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

}  // namespace ov::intel_gpu
