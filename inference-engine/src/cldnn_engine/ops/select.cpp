// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/select.hpp"

#include "api/select.hpp"
#include "api/reorder.hpp"
#include "api/reshape.hpp"

namespace CLDNNPlugin {

void CreateSelectOp(Program& p, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v1::Select>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    p.ValidateInputs(op, {3});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto outDims = op->get_output_shape(0);
    auto outDimsN = outDims.size();

    auto broadcast_type = op->get_auto_broadcast();

    if (broadcast_type.m_type != ngraph::op::AutoBroadcastType::NONE &&
        broadcast_type.m_type != ngraph::op::AutoBroadcastType::NUMPY) {
        THROW_IE_EXCEPTION << "Unsupported broadcast type (" << broadcast_type.m_type << ") in layer " + op->get_friendly_name();
    }

    auto selectSpecificTensor = [](const InferenceEngine::SizeVector& dims, int def = 1) {
        switch (dims.size()) {
        case 0: return cldnn::tensor(cldnn::batch(def), cldnn::feature(def), cldnn::spatial(def, def));
        case 1: return cldnn::tensor(cldnn::batch(dims[0]), cldnn::feature(def), cldnn::spatial(def, def));
        case 2: return cldnn::tensor(cldnn::batch(dims[0]), cldnn::feature(dims[1]), cldnn::spatial(def, def));
        case 3: return cldnn::tensor(cldnn::batch(dims[0]), cldnn::feature(dims[1]), cldnn::spatial(def, dims[2]));
        case 4: return cldnn::tensor(cldnn::batch(dims[0]), cldnn::feature(dims[1]), cldnn::spatial(dims[3], dims[2]));
        case 5: return cldnn::tensor(cldnn::batch(dims[0]), cldnn::feature(dims[1]), cldnn::spatial(dims[4], dims[3], dims[2]));
        case 6: return cldnn::tensor(cldnn::batch(dims[0]), cldnn::feature(dims[1]), cldnn::spatial(dims[5], dims[4], dims[3], dims[2]));
        default: THROW_IE_EXCEPTION << "Invalid dimensions size(" << dims.size() << ") for Select layer";
        }
    };

    if (broadcast_type.m_type == ngraph::op::AutoBroadcastType::NUMPY) {
        // Preprocess inputs
        for (size_t i = 0; i < inputPrimitives.size(); ++i) {
            auto inputDims = op->get_input_shape(i);
            auto inputDimsN = inputDims.size();

            // Add reorder if changing number of dimensions requires changing format
            auto targetFormat = DefaultFormatForDims(outDimsN);

            if (targetFormat.value != DefaultFormatForDims(inputDimsN).value) {
                auto reorderName = layerName + "_cldnn_in" + std::to_string(i) + "_reorder";
                auto targetDatatype = DataTypeFromPrecision(op->get_input_element_type(i));
                auto reorderPrim = cldnn::reorder(reorderName, inputPrimitives[i], targetFormat, targetDatatype);

                p.AddPrimitive(reorderPrim);
                p.AddInnerPrimitiveToProfiler(reorderName, layerName, op);

                inputPrimitives[i] = reorderName;
            }

            // Reshape input if they differ or select specific shape matches default one
            if (inputDimsN != outDimsN || inputDimsN < 4) {
                auto reshapeName = layerName + "_cldnn_in" + std::to_string(i) + "_reshape";

                // Extend input dimensions to the same size as output dimensions by prepending ones
                inputDims.insert(inputDims.begin(), outDimsN - inputDimsN, 1ul);

                auto targetShape = selectSpecificTensor(inputDims);

                auto reshapePrim = cldnn::reshape(reshapeName, inputPrimitives[i], targetShape);

                p.AddPrimitive(reshapePrim);
                p.AddInnerPrimitiveToProfiler(reshapeName, layerName, op);

                inputPrimitives[i] = reshapeName;
            }
        }
    }

    std::string bc_string = broadcast_type.m_type == ngraph::op::AutoBroadcastType::NUMPY ? "numpy" : "none";

    auto selectPrim = cldnn::select(layerName,
                                    inputPrimitives[0],
                                    inputPrimitives[1],
                                    inputPrimitives[2],
                                    cldnn::padding(),
                                    bc_string);

    p.AddPrimitive(selectPrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v1, Select);

}  // namespace CLDNNPlugin
