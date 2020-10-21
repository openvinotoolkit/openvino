// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/non_max_suppression.hpp"

#include "api/reorder.hpp"
#include "api/non_max_suppression.hpp"

namespace CLDNNPlugin {

static bool GetCenterPointBox(ngraph::op::v5::NonMaxSuppression::BoxEncodingType encoding) {
    switch (encoding) {
        case ::ngraph::op::v5::NonMaxSuppression::BoxEncodingType::CENTER: return true;
        case ::ngraph::op::v5::NonMaxSuppression::BoxEncodingType::CORNER: return false;
        default: THROW_IE_EXCEPTION << "NonMaxSuppression layer has unsupported box encoding";
    }
    return false;
}

//TODO: Align with v5 spec
void CreateNonMaxSuppressionOp(Program& p, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v5::NonMaxSuppression>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    p.ValidateInputs(op, {2, 3, 4, 5});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op) + ".0";

    std::vector<cldnn::primitive_id> reorderedInputs;
    reorderedInputs.resize(inputPrimitives.size());

    for (size_t portIndex = 0; portIndex < inputPrimitives.size(); portIndex++) {
        auto inputDataType = DataTypeFromPrecision(op->get_input_element_type(portIndex));
        if ((portIndex == 2) && (inputDataType == cldnn::data_types::i64)) {
            // clDNN primitive supports only i32 data type for 'max_output_boxes_per_class' input
            // so we need additional reorder if it's provided as i64
            auto reorderPrimName = inputPrimitives[portIndex] + "_" + op->get_friendly_name() +Program::m_preProcessTag;
            auto targetFormat = DefaultFormatForDims(op->get_input_shape(portIndex).size());
            auto preprocessPrim = cldnn::reorder(reorderPrimName,
                                                 inputPrimitives[portIndex],
                                                 targetFormat,
                                                 cldnn::data_types::i32);
            p.AddPrimitive(preprocessPrim);
            p.AddInnerPrimitiveToProfiler(reorderPrimName, layer_type_name_ID(op), op);
            reorderedInputs[portIndex] = (reorderPrimName);
        } else {
            reorderedInputs[portIndex] = inputPrimitives[portIndex];
        }
    }

    auto centerPointBox = GetCenterPointBox(op->get_box_encoding());
    auto outputIndices = op->get_output_shape(0)[0];

    auto prim = cldnn::non_max_suppression(layerName,
                                           reorderedInputs[0],
                                           reorderedInputs[1],
                                           static_cast<int>(outputIndices),
                                           centerPointBox);

    switch (reorderedInputs.size()) {
    case 5: prim.score_threshold = reorderedInputs[4];
    case 4: prim.iou_threshold = reorderedInputs[3];
    case 3: prim.num_select_per_class = reorderedInputs[2];
    case 2:
    case 1:
        break;
    default: THROW_IE_EXCEPTION << "Incorrect number of input primitives for layer: " << op->get_friendly_name();
    }

    // clDNN primitive supports only i32 as output data type
    auto out_type = op->get_output_element_type(0);
    if (out_type == ngraph::element::i64) {
        out_type = ngraph::element::i32;
    }

    prim.output_data_type = DataTypeFromPrecision(out_type);

    p.AddPrimitive(prim);
    p.AddPrimitiveToProfiler(layerName, op);
}

REGISTER_FACTORY_IMPL(v5, NonMaxSuppression);

}  // namespace CLDNNPlugin
