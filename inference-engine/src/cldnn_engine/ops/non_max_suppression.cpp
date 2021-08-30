// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/non_max_suppression.hpp"
#include <ngraph/opsets/opset3.hpp>
#include <ngraph_ops/nms_ie_internal.hpp>

#include "cldnn/primitives/reorder.hpp"
#include "cldnn/primitives/mutable_data.hpp"
#include "cldnn/primitives/non_max_suppression.hpp"

namespace CLDNNPlugin {

static bool GetCenterPointBox(ngraph::op::v5::NonMaxSuppression::BoxEncodingType encoding) {
    switch (encoding) {
        case ::ngraph::op::v5::NonMaxSuppression::BoxEncodingType::CENTER: return true;
        case ::ngraph::op::v5::NonMaxSuppression::BoxEncodingType::CORNER: return false;
        default: IE_THROW() << "NonMaxSuppression layer has unsupported box encoding";
    }
    return false;
}

void CreateNonMaxSuppressionIEInternalOp(Program& p, const std::shared_ptr<ngraph::op::internal::NonMaxSuppressionIEInternal>& op) {
    p.ValidateInputs(op, {2, 3, 4, 5, 6});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);

    std::vector<cldnn::primitive_id> reorderedInputs;
    reorderedInputs.resize(inputPrimitives.size());

    for (size_t portIndex = 0; portIndex < inputPrimitives.size(); portIndex++) {
        auto inputDataType = DataTypeFromPrecision(op->get_input_element_type(portIndex));
        if ((portIndex == 2) && (inputDataType == cldnn::data_types::i64)) {
            // clDNN primitive supports only i32 data type for 'max_output_boxes_per_class' input
            // so we need additional reorder if it's provided as i64
            auto reorderPrimName = inputPrimitives[portIndex] + "_" + op->get_friendly_name() + Program::m_preProcessTag;
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

    // clDNN primitive supports only i32 as output data type
    auto out_type = op->get_output_element_type(0);
    if (out_type == ngraph::element::i64) {
        out_type = ngraph::element::i32;
    }

    auto outputIndices = op->get_output_shape(0)[0];

    auto boxesShape = op->get_input_shape(0);

    std::size_t num_output = op->get_output_size();

    std::vector<cldnn::memory::ptr> shared_memory;
    switch (num_output) {
        case 3: {
            auto mutable_precision_second = op->get_output_element_type(2);
            if (mutable_precision_second == ngraph::element::i64) {
                mutable_precision_second = ngraph::element::i32;
            }
            cldnn::layout mutableLayoutSecond = cldnn::layout(
                DataTypeFromPrecision(mutable_precision_second),
                DefaultFormatForDims(op->get_output_shape(2).size()),
                CldnnTensorFromIEDims(op->get_output_shape(2)));

            shared_memory.emplace_back(p.GetEngine().allocate_memory(mutableLayoutSecond));

            cldnn::primitive_id non_max_supression_mutable_id_w_second = layer_type_name_ID(op) + "_md_write_second";
            auto nms_mutable_prim_second = cldnn::mutable_data(non_max_supression_mutable_id_w_second, shared_memory.back());
            p.primitivesToIRLayersMap[non_max_supression_mutable_id_w_second] = { op->get_friendly_name() };
            p.primitiveIDs[non_max_supression_mutable_id_w_second] = non_max_supression_mutable_id_w_second;
            p.AddPrimitive(nms_mutable_prim_second);
            inputPrimitives.push_back(non_max_supression_mutable_id_w_second);
        }
        case 2: {
            auto mutable_precision_first = op->get_output_element_type(1);

            cldnn::layout mutableLayoutFirst = cldnn::layout(
                DataTypeFromPrecision(mutable_precision_first),
                cldnn::format::bfyx,
                cldnn::tensor(static_cast<int32_t>(outputIndices), 3, 1, 1));

            shared_memory.emplace_back(p.GetEngine().allocate_memory(mutableLayoutFirst));

            cldnn::primitive_id non_max_supression_mutable_id_w_first = layer_type_name_ID(op) + "_md_write_first";
            auto nms_mutable_prim_first = cldnn::mutable_data(non_max_supression_mutable_id_w_first, shared_memory.back());
            p.primitivesToIRLayersMap[non_max_supression_mutable_id_w_first] = { op->get_friendly_name() };
            p.primitiveIDs[non_max_supression_mutable_id_w_first] = non_max_supression_mutable_id_w_first;
            p.AddPrimitive(nms_mutable_prim_first);
            inputPrimitives.push_back(non_max_supression_mutable_id_w_first);
        }
        case 1: break;
        default: IE_THROW() << "Incorrect number of output for layer: " << op->get_friendly_name();
    }

    auto nonMaxSupressionLayerName = num_output > 1 ? layer_type_name_ID(op) + ".0" : layer_type_name_ID(op);

    auto prim = cldnn::non_max_suppression(
            nonMaxSupressionLayerName,
            reorderedInputs[0],
            reorderedInputs[1],
            static_cast<int>(outputIndices),
            op->m_center_point_box,
            op->m_sort_result_descending);

    prim.output_data_type = DataTypeFromPrecision(out_type);

    switch (reorderedInputs.size()) {
        case 6: prim.soft_nms_sigma = reorderedInputs[5];
        case 5: prim.score_threshold = reorderedInputs[4];
        case 4: prim.iou_threshold = reorderedInputs[3];
        case 3: prim.num_select_per_class = reorderedInputs[2];
        case 2: break;
        default: IE_THROW() << "Incorrect number of input primitives for layer: " << op->get_friendly_name();
    }

    switch (num_output) {
        case 3: prim.third_output = inputPrimitives[inputPrimitives.size() - 2];
        case 2: prim.second_output = inputPrimitives[inputPrimitives.size() - 1];
        default: break;
    }

    p.AddPrimitive(prim);

    switch (num_output) {
        case 3: {
            cldnn::primitive_id non_max_supression_id_r_second = layer_type_name_ID(op) + ".2";
            auto nms_mutable_prim_r_second = cldnn::mutable_data(non_max_supression_id_r_second, { nonMaxSupressionLayerName }, shared_memory.front());
            p.primitivesToIRLayersMap[non_max_supression_id_r_second] = { op->get_friendly_name() };
            p.primitiveIDs[non_max_supression_id_r_second] = non_max_supression_id_r_second;
            p.AddPrimitive(nms_mutable_prim_r_second);
        }
        case 2: {
            cldnn::primitive_id non_max_supression_id_r_first = layer_type_name_ID(op) + ".1";
            auto nms_mutable_prim_r_first = cldnn::mutable_data(non_max_supression_id_r_first, { nonMaxSupressionLayerName }, shared_memory.back());
            p.primitivesToIRLayersMap[non_max_supression_id_r_first] = { op->get_friendly_name() };
            p.primitiveIDs[non_max_supression_id_r_first] = non_max_supression_id_r_first;
            p.AddPrimitive(nms_mutable_prim_r_first);
        }
        default: break;
    }

    p.AddPrimitiveToProfiler(nonMaxSupressionLayerName, op);
}

REGISTER_FACTORY_IMPL(internal, NonMaxSuppressionIEInternal);

}  // namespace CLDNNPlugin
