// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/embedding_segments_sum.hpp"
#include "ngraph/op/embeddingbag_offsets_sum.hpp"
#include "ngraph/op/embeddingbag_packedsum.hpp"

#include "api/embedding_bag.hpp"
#include "api/reorder.hpp"

#include "transformations/utils/utils.hpp"

namespace CLDNNPlugin {

void CreateEmbeddingBagOffsetsSumOp(Program& p, const std::shared_ptr<ngraph::op::v3::EmbeddingBagOffsetsSum>& op) {
    p.ValidateInputs(op, {3, 4, 5});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    int32_t defaultIndex = -1;
    if (inputPrimitives.size() > 3) {
        auto index_node = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(3));
        if (!index_node) {
            THROW_IE_EXCEPTION << "Unsupported parameter nodes type in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";
        }

        float val;
        if (ngraph::shape_size(index_node->get_output_shape(0)) != 1 || !ngraph::op::util::get_single_value(index_node, val))
             THROW_IE_EXCEPTION << "Unsupported parameter size in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";

        defaultIndex = static_cast<int32_t>(val);
        inputPrimitives.erase(inputPrimitives.begin() + 3); // Remove "default_index"
    }

    std::vector<cldnn::primitive_id> reorderedInputs;
    reorderedInputs.resize(inputPrimitives.size());

    for (size_t portIndex = 0; portIndex < inputPrimitives.size(); portIndex++) {
        auto inputDataType = DataTypeFromPrecision(op->get_input_element_type(portIndex));
        if (((portIndex == 1) || (portIndex == 2)) && (inputDataType == cldnn::data_types::i64)) {
            // clDNN primitive supports only i32 data type for indices inputs,
            // so we need additional reorders if they are provided as i64
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

    auto embeddingBagPrim = cldnn::embedding_bag(layerName,
                                                 reorderedInputs,
                                                 cldnn::embedding_bag::offsets_sum,
                                                 CldnnTensorFromIEDims(op->get_output_shape(0)),
                                                 defaultIndex);

    p.AddPrimitive(embeddingBagPrim);
    p.AddPrimitiveToProfiler(op);
}

void CreateEmbeddingBagPackedSumOp(Program& p, const std::shared_ptr<ngraph::op::v3::EmbeddingBagPackedSum>& op) {
    p.ValidateInputs(op, {2, 3});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    std::vector<cldnn::primitive_id> reorderedInputs;
    reorderedInputs.resize(inputPrimitives.size());

    for (size_t portIndex = 0; portIndex < inputPrimitives.size(); portIndex++) {
        auto inputDataType = DataTypeFromPrecision(op->get_input_element_type(portIndex));
        if ((portIndex == 1) && (inputDataType == cldnn::data_types::i64)) {
            // clDNN primitive supports only i32 data type for indices input,
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

    auto embeddingBagPrim = cldnn::embedding_bag(layerName,
                                                 reorderedInputs,
                                                 cldnn::embedding_bag::packed_sum,
                                                 CldnnTensorFromIEDims(op->get_output_shape(0)));

    p.AddPrimitive(embeddingBagPrim);
    p.AddPrimitiveToProfiler(op);
}

void CreateEmbeddingSegmentsSumOp(Program& p, const std::shared_ptr<ngraph::op::v3::EmbeddingSegmentsSum>& op) {
    p.ValidateInputs(op, {4, 5, 6});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    inputPrimitives.erase(inputPrimitives.begin() + 3); // Remove "num_segments"

    int32_t defaultIndex = -1;
    // port of default_index is 4 by default, but we removed "num_segments" above, so now it's equal to 3
    if (inputPrimitives.size() > 3) {
        auto index_node = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(4));
        if (!index_node) {
            THROW_IE_EXCEPTION << "Unsupported parameter nodes type in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";
        }

        float val;
        if (ngraph::shape_size(index_node->get_output_shape(0)) != 1 || !ngraph::op::util::get_single_value(index_node, val))
             THROW_IE_EXCEPTION << "Unsupported parameter size in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";

        defaultIndex = static_cast<int32_t>(val);
        inputPrimitives.erase(inputPrimitives.begin() + 3); // Remove "default_index"
    }

    std::vector<cldnn::primitive_id> reorderedInputs;
    reorderedInputs.resize(inputPrimitives.size());

    for (size_t portIndex = 0; portIndex < inputPrimitives.size(); portIndex++) {
        auto inputDataType = DataTypeFromPrecision(op->get_input_element_type(portIndex));
        if (((portIndex == 1) || (portIndex == 2)) && (inputDataType == cldnn::data_types::i64)) {
            // clDNN primitive supports only i32 data type for indices inputs,
            // so we need additional reorders if they are provided as i64
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

    auto embeddingBagPrim = cldnn::embedding_bag(layerName,
                                                 reorderedInputs,
                                                 cldnn::embedding_bag::segments_sum,
                                                 CldnnTensorFromIEDims(op->get_output_shape(0)),
                                                 defaultIndex);

    p.AddPrimitive(embeddingBagPrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v3, EmbeddingBagOffsetsSum);
REGISTER_FACTORY_IMPL(v3, EmbeddingBagPackedSum);
REGISTER_FACTORY_IMPL(v3, EmbeddingSegmentsSum);

}  // namespace CLDNNPlugin
