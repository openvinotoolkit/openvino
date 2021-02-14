// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/ctc_greedy_decoder.hpp"
#include "ngraph/op/ctc_greedy_decoder_seq_len.hpp"

#include "api/ctc_greedy_decoder.hpp"
#include "api/reorder.hpp"
#include "api/mutable_data.hpp"

#include "transformations/utils/utils.hpp"

namespace CLDNNPlugin {

void CreateCommonCTCGreedyDecoderOp(Program& p, const std::shared_ptr<ngraph::Node>& op, bool ctc_merge_repeated) {
    p.ValidateInputs(op, {2, 3});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);

    std::vector<cldnn::primitive_id> reorderedInputs;
    reorderedInputs.resize(inputPrimitives.size());

    for (size_t portIndex = 0; portIndex < inputPrimitives.size(); portIndex++) {
        auto inputDataType = DataTypeFromPrecision(op->get_input_element_type(portIndex));
        if (inputDataType == cldnn::data_types::i64) {
            // clDNN primitive supports only i32 data type for 'sequence_length' and 'blank_index' inputs
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

    uint32_t blank_index = op->get_input_shape(0).back() - 1;
    if (reorderedInputs.size() == 3) {
        auto blank_index_node = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(2));
        if (!blank_index_node) {
            THROW_IE_EXCEPTION << "Unsupported blank_index node type in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";
        }
        float val;
        if (ngraph::shape_size(blank_index_node->get_output_shape(0)) != 1 || !ngraph::op::util::get_single_value(blank_index_node, val)) {
            THROW_IE_EXCEPTION << "Unsupported parameter size in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";
        }
        blank_index = static_cast<uint32_t>(val);
        reorderedInputs.pop_back();
    }

    std::size_t num_output = op->get_output_size();

    std::vector<cldnn::memory> shared_memory;
    if (num_output == 2) {
        auto mutable_precision = op->get_output_element_type(1);
         if (mutable_precision == ngraph::element::i64) {
            mutable_precision = ngraph::element::i32;
        }

        cldnn::layout mutableLayout = cldnn::layout(
            DataTypeFromPrecision(mutable_precision),
            DefaultFormatForDims(op->get_output_shape(1).size()),
            CldnnTensorFromIEDims(op->get_output_shape(1)));

        shared_memory.emplace_back(cldnn::memory::allocate(p.GetEngine(), mutableLayout));

        cldnn::primitive_id ctc_gd_mutable_id_w = layer_type_name_ID(op) + "_md_write";
        auto ctc_gd_mutable_prim = cldnn::mutable_data(ctc_gd_mutable_id_w, shared_memory[0]);
        p.primitivesToIRLayersMap[ctc_gd_mutable_id_w] = { op->get_friendly_name() };
        p.primitiveIDs[ctc_gd_mutable_id_w] = ctc_gd_mutable_id_w;
        p.AddPrimitive(ctc_gd_mutable_prim);
        reorderedInputs.push_back(ctc_gd_mutable_id_w);
    }

    auto CTCGreedyDecoderLayerName = num_output == 2 ? layer_type_name_ID(op) + ".0" : layer_type_name_ID(op);
    auto primitive = cldnn::ctc_greedy_decoder(
                CTCGreedyDecoderLayerName,
                reorderedInputs,
                blank_index,
                ctc_merge_repeated,
                CldnnTensorFromIEDims(op->get_output_shape(0)));

    // clDNN primitive supports only i32 as output data type
    primitive.output_data_type = DataTypeFromPrecision(ngraph::element::i32);

    if (num_output == 2) {
        primitive.second_output = reorderedInputs.back();
    }

    p.AddPrimitive(primitive);

    if (num_output == 2) {
        cldnn::primitive_id ctc_gd_mutable_id_r = layer_type_name_ID(op) + ".1";
        auto ctc_gd_mutable_prim_r = cldnn::mutable_data(ctc_gd_mutable_id_r, { CTCGreedyDecoderLayerName }, shared_memory[0]);
        p.primitivesToIRLayersMap[ctc_gd_mutable_id_r] = { op->get_friendly_name() };
        p.primitiveIDs[ctc_gd_mutable_id_r] = ctc_gd_mutable_id_r;
        p.AddPrimitive(ctc_gd_mutable_prim_r);
    }

    p.AddPrimitiveToProfiler(CTCGreedyDecoderLayerName, op);
}

void CreateCTCGreedyDecoderOp(Program& p, const std::shared_ptr<ngraph::op::v0::CTCGreedyDecoder>& op) {
    CreateCommonCTCGreedyDecoderOp(p, op, op->get_ctc_merge_repeated());
}

void CreateCTCGreedyDecoderSeqLenOp(Program& p, const std::shared_ptr<ngraph::op::v6::CTCGreedyDecoderSeqLen>& op) {
    CreateCommonCTCGreedyDecoderOp(p, op, op->get_merge_repeated());
}

REGISTER_FACTORY_IMPL(v0, CTCGreedyDecoder);
REGISTER_FACTORY_IMPL(v6, CTCGreedyDecoderSeqLen);

}  // namespace CLDNNPlugin
