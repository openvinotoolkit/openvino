// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/ctc_greedy_decoder.hpp"
#include "openvino/op/ctc_greedy_decoder_seq_len.hpp"

#include "intel_gpu/primitives/ctc_greedy_decoder.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

static void CreateCommonCTCGreedyDecoderOp(ProgramBuilder& p, const std::shared_ptr<ov::Node>& op, bool ctc_merge_repeated) {
    validate_inputs_count(op, {2, 3});
    auto inputs = p.GetInputInfo(op);

    std::vector<cldnn::input_info> reordered_inputs;
    reordered_inputs.resize(inputs.size());

    for (size_t portIndex = 0; portIndex < inputs.size(); portIndex++) {
        auto inputDataType = cldnn::element_type_to_data_type(op->get_input_element_type(portIndex));
        if (inputDataType == cldnn::data_types::i64) {
            // GPU primitive supports only i32 data type for 'sequence_length' and 'blank_index' inputs
            // so we need additional reorder if it's provided as i64
            auto reorderPrimName = inputs[portIndex].pid + "_" + op->get_friendly_name() + ProgramBuilder::m_preProcessTag;
            auto targetFormat = cldnn::format::get_default_format(op->get_input_shape(portIndex).size());
            auto preprocessPrim = cldnn::reorder(reorderPrimName,
                                                 inputs[portIndex],
                                                 targetFormat,
                                                 cldnn::data_types::i32);
            p.add_primitive(*op, preprocessPrim);
            reordered_inputs[portIndex] = cldnn::input_info(reorderPrimName);
        } else {
            reordered_inputs[portIndex] = inputs[portIndex];
        }
    }

    if (p.use_new_shape_infer()) {
        uint32_t blank_index = UINT32_MAX;
        if (reordered_inputs.size() == 3) {
            auto blank_index_node = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(2));
            if (!blank_index_node) {
                OPENVINO_THROW("Unsupported blank_index node type in ", op->get_friendly_name(), " (", op->get_type_name(), ")");
            }
            float val;
            if (ov::shape_size(blank_index_node->get_output_shape(0)) != 1 || !ov::op::util::get_single_value(blank_index_node, val)) {
                OPENVINO_THROW("Unsupported parameter size in ", op->get_friendly_name(), " (", op->get_type_name(), ")");
            }
            blank_index = static_cast<uint32_t>(val);
            reordered_inputs.pop_back();
        }

        auto primitive = cldnn::ctc_greedy_decoder(
                    layer_type_name_ID(op),
                    reordered_inputs,
                    blank_index,
                    ctc_merge_repeated,
                    cldnn::element_type_to_data_type(op->get_output_element_type(0)),
                    op->get_output_size());
        primitive.output_data_types = get_output_data_types(op);
        p.add_primitive(*op, primitive);
    } else {
        uint32_t blank_index = static_cast<uint32_t>(op->get_input_shape(0).back() - 1);
        if (reordered_inputs.size() == 3) {
            auto blank_index_node = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(2));
            if (!blank_index_node) {
                OPENVINO_THROW("Unsupported blank_index node type in ", op->get_friendly_name(), " (", op->get_type_name(), ")");
            }
            float val;
            if (ov::shape_size(blank_index_node->get_output_shape(0)) != 1 || !ov::op::util::get_single_value(blank_index_node, val)) {
                OPENVINO_THROW("Unsupported parameter size in ", op->get_friendly_name(), " (", op->get_type_name(), ")");
            }
            blank_index = static_cast<uint32_t>(val);
            reordered_inputs.pop_back();
        }

        std::size_t num_output = op->get_output_size();

        std::vector<cldnn::memory::ptr> shared_memory;
        if (num_output == 2) {
            auto mutable_precision = op->get_output_element_type(1);
            if (mutable_precision == ov::element::i64) {
                mutable_precision = ov::element::i32;
            }

            cldnn::layout mutableLayout = cldnn::layout(
                cldnn::element_type_to_data_type(mutable_precision),
                cldnn::format::get_default_format(op->get_output_shape(1).size()),
                tensor_from_dims(op->get_output_shape(1)));

            GPU_DEBUG_LOG << "[" << layer_type_name_ID(op) << ": mutable data]" << std::endl;
            shared_memory.emplace_back(p.get_engine().allocate_memory(mutableLayout));

            cldnn::primitive_id ctc_gd_mutable_id_w = layer_type_name_ID(op) + "_md_write";
            auto ctc_gd_mutable_prim = cldnn::mutable_data(ctc_gd_mutable_id_w,
                                                        shared_memory[0]);
            p.add_primitive(*op, ctc_gd_mutable_prim);
            reordered_inputs.push_back(ctc_gd_mutable_id_w);
        }

        auto CTCGreedyDecoderLayerName = num_output == 2 ? layer_type_name_ID(op) + ".out0" : layer_type_name_ID(op);
        auto primitive = cldnn::ctc_greedy_decoder(
                    CTCGreedyDecoderLayerName,
                    reordered_inputs,
                    blank_index,
                    ctc_merge_repeated,
                    tensor_from_dims(op->get_output_shape(0)));

        // GPU primitive supports only i32 as output data type
        primitive.output_data_types = {cldnn::element_type_to_data_type(ov::element::i32)};

        if (num_output == 2) {
            primitive.second_output = reordered_inputs.back().pid;
        }

        p.add_primitive(*op, primitive);

        if (num_output == 2) {
            cldnn::primitive_id ctc_gd_mutable_id_r = layer_type_name_ID(op) + ".out1";
            auto ctc_gd_mutable_prim_r = cldnn::mutable_data(ctc_gd_mutable_id_r,
                                                            { cldnn::input_info(CTCGreedyDecoderLayerName) },
                                                            shared_memory[0]);
            p.add_primitive(*op, ctc_gd_mutable_prim_r);
        }
    }
}

static void CreateCTCGreedyDecoderOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::CTCGreedyDecoder>& op) {
    CreateCommonCTCGreedyDecoderOp(p, op, op->get_ctc_merge_repeated());
}

static void CreateCTCGreedyDecoderSeqLenOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v6::CTCGreedyDecoderSeqLen>& op) {
    CreateCommonCTCGreedyDecoderOp(p, op, op->get_merge_repeated());
}

REGISTER_FACTORY_IMPL(v0, CTCGreedyDecoder);
REGISTER_FACTORY_IMPL(v6, CTCGreedyDecoderSeqLen);

}  // namespace ov::intel_gpu
