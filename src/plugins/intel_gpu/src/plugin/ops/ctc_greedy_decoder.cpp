// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/ctc_greedy_decoder.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "openvino/op/ctc_greedy_decoder_seq_len.hpp"

#include "intel_gpu/primitives/ctc_greedy_decoder.hpp"

#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_gpu {

static void CreateCommonCTCGreedyDecoderOp(ProgramBuilder& p, const std::shared_ptr<ov::Node>& op, bool ctc_merge_repeated) {
    validate_inputs_count(op, {2, 3});
    auto inputs = p.GetInputInfo(op);

    uint32_t blank_index = UINT32_MAX;
    if (inputs.size() == 3) {
        auto blank_index_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(op->get_input_node_shared_ptr(2));
        if (!blank_index_node) {
            OPENVINO_THROW("Unsupported blank_index node type in ", op->get_friendly_name(), " (", op->get_type_name(), ")");
        }
        float val;
        if (ov::shape_size(blank_index_node->get_output_shape(0)) != 1 || !ov::op::util::get_single_value(blank_index_node, val)) {
            OPENVINO_THROW("Unsupported parameter size in ", op->get_friendly_name(), " (", op->get_type_name(), ")");
        }
        blank_index = static_cast<uint32_t>(val);
        inputs.pop_back();
    }

    auto primitive = cldnn::ctc_greedy_decoder(
                layer_type_name_ID(op),
                inputs,
                blank_index,
                ctc_merge_repeated,
                cldnn::element_type_to_data_type(op->get_output_element_type(0)),
                op->get_output_size());
    primitive.output_data_types = get_output_data_types(op, {{ov::element::i64, ov::element::i32}});
    primitive.output_data_types[0] = cldnn::optional_data_type{ov::element::i32}; // // GPU primitive supports only i32 as output data type
    p.add_primitive(*op, primitive);
}

static void CreateCTCGreedyDecoderOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::CTCGreedyDecoder>& op) {
    CreateCommonCTCGreedyDecoderOp(p, op, op->get_ctc_merge_repeated());
}

static void CreateCTCGreedyDecoderSeqLenOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v6::CTCGreedyDecoderSeqLen>& op) {
    CreateCommonCTCGreedyDecoderOp(p, op, op->get_merge_repeated());
}

REGISTER_FACTORY_IMPL(v0, CTCGreedyDecoder);
REGISTER_FACTORY_IMPL(v6, CTCGreedyDecoderSeqLen);

}  // namespace intel_gpu
}  // namespace ov
