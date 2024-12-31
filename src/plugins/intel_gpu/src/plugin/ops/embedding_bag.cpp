// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/embedding_segments_sum.hpp"
#include "openvino/op/embeddingbag_offsets_sum.hpp"
#include "openvino/op/embeddingbag_packedsum.hpp"

#include "intel_gpu/primitives/embedding_bag.hpp"
#include "intel_gpu/primitives/reorder.hpp"

#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_gpu {

static void CreateEmbeddingBagOffsetsSumOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v3::EmbeddingBagOffsetsSum>& op) {
    validate_inputs_count(op, {3, 4, 5});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    int32_t defaultIndex = -1;
    if (inputs.size() > 3) {
        auto index_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(op->get_input_node_shared_ptr(3));
        OPENVINO_ASSERT(index_node != nullptr, "[GPU] Unsupported parameter nodes type in ", op->get_friendly_name(), " (", op->get_type_name(), ")");

        float val;
        if (ov::shape_size(index_node->get_output_shape(0)) != 1 || !ov::op::util::get_single_value(index_node, val))
             OPENVINO_THROW("Unsupported parameter size in ", op->get_friendly_name(), " (", op->get_type_name(), ")");

        defaultIndex = static_cast<int32_t>(val);
    }

    auto embeddingBagPrim = cldnn::embedding_bag(layerName,
                                                 inputs,
                                                 cldnn::embedding_bag::offsets_sum,
                                                 defaultIndex);

    p.add_primitive(*op, embeddingBagPrim);
}

static void CreateEmbeddingBagPackedSumOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v3::EmbeddingBagPackedSum>& op) {
    validate_inputs_count(op, {2, 3});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto embeddingBagPrim = cldnn::embedding_bag(layerName,
                                                 inputs,
                                                 cldnn::embedding_bag::packed_sum,
                                                 -1);

    p.add_primitive(*op, embeddingBagPrim);
}

static void CreateEmbeddingSegmentsSumOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v3::EmbeddingSegmentsSum>& op) {
    validate_inputs_count(op, {4, 5, 6});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    int32_t defaultIndex = -1;
    if (op->get_input_size() > 4) {
        auto index_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(op->get_input_node_shared_ptr(4));
        OPENVINO_ASSERT(index_node != nullptr, "[GPU] Unsupported parameter nodes type in ", op->get_friendly_name(), " (", op->get_type_name(), ")");

        float val;
        if (ov::shape_size(index_node->get_output_shape(0)) != 1 || !ov::op::util::get_single_value(index_node, val))
            OPENVINO_THROW("Unsupported parameter size in ", op->get_friendly_name(), " (", op->get_type_name(), ")");

        defaultIndex = static_cast<int32_t>(val);
    }

    auto embeddingBagPrim = cldnn::embedding_bag(layerName,
                                                 inputs,
                                                 cldnn::embedding_bag::segments_sum,
                                                 defaultIndex);

    p.add_primitive(*op, embeddingBagPrim);
}

REGISTER_FACTORY_IMPL(v3, EmbeddingBagOffsetsSum);
REGISTER_FACTORY_IMPL(v3, EmbeddingBagPackedSum);
REGISTER_FACTORY_IMPL(v3, EmbeddingSegmentsSum);

}  // namespace intel_gpu
}  // namespace ov
