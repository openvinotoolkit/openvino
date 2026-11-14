// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sstream>
#include <string>
#include <vector>

#include "json_object.h"
#include "memory_accessor.hpp"
#include "primitive_type_base.h"
#include "qsa_indexer_inst.h"
#include "qsa_indexer_shape_inference.hpp"
#include "to_string_utils.h"

namespace cldnn {

GPU_DEFINE_PRIMITIVE_TYPE_ID(qsa_indexer)

layout qsa_indexer_inst::calc_output_layout(const qsa_indexer_node& node, const kernel_impl_params& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template <typename ShapeType>
std::vector<layout> qsa_indexer_inst::calc_output_layouts(const qsa_indexer_node& node, const kernel_impl_params& impl_param) {
    const auto& desc = impl_param.typed_desc<qsa_indexer>();
    std::vector<ShapeType> input_shapes;
    input_shapes.reserve(impl_param.input_layouts.size());
    for (const auto& input_layout : impl_param.input_layouts) {
        input_shapes.push_back(input_layout.get<ShapeType>());
    }

    ov::op::internal::QSAIndexer op;
    op.set_config(desc->get_config());
    // key_length is a shape-inference dependency because it sets the block count of the updated cache.
    const auto ta = MemoryAccessor(&impl_param.memory_deps, impl_param.get_stream());
    const auto output_shapes = ov::op::internal::shape_infer(&op, input_shapes, ta);

    using In = qsa_indexer::InputIdx;
    using Out = qsa_indexer::OutputIdx;
    // The updated states keep the element type of the corresponding state input; the selection outputs are i32.
    std::vector<ov::element::Type> data_types(Out::COUNT, ov::element::i32);
    data_types[Out::BLOCK_INDEX_KEYS_CACHE] = impl_param.get_input_layout(In::BLOCK_INDEX_KEYS_CACHE).data_type;
    data_types[Out::PAST_INCOMPLETE_RAW_K] = impl_param.get_input_layout(In::PAST_INCOMPLETE_RAW_K).data_type;
    data_types[Out::PAST_INCOMPLETE_ROPE] = impl_param.get_input_layout(In::PAST_INCOMPLETE_ROPE).data_type;

    std::vector<layout> layouts;
    layouts.reserve(output_shapes.size());
    for (size_t out = 0; out < output_shapes.size(); ++out) {
        layouts.emplace_back(output_shapes[out], data_types[out], cldnn::format::get_default_format(output_shapes[out].size()));
    }
    return layouts;
}

template std::vector<layout> qsa_indexer_inst::calc_output_layouts<ov::PartialShape>(const qsa_indexer_node& node, const kernel_impl_params& impl_param);

std::string qsa_indexer_inst::to_string(const qsa_indexer_node& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();

    std::stringstream primitive_description;
    json_composite info;
    info.add("key_block_size", desc->key_block_size);
    info.add("block_topk", desc->block_topk);
    info.add("num_selection_heads", desc->num_selection_heads);
    info.add("indexer_head_dim", desc->indexer_head_dim);
    info.add("indexer_rotary_dim", desc->indexer_rotary_dim);
    info.add("scale", desc->scale);
    node_info->add("qsa_indexer_info", info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

qsa_indexer_inst::typed_primitive_inst(network& network, const qsa_indexer_node& node) : parent(network, node) {
    std::string reason;
    OPENVINO_ASSERT(ov::op::internal::QSAIndexer::validate_config(node.get_primitive()->get_config(), reason),
                    "[GPU] Invalid qsa_indexer primitive: ",
                    reason);
}

}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::qsa_indexer)
