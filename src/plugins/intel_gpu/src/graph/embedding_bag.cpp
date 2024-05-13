// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "embedding_bag_inst.h"

#include "openvino/op/embedding_segments_sum.hpp"
#include "openvino/op/embeddingbag_offsets_sum.hpp"
#include "openvino/op/embeddingbag_packedsum.hpp"
#include "primitive_type_base.h"

#include "embeddingbag_offsets_shape_inference.hpp"
#include "embeddingbag_packed_shape_inference.hpp"
#include "embedding_segments_sum_shape_inference.hpp"

#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(embedding_bag)

template<typename ShapeType>
std::vector<layout> embedding_bag_inst::calc_output_layouts(embedding_bag_node const& /*node*/, const kernel_impl_params& impl_param) {
    const auto& input_layout = impl_param.get_input_layout();
    auto desc = impl_param.typed_desc<embedding_bag>();

    auto& memory_deps = impl_param.memory_deps;
    std::vector<ShapeType> input_shapes;
    for (size_t i = 0; i < desc->input_size(); i++) {
        input_shapes.push_back(impl_param.get_input_layout(i).get<ShapeType>());
    }

    std::vector<ShapeType> output_shapes;

    switch (desc->type) {
        case embedding_bag::packed_sum: {
            ov::op::v3::EmbeddingBagPackedSum op;
            output_shapes = ov::op::util::shape_infer(&op, input_shapes);
            break;
        }
        case embedding_bag::offsets_sum: {
            ov::op::v3::EmbeddingBagOffsetsSum op;
            output_shapes = ov::op::util::shape_infer(&op, input_shapes);
            break;
        }
        case embedding_bag::segments_sum: {
            ov::op::v3::EmbeddingSegmentsSum op;

            const size_t num_segments_idx = 3;
            TensorsContainer const_data(&impl_param.get_stream());
            if (memory_deps.count(num_segments_idx) > 0) {
                const_data.emplace(3, memory_deps.at(num_segments_idx));
            }
            output_shapes = ov::op::v3::shape_infer(&op, input_shapes, cldnn::make_tensor_accessor(const_data));
            break;
        }
    }

    return { layout(output_shapes[0], input_layout.data_type, input_layout.format) };
}

template std::vector<layout> embedding_bag_inst::calc_output_layouts<ov::PartialShape>(embedding_bag_node const& node, const kernel_impl_params& impl_param);

layout embedding_bag_inst::calc_output_layout(embedding_bag_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<embedding_bag>();

    auto input_layout = impl_param.get_input_layout();
    auto output_format = input_layout.format;

    auto output_shape = desc->output_shape;

    return layout(input_layout.data_type, output_format, output_shape);
}

std::string embedding_bag_inst::to_string(embedding_bag_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite embedding_bag_info;
    embedding_bag_info.add("input id", input.id());
    switch (desc->type) {
    case embedding_bag::packed_sum:
        embedding_bag_info.add("embedding bag type", "PackedSum");
        break;
    case embedding_bag::offsets_sum:
        embedding_bag_info.add("embedding bag type", "OffsetsSum");
        break;
    case embedding_bag::segments_sum:
        embedding_bag_info.add("embedding bag type", "SegmentsSum");
        break;
    }

    node_info->add("embedding_bag info", embedding_bag_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

embedding_bag_inst::typed_primitive_inst(network& network, embedding_bag_node const& node)
    : parent(network, node) {}
}  // namespace cldnn
