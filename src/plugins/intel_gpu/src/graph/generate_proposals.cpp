// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "generate_proposals_inst.h"
#include "openvino/op/generate_proposals.hpp"
#include "generate_proposals_shape_inference.hpp"
#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(generate_proposals)

template<typename ShapeType>
std::vector<layout> generate_proposals_inst::calc_output_layouts(generate_proposals_node const& /*node*/, const kernel_impl_params& impl_param) {
    auto input_layout = impl_param.get_input_layout();
    const auto desc = impl_param.typed_desc<generate_proposals>();
    const auto num_outputs = desc->output_size();

    const auto im_info_shape = input_layout.get_partial_shape();
    const auto num_batches = im_info_shape[0];

    const ov::Dimension post_nms_count{desc->attrs.post_nms_count};
    const auto first_dim_shape = num_batches * post_nms_count;

    const auto rois_shape = ov::PartialShape({first_dim_shape, 4});
    const auto scores_shape = ov::PartialShape({first_dim_shape});
    const auto roisnum_shape = ov::PartialShape({num_batches});

    std::vector<ShapeType> out_shapes = {rois_shape, scores_shape, roisnum_shape};

    std::vector<layout> out_layouts;
    for (size_t i = 0; i < num_outputs; i++) {
        out_layouts.push_back(layout(out_shapes[i], desc->output_data_types[i].value(), format::get_default_format(out_shapes[i].size())));
    }

    return out_layouts;
}

template std::vector<layout>
generate_proposals_inst::calc_output_layouts<ov::PartialShape>(generate_proposals_node const& node, const kernel_impl_params& impl_param);

layout generate_proposals_inst::calc_output_layout(const generate_proposals_node& node, kernel_impl_params const& impl_param) {
    const layout data_layout = impl_param.get_input_layout();
    const auto num_batches = data_layout.batch();
    const auto desc = impl_param.typed_desc<generate_proposals>();
    return layout(data_layout.data_type, data_layout.format, {static_cast<int>(num_batches * desc->attrs.post_nms_count), 4, 1, 1});
}

std::string generate_proposals_inst::to_string(const generate_proposals_node& node) {
    auto desc = node.get_primitive();

    std::stringstream primitive_description;

    json_composite info;
    info.add("min_size", desc->attrs.min_size);
    info.add("nms_threshold", desc->attrs.nms_threshold);
    info.add("pre_nms_count", desc->attrs.pre_nms_count);
    info.add("post_nms_count", desc->attrs.post_nms_count);
    info.add("normalized", desc->attrs.normalized);
    info.add("nms_eta", desc->attrs.nms_eta);

    auto node_info = node.desc_to_json();
    node_info->add("generate_proposals_info", info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}
}  // namespace cldnn
