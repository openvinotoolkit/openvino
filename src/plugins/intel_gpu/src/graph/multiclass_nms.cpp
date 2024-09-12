// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/multiclass_nms.hpp"
#include "multiclass_nms_shape_inference.hpp"
#include <string>

#include "json_object.h"
#include "multiclass_nms_inst.h"
#include "primitive_type_base.h"

#include "openvino/op/multiclass_nms.hpp"
#include "multiclass_nms_shape_inference.hpp"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(multiclass_nms)

template<typename ShapeType>
std::vector<layout> multiclass_nms_inst::calc_output_layouts(multiclass_nms_node const& /*node*/, const kernel_impl_params& impl_param) {
    auto input_layout = impl_param.get_input_layout();
    const auto desc = impl_param.typed_desc<multiclass_nms>();
    const auto attrs = desc->attrs;

    ov::op::v9::MulticlassNms op;
    op.set_attrs(attrs);
    op.set_output_size(desc->output_size());

    std::vector<ShapeType> input_shapes;
    for (size_t i = 0; i < desc->input_size(); i++) {
        input_shapes.push_back(impl_param.get_input_layout(i).get<ShapeType>());
    }

    const auto out_shapes = shape_infer(&op, input_shapes, false, true);

    std::vector<layout> out_layouts;
    for (size_t i = 0; i < desc->output_size(); i++) {
        out_layouts.push_back(layout(out_shapes[i].get_max_shape(), desc->output_data_types[i].value(), format::get_default_format(out_shapes[i].size())));
    }

    return out_layouts;
}

template std::vector<layout> multiclass_nms_inst::calc_output_layouts<ov::PartialShape>(multiclass_nms_node const& node, const kernel_impl_params& impl_param);

std::string multiclass_nms_inst::to_string(const multiclass_nms_node& node) {
    const auto attrs = node.get_primitive()->attrs;
    std::stringstream primitive_description;

    json_composite ed_info;
    ed_info.add("sort_result_type", static_cast<int>(attrs.sort_result_type));
    ed_info.add("sort_result_across_batch", attrs.sort_result_across_batch);
    ed_info.add("iou_threshold", attrs.iou_threshold);
    ed_info.add("score_threshold", attrs.score_threshold);
    ed_info.add("nms_top_k", attrs.nms_top_k);
    ed_info.add("keep_top_k", attrs.keep_top_k);
    ed_info.add("background_class", attrs.background_class);
    ed_info.add("normalized", attrs.normalized);
    ed_info.add("nms_eta", attrs.nms_eta);

    auto node_info = node.desc_to_json();
    node_info->add("multiclass_nms_info", ed_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}
}  // namespace cldnn
