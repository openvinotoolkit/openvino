// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <roi_align_inst.h>
#include "primitive_type_base.h"
#include <sstream>
#include <json_object.h>
#include "openvino/core/enum_names.hpp"
#include "roi_align_shape_inference.hpp"
#include "roi_align_rotated_shape_inference.hpp"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(roi_align)

roi_align_inst::typed_primitive_inst(network& network, roi_align_node const& node)
    : parent(network, node) {}

layout roi_align_inst::calc_output_layout(roi_align_node const& node, kernel_impl_params const& impl_param) {
    auto primitive = impl_param.typed_desc<roi_align>();
    auto input_layout = impl_param.get_input_layout(0);
    auto rois_layout = impl_param.get_input_layout(1);
    auto num_rois = rois_layout.batch();
    auto num_channels = input_layout.feature();
    return layout({num_rois, num_channels, primitive->pooled_h, primitive->pooled_w},
                  input_layout.data_type,
                  input_layout.format);
}

template<typename ShapeType>
std::vector<layout> roi_align_inst::calc_output_layouts(roi_align_node const& node, kernel_impl_params const& impl_param) {
    auto primitive = impl_param.typed_desc<roi_align>();

    auto input0_layout  = impl_param.get_input_layout(0);
    auto input1_layout  = impl_param.get_input_layout(1);
    auto input2_layout  = impl_param.get_input_layout(2);

    std::vector<ShapeType> input_shapes = {
        input0_layout.get<ShapeType>(),     // input shape
        input1_layout.get<ShapeType>(),     // roi shape
        input2_layout.get<ShapeType>()      // batch indices shape
    };

    std::vector<ShapeType> output_shapes;

#define PERFORM_SHAPE_INFERENCE(ROI_ALIGN_CLASS_NAME) \
    ROI_ALIGN_CLASS_NAME op;                          \
    op.set_pooled_h(primitive->pooled_h);             \
    op.set_pooled_w(primitive->pooled_w);             \
    output_shapes = shape_infer(&op, input_shapes);

    if (primitive->roi_mode == roi_align::ROIMode::rotated) {
        PERFORM_SHAPE_INFERENCE(ov::op::v15::ROIAlignRotated);
    } else {
        PERFORM_SHAPE_INFERENCE(ov::op::v3::ROIAlign);
    }

#undef PERFORM_SHAPE_INFERENCE

    return { layout{output_shapes[0], input0_layout.data_type, input0_layout.format} };
}

template
std::vector<layout> roi_align_inst::calc_output_layouts<ov::PartialShape>(roi_align_node const& node, const kernel_impl_params& impl_param);

std::string roi_align_inst::to_string(roi_align_node const& node) {
    auto node_info = node.desc_to_json();
    json_composite roi_align_info;
    roi_align_info.add("input id", node.input().id());
    roi_align_info.add("rois id", node.get_dependency(1).id());
    roi_align_info.add("batches id", node.get_dependency(2).id());
    roi_align_info.add("pooled_h", node.get_primitive()->pooled_h);
    roi_align_info.add("pooled_w", node.get_primitive()->pooled_w);
    roi_align_info.add("sampling_ratio", node.get_primitive()->sampling_ratio);
    roi_align_info.add("spatial_scale", node.get_primitive()->spatial_scale);
    roi_align_info.add("pooling_mode", ov::as_string(node.get_primitive()->pooling_mode));
    roi_align_info.add("aligned_mode", ov::as_string(node.get_primitive()->aligned_mode));
    node_info->add("roi_align info", roi_align_info);
    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

}  // namespace cldnn

namespace ov {
using cldnn::roi_align;

template <> EnumNames<roi_align::PoolingMode>& EnumNames<roi_align::PoolingMode>::get() {
  static auto enum_names =
      EnumNames<roi_align::PoolingMode>("PoolingMode", {{"max", roi_align::PoolingMode::max},
                                                        {"avg", roi_align::PoolingMode::avg}});
  return enum_names;
}

template <> EnumNames<roi_align::AlignedMode>& EnumNames<roi_align::AlignedMode>::get() {
  static auto enum_names =
      EnumNames<roi_align::AlignedMode>("AlignedMode", {{"asymmetric", roi_align::AlignedMode::asymmetric},
                                                        {"half_pixel_for_nn", roi_align::AlignedMode::half_pixel_for_nn},
                                                        {"half_pixel", roi_align::AlignedMode::half_pixel}});
  return enum_names;
}
} // namespace ov
