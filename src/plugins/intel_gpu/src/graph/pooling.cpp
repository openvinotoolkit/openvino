// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pooling_inst.h"
#include "primitive_type_base.h"
#include "sliding_window_utils.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

using namespace ov::intel_gpu;

namespace cldnn {
primitive_type_id pooling::type_id() {
    static primitive_type_base<pooling> instance;
    return &instance;
}

layout pooling_inst::calc_output_layout(parent::typed_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<pooling>();

    auto input_layout = impl_param.get_input_layout();

    auto pad = desc->pads_begin;
    auto stride = desc->stride;
    auto window_size = desc->size;

    // auto output_type = node.get_primitive()->output_data_type ? *node.get_primitive()->output_data_type : input_layout.data_type;
    // FIXME: dirty hack. Replace it with optional output data type (above) once IE returns correct precision on edges
    auto output_type = input_layout.data_type;

    if (output_type == data_types::u8 || output_type == data_types::i8) {
        if (desc->mode == pooling_mode::average_no_padding || desc->mode == pooling_mode::average) {
            output_type = data_types::f32;
        }
    }

    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_fused_output_layout().data_type;

        // pooling doesn't support i32 data type
        // FIXME: Someday delete this, when pooling supports i32 output.
        if (desc->mode == pooling_mode::max && output_type == data_types::i32) {
            output_type = data_types::f32;
        }
    }

    uint32_t stride_z = stride.size() >= 3 ? stride[stride.size() - 3] : 1;
    uint32_t stride_y = stride.size() >= 2 ? stride[stride.size() - 2] : 1;
    uint32_t stride_x = stride.size() >= 1 ? stride[stride.size() - 1] : 1;

    uint32_t kernel_z = window_size.size() >= 3 ? window_size[window_size.size() - 3] : 1;
    uint32_t kernel_y = window_size.size() >= 2 ? window_size[window_size.size() - 2] : 1;
    uint32_t kernel_x = window_size.size() >= 1 ? window_size[window_size.size() - 1] : 1;

    // TODO: Consider moving general parameter verification to arguments constructor.
    CLDNN_ERROR_LESS_OR_EQUAL_THAN(desc->id,
                                   "stride spatial X",
                                   stride_x,
                                   "",
                                   0,
                                   "Stride spatial X must be positive (>= 1)");
    CLDNN_ERROR_LESS_OR_EQUAL_THAN(desc->id,
                                   "stride spatial Y",
                                   stride_y,
                                   "",
                                   0,
                                   "Stride spatial Y must be positive (>= 1)");
    CLDNN_ERROR_LESS_OR_EQUAL_THAN(desc->id,
                                   "window size spatial X",
                                   kernel_x,
                                   "",
                                   0,
                                   "Size X (of pooling window) must be positive (>= 1)");
    CLDNN_ERROR_LESS_OR_EQUAL_THAN(desc->id,
                                   "window size spatial Y",
                                   kernel_y,
                                   "",
                                   0,
                                   "Size Y (of pooling window) must be positive (>= 1)");
    if (input_layout.format.spatial_num() == 3) {
        // 3D
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(desc->id,
                                       "stride spatial Z",
                                       stride_z,
                                       "",
                                       0,
                                       "Stride spatial Z must be positive (>= 1)");
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(desc->id,
                                       "window size spatial Z",
                                       kernel_z,
                                       "",
                                       0,
                                       "Size Z (of pooling window) must be positive (>= 1)");
    }

    if (desc->with_output_size) {
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(desc->id,
                                       "User-defined size of output X",
                                       desc->output_size.spatial[0],
                                       "",
                                       0,
                                       "User-defined size of output layout (spatial X) must be positive (>= 1)");
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(desc->id,
                                       "User-defined size of output Y",
                                       desc->output_size.spatial[1],
                                       "",
                                       0,
                                       "User-defined size of output layout (spatial Y) must be positive (>= 1)");
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(desc->id,
                                       "User-defined size of output Z",
                                       desc->output_size.spatial[2],
                                       "",
                                       0,
                                       "User-defined size of output layout (spatial Z) must be positive (>= 1)");

        tensor output_size(input_layout.batch(),
                           input_layout.feature(),
                           desc->output_size.spatial[0],
                           desc->output_size.spatial[1],
                           desc->output_size.spatial[2]);
        return {output_type, input_layout.format, output_size};
    }

    // TODO: Check compatibility of output size calculation (with caffe).
    tensor size(1);
    for (size_t i = 0; i < window_size.size(); i++) {
        size.spatial[i] = window_size[window_size.size() - i - 1];
    }
    auto output_range = calc_sliding_window_output_range<swor_mode::exceed_once_data>(input_layout.get_tensor(),
                                                                                      size,
                                                                                      ov::CoordinateDiff(pad.begin(), pad.end()),
                                                                                      stride,
                                                                                      ov::Strides(window_size.size(), 1),
                                                                                      true,
                                                                                      1);

    tensor output_size(input_layout.batch(),
                       input_layout.feature(),
                       output_range.spatial[0],
                       output_range.spatial[1],
                       output_range.spatial[2]);
    return {output_type, input_layout.format, output_size};
}

std::string pooling_inst::to_string(pooling_node const& node) {
    auto desc = node.get_primitive();
    auto strd = desc->stride;
    auto mode = desc->mode == pooling_mode::max ? "max" : "average";
    auto node_info = node.desc_to_json();
    auto kernel_size = desc->size;

    std::stringstream primitive_description;

    json_composite pooling_info;
    pooling_info.add("mode", mode);
    pooling_info.add("stride", cldnn::to_string(strd));
    pooling_info.add("kernel size", cldnn::to_string(kernel_size));
    if (desc->with_output_size) {
        json_composite ud_out_size_info;
        ud_out_size_info.add("size", desc->output_size.to_string());
        pooling_info.add("with_user_defined_output_size", ud_out_size_info);
    }

    node_info->add("pooling info", pooling_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

}  // namespace cldnn
