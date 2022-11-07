// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pooling_inst.h"
#include "sliding_window_utils.hpp"

#include "ngraph/validation_util.hpp"

#include "intel_gpu/runtime/error_handler.hpp"
#include "primitive_type_base.h"
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

template<typename ShapeType>
std::vector<layout> pooling_inst::calc_output_layouts(pooling_node const& /*node*/, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<pooling>();
    auto input_layout = impl_param.get_input_layout();
    auto input_shape = input_layout.get<ShapeType>();

    auto output_dtype = input_layout.data_type;
    if (output_dtype == data_types::u8 || output_dtype == data_types::i8) {
        if (desc->mode == pooling_mode::average_no_padding || desc->mode == pooling_mode::average) {
            output_dtype = data_types::f32;
        }
    }
    if (impl_param.has_fused_primitives()) {
        output_dtype = impl_param.get_fused_output_layout().data_type;

        // pooling doesn't support i32 data type
        // FIXME: Someday delete this, when pooling supports i32 output.
        if (desc->mode == pooling_mode::max && output_dtype == data_types::i32) {
            output_dtype = data_types::f32;
        }
    }

    ShapeType output_shape = ShapeType::dynamic(input_shape.size());
    output_shape[0] = input_shape[0];
    output_shape[1] = input_shape[1];

    if (input_shape.is_dynamic()) {
        return { layout{output_shape, input_layout.data_type, input_layout.format} };
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
        return {{output_dtype, input_layout.format, output_size}};
    }

    auto kernel_size = desc->size;
    auto stride = desc->stride;
    auto dilation = desc->dilation.empty() ? ov::Strides(stride.size(), 1)
                                           : desc->dilation;
    bool ceil_mod = desc->rounding_type == ov::op::RoundingType::CEIL;

    auto is_positive_values = [](const std::vector<size_t>& values) {
        return !std::any_of(values.begin(), values.end(), [](size_t val) { return val == 0; });
    };

    OPENVINO_ASSERT(is_positive_values(kernel_size), "Size of pooling window must be positive (>= 1)");
    OPENVINO_ASSERT(is_positive_values(stride), "Strides must be positive (>= 1)");
    OPENVINO_ASSERT(is_positive_values(dilation), "Dilations must be positive (>= 1)");

    ov::CoordinateDiff pads_begin(desc->pads_begin.begin(), desc->pads_begin.end());
    ov::CoordinateDiff pads_end(desc->pads_end.begin(), desc->pads_end.end());
    auto auto_pad = desc->auto_pad;

    if (auto_pad == ov::op::PadType::SAME_UPPER || auto_pad == ov::op::PadType::SAME_LOWER) {
        pads_begin.clear();
        pads_end.clear();
        ngraph::try_apply_auto_padding(input_shape,
                                       kernel_size,
                                       stride,
                                       dilation,
                                       auto_pad,
                                       pads_end,
                                       pads_begin);
    }
    if (auto_pad == ov::op::PadType::VALID) {
        pads_begin = ov::CoordinateDiff(pads_begin.size(), 0);
        pads_end = ov::CoordinateDiff(pads_end.size(), 0);
    }

    size_t spatial_size = input_shape.size() - 2;
    for (size_t i = 0; i < spatial_size; ++i) {
        int64_t padded_input_dim = input_shape[i + 2].get_length() + pads_begin[i] + pads_end[i];
        int64_t kernel_dilated_dim = dilation[i] * (kernel_size[i] - 1) + 1;
        int64_t out_dim = ceil_mod ? ceil_div(padded_input_dim - kernel_dilated_dim, stride[i]) + 1 :
                                     (padded_input_dim - kernel_dilated_dim) / stride[i] + 1;
        output_shape[i + 2] = out_dim;
    }

    return { layout{output_shape, output_dtype, input_layout.format} };
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
