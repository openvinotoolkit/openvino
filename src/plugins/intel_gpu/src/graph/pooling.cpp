// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>

#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include "max_pool_shape_inference.hpp"
#include "openvino/core/validation_util.hpp"
#include "pooling_inst.h"
#include "primitive_type_base.h"
#include "sliding_window_utils.hpp"

using namespace ov::intel_gpu;

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(pooling)

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
        output_dtype = impl_param.get_output_element_type();

        // pooling doesn't support i32 data type
        // FIXME: Someday delete this, when pooling supports i32 output.
        if (desc->mode == pooling_mode::max && output_dtype == data_types::i32) {
            output_dtype = data_types::f32;
        }
    }

    ShapeType output_shape = ShapeType::dynamic(input_shape.size());
    output_shape[0] = input_shape[0];
    output_shape[1] = input_shape[1];

    std::vector<layout> out_layouts = {
        layout{output_shape, output_dtype, input_layout.format}
    };

    if (desc->num_outputs == 2) {
        auto l = out_layouts[0];
        l.data_type = desc->index_element_type;
        out_layouts.push_back(l);
    }

    if (input_shape.is_dynamic()) {
        return out_layouts;
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

    ov::op::v8::MaxPool op;
    op.set_strides(stride);
    op.set_kernel(kernel_size);
    op.set_auto_pad(auto_pad);

    ov::op::pooling::apply_padding(&op, input_layout.get_partial_shape(), dilation, pads_begin, pads_end);

    size_t spatial_size = input_shape.size() - 2;
    for (size_t i = 0; i < spatial_size; ++i) {
        int64_t padded_input_dim = input_shape[i + 2].get_length() + pads_begin[i] + pads_end[i];
        int64_t kernel_dilated_dim = dilation[i] * (kernel_size[i] - 1) + 1;
        int64_t out_dim = ceil_mod ? ceil_div(padded_input_dim - kernel_dilated_dim, stride[i]) + 1 :
                                     (padded_input_dim - kernel_dilated_dim) / stride[i] + 1;
        output_shape[i + 2] = out_dim;
    }

    for (auto& ol : out_layouts) {
        ol.set_partial_shape(output_shape);
    }

    return out_layouts;
}

template std::vector<layout> pooling_inst::calc_output_layouts<ov::PartialShape>(pooling_node const& node, const kernel_impl_params& impl_param);

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
    node_info->add("pooling info", pooling_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

}  // namespace cldnn
