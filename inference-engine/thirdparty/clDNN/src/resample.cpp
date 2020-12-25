/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "resample_inst.h"
#include "primitive_type_base.h"
#include "error_handler.h"
#include <string>
#include <src/include/json_object.h>

namespace cldnn {
primitive_type_id resample::type_id() {
    static primitive_type_base<resample> instance;
    return &instance;
}

layout resample_inst::calc_output_layout(resample_node const& node) {
    auto desc = node.get_primitive();
    auto input_layout = node.input().get_output_layout();

    auto output_type = input_layout.data_type;
    if ((input_layout.data_type == data_types::i8 || input_layout.data_type == data_types::u8)
        && desc->operation_type != resample_type::nearest) {
        output_type = data_types::f32;
    }
    if (node.has_fused_primitives()) {
        output_type = node.get_fused_output_layout().data_type;
    }

    auto result_sizes = desc->output_size;

    CLDNN_ERROR_NOT_EQUAL(node.id(), "Input batch size", input_layout.size.batch[0], "output batch size", result_sizes.batch[0], "");
    CLDNN_ERROR_NOT_EQUAL(node.id(), "Input feature size", input_layout.size.feature[0], "output feature size", result_sizes.feature[0], "");

    auto result = layout({output_type, input_layout.format, result_sizes});
    return result;
}

std::string resample_inst::to_string(resample_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite resample_info;
    if (desc->operation_type == resample_type::nearest)
        resample_info.add("resample_type:", "nearest_neighbor");
    else if (desc->operation_type == resample_type::bilinear)
        resample_info.add("resample_type:", "bilinear_interp");
    else if (desc->operation_type == resample_type::caffe_bilinear)
        resample_info.add("resample_type:", "caffe_bilinear_interp");
    else if (desc->operation_type == resample_type::cubic)
        resample_info.add("resample_type:", "cubic");
    else
        resample_info.add("resample_type:", "not supported sample type");

    if (desc->shape_calc_mode == shape_calculation_mode::sizes)
        resample_info.add("shape_calculation_mode:", "sizes");
    else
        resample_info.add("shape_calculation_mode:", "scales");

    if (desc->shape_calc_mode == shape_calculation_mode::scales) {
        std::string axesAndScalesDump;
        std::string delim = "";
        for (auto& it : desc->axesAndScales) {
            axesAndScalesDump += delim;
            delim = ", ";
            if (it.first == resample::resample_axis::along_b)
                axesAndScalesDump += "b: ";
            else if (it.first == resample::resample_axis::along_f)
                axesAndScalesDump += "f: ";
            else if (it.first == resample::resample_axis::along_x)
                axesAndScalesDump += "x: ";
            else if (it.first == resample::resample_axis::along_y)
                axesAndScalesDump += "y: ";
            else if (it.first == resample::resample_axis::along_z)
                axesAndScalesDump += "z: ";
            else
                axesAndScalesDump += "w: ";
            axesAndScalesDump += std::to_string(it.second);
        }
        resample_info.add("scales:", axesAndScalesDump);
    }

    if (desc->coord_trans_mode == coordinate_transformation_mode::half_pixel)
        resample_info.add("coordinate_transformation_mode:", "half_pixel");
    else if (desc->coord_trans_mode == coordinate_transformation_mode::pytorch_half_pixel)
        resample_info.add("coordinate_transformation_mode:", "pytorch_half_pixel");
    else if (desc->coord_trans_mode == coordinate_transformation_mode::tf_half_pixel_for_nn)
        resample_info.add("coordinate_transformation_mode:", "tf_half_pixel_for_nn");
    else if (desc->coord_trans_mode == coordinate_transformation_mode::align_corners)
        resample_info.add("coordinate_transformation_mode:", "align_corners");
    else
        resample_info.add("coordinate_transformation_mode:", "asymmetric");

    if (desc->round_mode == nearest_mode::round_prefer_floor)
        resample_info.add("nearest_mode:", "round_prefer_floor");
    if (desc->round_mode == nearest_mode::round_prefer_ceil)
        resample_info.add("nearest_mode:", "round_prefer_ceil");
    if (desc->round_mode == nearest_mode::floor)
        resample_info.add("nearest_mode:", "floor");
    if (desc->round_mode == nearest_mode::ceil)
        resample_info.add("nearest_mode:", "ceil");
    else
        resample_info.add("nearest_mode:", "simple");

    resample_info.add("output_size", desc->output_size);
    resample_info.add("output padding lower size", desc->output_padding.lower_size());
    resample_info.add("output padding upper size", desc->output_padding.upper_size());

    if (desc->operation_type == resample_type::bilinear) {
        resample_info.add("align_corners", desc->align_corners);
    }

    node_info->add("resample_info", resample_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

resample_inst::typed_primitive_inst(network_impl& network, resample_node const& node) : parent(network, node) {
    if (node.get_primitive()->operation_type == resample_type::bilinear &&
        node.get_output_layout().format.dimension() > 4) {
        CLDNN_ERROR_MESSAGE(node.id(), "5D not supported for interp resample type.");
    }
}
}  // namespace cldnn
