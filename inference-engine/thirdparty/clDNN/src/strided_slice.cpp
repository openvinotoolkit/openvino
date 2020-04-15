/*
// Copyright (c) 2019 Intel Corporation
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

#include "strided_slice_inst.h"
#include "primitive_type_base.h"
#include "error_handler.h"
#include "json_object.h"
#include "data_inst.h"
#include <string>
#include <vector>

namespace cldnn {
primitive_type_id strided_slice::type_id() {
    static primitive_type_base<strided_slice> instance;
    return &instance;
}

layout strided_slice_inst::calc_output_layout(strided_slice_node const& node) {
    auto desc = node.get_primitive();
    auto input_layout = node.input(0).get_output_layout();
    auto input_format = input_layout.format;
    const size_t dims_num = format::dimension(input_format);
    format dimension_format = get_default_format_for_dim(dims_num);

    // Getting data from constant inputs. There are 3 args: Begin, End, Stride
    std::vector<std::vector<int32_t>> args;
    for (size_t i = 1; i < node.get_dependencies().size(); ++i) {
        auto& input = node.get_dependency(i).as<data>();
        auto& mem = input.get_attached_memory();
        int32_t* data = static_cast<int32_t*>(mem.lock());
        std::vector<int32_t> sizes = std::vector<int32_t>(data, data + input.get_output_layout().count());
        pad_vector_to_size(sizes, dims_num, i != 1);  // for "begin" completion used 0 value, for other - 1
        args.push_back(sizes);
        mem.unlock();
    }

    std::vector<uint8_t> begin_mask(desc->begin_mask);
    pad_vector_to_size(begin_mask, dims_num, 1);
    std::vector<uint8_t> end_mask(desc->end_mask);
    pad_vector_to_size(end_mask, dims_num, 1);

    auto& begin = args[0];
    auto& end = args[1];
    const auto& strides = args[2];

    for (size_t i = 0; i < dims_num; ++i) {
        auto max_size = input_layout.size.sizes(dimension_format)[i];
        if (end[i] > max_size) {
            end[i] = max_size;
        } else if (end[i] < 0) {
            end[i] = end[i] % max_size;
        }
        if (begin[i] < 0) {
            begin[i] = begin[i] % max_size;
        }
    }

    // If the ith bit of begin_mask is not set, begin[i] is ignored and the range of the appropriate dimension starts from 0.
    vector_assign_if_not_mask(begin, 0, begin_mask);
    // If the ith bit of end_mask is not set, end[i] is ignored and the fullest possible range in that dimension is used
    // instead.
    vector_assign_if_not_mask(end, input_layout.size.sizes(dimension_format), end_mask);

    std::vector<int32_t> output_shape;
    if (std::find(desc->new_axis_mask.begin(), desc->new_axis_mask.end(), 1) == desc->new_axis_mask.end()) {
        for (size_t i = 0; i < dims_num; ++i) {
            int32_t b = begin[i] < 0 ? input_layout.size.sizes(input_format)[i] - 1 : begin[i];
            int32_t e = end[i] < 0 ? input_layout.size.sizes(input_format)[i] - 1 : end[i];
            int32_t s = strides[i];
            int32_t outputDimSize = std::abs((e - b) / s);
            if ((e - b) % s != 0)
                outputDimSize++;
            output_shape.push_back(outputDimSize);
        }
    } else {
        output_shape = input_layout.size.sizes(input_format);
    }

    if (input_format == format::bfzyx)
        return layout{input_layout.data_type,
                      input_format,
                      tensor(batch(output_shape[0]), feature(output_shape[1]), spatial(output_shape[4], output_shape[3],
                                                                                       output_shape[2]))};
    else
        return layout{input_layout.data_type,
                      input_format,
                      tensor(batch(output_shape[0]), feature(output_shape[1]), spatial(output_shape[3], output_shape[2]))};
}

std::string strided_slice_inst::to_string(strided_slice_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite strided_slice_info;
    strided_slice_info.add("input id", input.id());
    strided_slice_info.add("begin_param id", node.get_dependency(1).id());
    strided_slice_info.add("end_param id", node.get_dependency(2).id());
    strided_slice_info.add("stride_param id", node.get_dependency(3).id());
    strided_slice_info.add("begin mask", node.get_primitive()->begin_mask);
    strided_slice_info.add("end mask", node.get_primitive()->end_mask);
    strided_slice_info.add("new axis mask", node.get_primitive()->new_axis_mask);
    strided_slice_info.add("shrink axis mask", node.get_primitive()->shrink_axis_mask);
    strided_slice_info.add("begin_param shape", node.get_dependency(1).get_output_layout().size.to_string());
    strided_slice_info.add("end_param shape", node.get_dependency(2).get_output_layout().size.to_string());
    strided_slice_info.add("stride_param shape", node.get_dependency(3).get_output_layout().size.to_string());

    node_info->add("strided_slice info", strided_slice_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

strided_slice_inst::typed_primitive_inst(network_impl& network, strided_slice_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
