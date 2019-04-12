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

namespace cldnn
{
primitive_type_id strided_slice_type_id()
{
    static primitive_type_base<strided_slice> instance;
    return &instance;
}

layout strided_slice_inst::calc_output_layout(strided_slice_node const& node) {
    const size_t numberOfDims = 4;
    auto desc = node.get_primitive();
    auto input_layout = node.input(0).get_output_layout();
    auto input_format = input_layout.format;

    auto completeStridedSliceParams = [&](std::vector<int32_t>& param) {
        for (size_t i = param.size(); i < numberOfDims; ++i)
            param.push_back(1);
    };

    auto completeStridedSliceMasks = [&](std::vector<uint8_t>& mask) {
        for (size_t i = mask.size(); i < numberOfDims; ++i)
            mask.push_back(0);
    };

    auto maskStridedSliceParams = [&](std::vector<int32_t>& param, const std::vector<uint8_t>& mask) {
        for (size_t i = 0; i < param.size(); ++i)
            if (mask[i])
                param[i] = input_layout.size.sizes(format::bfyx)[i];
    };

    // Getting data from constant inputs. There are 3 args: Begin, End, Stride
    std::vector<std::vector<int32_t>> stridedSliceArgs;
    for (size_t i = 1; i < node.get_dependencies().size(); ++i) {
        auto& input = node.get_dependency(i).as<data>();
        auto& mem = input.get_attached_memory();
        int32_t* data = static_cast<int32_t*>(mem.lock());
        std::vector<int32_t> vData = std::vector<int32_t>(data, data + input.get_output_layout().count());
        completeStridedSliceParams(vData);
        stridedSliceArgs.push_back(vData);
        mem.unlock();
    }

    std::vector<uint8_t> beginMask(desc->begin_mask);
    completeStridedSliceMasks(beginMask);
    std::vector<uint8_t> endMask(desc->end_mask);
    completeStridedSliceMasks(endMask);

    auto& begin = stridedSliceArgs[0];
    auto& end = stridedSliceArgs[1];
    const auto& strides = stridedSliceArgs[2];
    std::vector<int32_t> outputDimsSizes;

    // If the ith bit of begin_mask is set, begin[i] is ignored and the fullest possible range in that dimension is used instead.
    maskStridedSliceParams(begin, beginMask);
    // end_mask works analogously
    maskStridedSliceParams(end, endMask);

    auto isShiftPossible = [] (std::vector<int32_t>& dims) -> bool {
        if (dims[dims.size() - 1] == 1)
            return true;
        else
            return false;
    };

    // If the new_axis_mask is set, then begin, end, and stride are ignored
    if (std::find(desc->new_axis_mask.begin(), desc->new_axis_mask.end(), 1) == desc->new_axis_mask.end()) {
        for (size_t i = 0; i < numberOfDims; ++i) {
            int32_t outputDimSize = (end[i] - begin[i]) / strides[i];
            if ((end[i] - begin[i]) % strides[i] != 0)
                outputDimSize++;
            outputDimsSizes.push_back(outputDimSize);
        }
    } else {
        outputDimsSizes = input_layout.size.sizes(format::bfyx);
        for (size_t i = 0; i < desc->new_axis_mask.size(); ++i)
            if (desc->new_axis_mask[desc->new_axis_mask.size() - i - 1] == 1)
                if (isShiftPossible(outputDimsSizes)) {
                    for (size_t j = outputDimsSizes.size() - 1; j > i; --j)
                        outputDimsSizes[j] = outputDimsSizes[j - 1];
                    outputDimsSizes[i] = 1;
                }
    }

    return layout{input_layout.data_type, input_format, tensor(outputDimsSizes[0], outputDimsSizes[1], outputDimsSizes[3], outputDimsSizes[2])};
}

std::string strided_slice_inst::to_string(strided_slice_node const& node)
{
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
    : parent(network, node)
{
}

}
