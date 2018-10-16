/*
// Copyright (c) 2018 Intel Corporation
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

#include "reorg_yolo_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"

namespace cldnn
{
    primitive_type_id reorg_yolo_type_id()
    {
        static primitive_type_base<reorg_yolo> instance;
        return &instance;
    }

    layout reorg_yolo_inst::calc_output_layout(reorg_yolo_node const& node)
    {
        auto input_layout = node.input().get_output_layout();
        auto desc = node.get_primitive();
        auto stride = desc->stride;

        cldnn::layout layoutTemp = cldnn::layout(input_layout.data_type, input_layout.format, tensor(input_layout.size.batch[0],
            input_layout.size.feature[0] * stride * stride, input_layout.size.spatial[0] / stride,
            input_layout.size.spatial[1] / stride));
        return layoutTemp;
    }

    std::string reorg_yolo_inst::to_string(reorg_yolo_node const& node)
    {
        auto desc = node.get_primitive();
        auto node_info = node.desc_to_json();
        auto stride = desc->stride;

        std::stringstream primitive_description;

        json_composite reorg_yolo_info;
        reorg_yolo_info.add("stride", stride);


        node_info.add("reorg yolo info", reorg_yolo_info);
        node_info.dump(primitive_description);

        return primitive_description.str();
    }
    reorg_yolo_inst::typed_primitive_inst(network_impl& network, reorg_yolo_node const& node)
        : parent(network, node)
    {
    }
}
