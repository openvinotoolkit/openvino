/*
// Copyright (c) 2017 Intel Corporation
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

#include "roi_pooling_inst.h"
#include "primitive_type_base.h"
#include "error_handler.h"
#include "json_object.h"

namespace cldnn
{
primitive_type_id roi_pooling_type_id()
{
    static primitive_type_base<roi_pooling> instance;
    return &instance;
}

layout roi_pooling_inst::calc_output_layout(roi_pooling_node const& node)
{
    auto desc = node.get_primitive();
    layout data_layout = node.input().get_output_layout();
    int fm = data_layout.size.feature[0];

    layout rois_layout = node.rois().get_output_layout();
    int num_rois = rois_layout.size.batch[0];

    int gss = desc->group_sz * desc->group_sz;


    CLDNN_ERROR_LESS_THAN(node.id(), "Group size", desc->group_sz, "value", 0, "");
    if (gss && fm % gss != 0)
    {
        CLDNN_ERROR_MESSAGE(node.id(), "group_sz must be either 0 (For RoIPooling) or satisfy fm % (group_sz^2) == 0");
    }
    
    if (gss)
    {
        fm /= gss;
    }

    return layout(rois_layout.data_type, format::bfyx, { num_rois, fm, desc->pooled_width, desc->pooled_height });
}

std::string roi_pooling_inst::to_string(roi_pooling_node const& node)
{
    auto desc      = node.get_primitive();
    auto mode      = desc->mode == pooling_mode::max ? "max" : "average";
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite roi_info;
    roi_info.add("mode", mode);
    roi_info.add("pooled_w", desc->pooled_width);
    roi_info.add("pooled_h", desc->pooled_height);
    roi_info.add("spatial_scale", desc->spatial_scale);
    roi_info.add("group_sz", desc->group_sz);

    node_info.add("roi info", roi_info);
    node_info.dump(primitive_description);

    return primitive_description.str();
}

}
