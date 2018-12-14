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

#include "lrn_inst.h"
#include "primitive_type_base.h"
#include "error_handler.h"
#include "json_object.h"

namespace cldnn
{
primitive_type_id lrn_type_id()
{
    static primitive_type_base<lrn> instance;
    return &instance;
}

layout lrn_inst::calc_output_layout(lrn_node const& node)
{
    return node.input().get_non_padded_output_layout();
}

std::string lrn_inst::to_string(lrn_node const& node)
{
    auto node_info   = node.desc_to_json();
    auto desc        = node.get_primitive();
    auto norm_size   = desc->size;
    auto k           = desc->k;
    auto alpha       = desc->alpha;
    auto beta        = desc->beta;
    auto norm_region = desc->norm_region == cldnn_lrn_norm_region::cldnn_lrn_norm_region_across_channel ? "across channel" : "within channel";
    auto& input      = node.input();

    std::stringstream primitive_description;

    json_composite lrn_info;
    lrn_info.add("input id", input.id());
    lrn_info.add("k", k);
    lrn_info.add("alpha", alpha);
    lrn_info.add("beta", beta);
    lrn_info.add("size of normalization", norm_size);
    lrn_info.add("normalization region", norm_region);

    node_info->add("lrn info", lrn_info);
    node_info->dump(primitive_description);
   
    return primitive_description.str();
}

lrn_inst::typed_primitive_inst(network_impl& network, lrn_node const& desc)
    :parent(network, desc)
{
    CLDNN_ERROR_LESS_OR_EQUAL_THAN(desc.id(), "LRN argument size", argument.size, "value", static_cast<uint32_t>(0), "LRN size must be greater than 0!");
}
}
