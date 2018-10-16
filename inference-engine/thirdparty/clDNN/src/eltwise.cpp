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
#include "eltwise_inst.h"
#include "primitive_type_base.h"
#include "error_handler.h"
#include "json_object.h"

namespace cldnn
{
primitive_type_id eltwise_type_id()
{
    static primitive_type_base<eltwise> instance;
    return &instance;
}

layout eltwise_inst::calc_output_layout(eltwise_node const& node)
{
    return node.input().get_non_padded_output_layout();
}

static inline std::string stringify_vector(std::vector<float> v)
{
    std::stringstream s;

    s << "{ ";

    for (size_t i = 0; i < v.size(); ++i)
    {
        s << v.at(i);
        if (i + 1 < v.size()) s << ", ";
    }

    s << " }";

    return s.str();
}

std::string eltwise_inst::to_string(eltwise_node const& node)
{
    auto node_info  = node.desc_to_json();
    auto desc       = node.get_primitive();
    auto activation = desc->with_activation ? " true" : "false";

    std::stringstream primitive_description;
    std::string       str_mode;

    switch(desc->mode)
    {
    case eltwise_mode::sum:
            str_mode = "sum";
            break;
    case eltwise_mode::sub:
            str_mode = "subtract";
            break;
    case eltwise_mode::max:
            str_mode = "max";
            break;
    case eltwise_mode::prod:
            str_mode = "product";
            break;
    case eltwise_mode::div:
            str_mode = "div";
            break;
    case eltwise_mode::min:
            str_mode = "min";
         break;
    case eltwise_mode::pow:
            str_mode = "pow";
            break;
    case eltwise_mode::mod:
            str_mode = "mod";
            break;
    default:
            str_mode = "not supported mode";
            break;
    }

    json_composite eltwise_info;
    for (size_t i = 0; i < node.inputs_count(); i++)
    {
        eltwise_info.add("input_"+std::to_string(i), node.input(i).id());
    }
    eltwise_info.add("mode", str_mode);
    if (desc->mode == eltwise_mode::sum)
    {
        eltwise_info.add("coefficients", stringify_vector(desc->coefficients));
    }
    if (desc->with_activation)
    {
        eltwise_info.add("with activation", activation);
        eltwise_info.add("slope", desc->activation_negative_slope);
    }
    node_info.add("eltwise info", eltwise_info);
    node_info.dump(primitive_description);

    return primitive_description.str();
}

eltwise_inst::typed_primitive_inst(network_impl& network, eltwise_node const& node)
    :parent(network, node)
{
    auto input_layout = node.input().get_output_layout();
    auto batch_size = input_layout.size.batch[0];
    auto feature_size = input_layout.size.feature[0];

    auto input_batch_size = input_layout.size.batch[0];
    auto input_feature_size = input_layout.size.feature[0];

    if (batch_size != 1)
    {
        CLDNN_ERROR_NOT_EQUAL(node.id(), "Eltwise batch size", batch_size, "input batch size", input_batch_size, "");
    }

    if (feature_size != 1)
    {
        CLDNN_ERROR_NOT_EQUAL(node.id(), "Eltwise feature size", feature_size, "input feature size", input_feature_size, "");
    }
}
}
