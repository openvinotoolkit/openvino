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


#include "contract_inst.h"

#include "error_handler.h"
#include "json_object.h"
#include "primitive_type_base.h"


namespace cldnn
{
    primitive_type_id contract_type_id()
    {
        static primitive_type_base<contract> instance;
        return &instance;
    }

    layout contract_inst::calc_output_layout(contract_node const& node)
    {
        auto input_layout = node.input().get_output_layout();
        const auto& input_sizes = input_layout.size;
        auto desc = node.get_primitive();
        auto reduction_axes = desc->reduction_axes;

        std::vector<tensor::value_type> input_dims = { input_sizes.batch[0], input_sizes.feature[0],
            input_sizes.spatial[1], input_sizes.spatial[0] };
        std::vector<tensor::value_type> output_sizes(4, 0);
        int cur_dim = 3;
        for (int i = 3; i >= 0; --i)
        {
            while (std::find(reduction_axes.begin(), reduction_axes.end(), cur_dim) != reduction_axes.end() && cur_dim >= 0)
                --cur_dim;
            output_sizes.at(i) = cur_dim >= 0 ? input_dims.at(cur_dim--) : 1;
        }

        return { input_layout.data_type, input_layout.format, cldnn::tensor(output_sizes[0], output_sizes[1], output_sizes[3], output_sizes[2]) };
    }

    std::string contract_inst::to_string(contract_node const& node)
    {
        auto desc = node.get_primitive();
        auto node_info = node.desc_to_json();
        const auto& reduction_axes = desc->reduction_axes;
        auto& input = node.input();

        std::stringstream primitive_description;
        std::stringstream ss_reduction_axes;

        for (size_t i = 0; i < reduction_axes.size(); ++i)
        {
            ss_reduction_axes << reduction_axes.at(i);
            i != (reduction_axes.size() - 1) ? ss_reduction_axes << ", " : ss_reduction_axes << "";
        }

        std::string str_mode;
        switch (desc->mode)
        {
        case contract_mode::sum:
            str_mode = "sum";
            break;
        case contract_mode::prod:
            str_mode = "product";
            break;
        case contract_mode::all:
            str_mode = "all";
            break;
        case contract_mode::any:
            str_mode = "any";
            break;
        case contract_mode::max:
            str_mode = "max";
            break;
        default:
            str_mode = "not supported mode";
            break;
        }

        json_composite contract_info;
        contract_info.add("input id", input.id());
        contract_info.add("mode", str_mode);
        contract_info.add("reduction axes", ss_reduction_axes.str());

        node_info->add("contract info", contract_info);
        node_info->dump(primitive_description);

        return primitive_description.str();
    }

    contract_inst::typed_primitive_inst(network_impl& network, contract_node const& node)
        : parent(network, node)
    {
        std::set<uint16_t> existing;
        const auto& reduction_axes = node.get_primitive()->reduction_axes;
        size_t reduction_axes_size = reduction_axes.size();

        if (reduction_axes.empty())
        {
            CLDNN_ERROR_MESSAGE(node.id(), "Incorrect parameters configuration: reduction_axes should not be empty.");
        }
        if (reduction_axes_size > 4)
        {
            CLDNN_ERROR_MESSAGE(node.id(), "Incorrect parameters configuration: reduction_axes size should be less or equal 4.");
        }
        for (size_t i = 0; i < reduction_axes_size; ++i)
        {
            if (reduction_axes.at(i) >= 4)
            {
                CLDNN_ERROR_MESSAGE(node.id(), "Incorrect parameters configuration: reduction_axes index should be within reduction_axes range.");
            }
            if (existing.find(reduction_axes.at(i)) != existing.end())
            {
                CLDNN_ERROR_MESSAGE(node.id(), "Incorrect parameters configuration: Duplicate axes numbers was found in reduction_axes.");
            }
            existing.insert(reduction_axes.at(i));
        }
    }
}
