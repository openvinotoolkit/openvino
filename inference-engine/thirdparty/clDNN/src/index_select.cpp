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

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "index_select_inst.h"
#include "primitive_type_base.h"
#include "error_handler.h"
#include "json_object.h"

namespace cldnn
{
	primitive_type_id index_select_type_id()
	{
		static primitive_type_base<index_select> instance;
		return &instance;
	}

	layout index_select_inst::calc_output_layout(index_select_node const& node)
	{
		auto desc = node.get_primitive();

        auto input_layout = node.input().get_output_layout();
        auto indices_layout = node.indices().get_output_layout();
        auto indices_size = indices_layout.size.spatial[0];

        auto axis = node.get_axis();
        
        int32_t output_b = input_layout.size.batch[0];
        int32_t output_f = input_layout.size.feature[0];
        int32_t output_x = input_layout.size.spatial[0];
        int32_t output_y = input_layout.size.spatial[1];

        switch (axis)
        {
        case index_select_axis_name::along_b:
            output_b = indices_size;
            break;
        case index_select_axis_name::along_f:
            output_f = indices_size;
            break;
        case index_select_axis_name::along_x:
            output_x = indices_size;
            break;
        case index_select_axis_name::along_y:
            output_y = indices_size;
            break;
        default:
            CLDNN_ERROR_MESSAGE(node.id(), "UNSPORTTED AXIS");
            break;
        }
        return layout{ input_layout.data_type, input_layout.format, { output_b, output_f, output_x, output_y } };
	}

	std::string index_select_inst::to_string(index_select_node const& node)
	{
		auto desc = node.get_primitive();
		auto node_info = node.desc_to_json();
		std::stringstream primitive_description;

        std::string axis_str = "";
        switch (desc->axis)
        {
        case index_select_axis_name::along_b:
            axis_str = "along_b";
            break;
        case index_select_axis_name::along_f:
            axis_str = "along_f";
            break;
        case index_select_axis_name::along_y:
            axis_str = "along_y";
            break;
        case index_select_axis_name::along_x:
            axis_str = "along_x";
            break;
        default:
            axis_str = "not supported axis";
            break;
        }

        json_composite index_select_info;
        index_select_info.add("axis", axis_str);

        node_info->add("index_select_info", index_select_info);
		node_info->dump(primitive_description);

		return primitive_description.str();
	}

    index_select_inst::typed_primitive_inst(network_impl& network, index_select_node const& node)
		: parent(network, node)
	{
        auto& input = node.input();
        auto input_layout = input.get_output_layout();
        auto& indices = node.indices();
        auto indices_layout = indices.get_output_layout();
        auto const node_id = node.id();

        CLDNN_ERROR_DATA_TYPES_MISMATCH(node_id, "indicies data_type", indices_layout.data_type, "i32 data_type ", data_types::i32, "");
        CLDNN_ERROR_NOT_PROPER_FORMAT(node_id, "input_format", input_layout.format, "supported input format", format::bfyx, format::yxfb);
        CLDNN_ERROR_NOT_PROPER_FORMAT(node_id, "input_format", indices_layout.format, "supported indicies format", format::bfyx, format::yxfb);
        CLDNN_ERROR_NOT_EQUAL(node_id, "indicies batch_size", indices_layout.size.batch[0], "expected size", 1, "");
        CLDNN_ERROR_NOT_EQUAL(node_id, "indicies feature_size", indices_layout.size.feature[0], "expected size", 1, "");
        CLDNN_ERROR_NOT_EQUAL(node_id, "indicies y_size", indices_layout.size.spatial[1], "expected size", 1, "");
        CLDNN_ERROR_LESS_THAN(node_id, "indicies x_size", indices_layout.size.spatial[0], "expected size", 1, "");

	}
}
