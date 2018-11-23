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
#include "arg_max_min_inst.h"
#include "primitive_type_base.h"
#include "sliding_window_utils.h"
#include "error_handler.h"
#include "json_object.h"

namespace cldnn
{
	primitive_type_id arg_max_min_type_id()
	{
		static primitive_type_base<arg_max_min> instance;
		return &instance;
	}

	layout arg_max_min_inst::calc_output_layout(arg_max_min_node const& node)
	{
		auto desc = node.get_primitive();

		auto input_layout = node.input().get_output_layout();

		if (desc->with_axis){
			switch (desc->axis){
				case arg_max_min::x:
					return layout{ data_types::f32, format::bfyx, tensor{ input_layout.size.batch[0], input_layout.size.feature[0], (int32_t)desc->top_k, input_layout.size.spatial[1] }};
				case arg_max_min::y:
					return layout{ data_types::f32, format::bfyx, tensor{ input_layout.size.batch[0], input_layout.size.feature[0], input_layout.size.spatial[0], (int32_t)desc->top_k }};
				case arg_max_min::feature:
					return layout{ data_types::f32, format::bfyx, tensor{ input_layout.size.batch[0], (int32_t)desc->top_k, input_layout.size.spatial[0], input_layout.size.spatial[1] }};
				case arg_max_min::batch:
					return layout{ data_types::f32, format::bfyx, tensor{ (int32_t)desc->top_k, input_layout.size.feature[0], input_layout.size.spatial[0], input_layout.size.spatial[1] }};
				default:
					break;
			}
		}

		return layout{ data_types::f32, input_layout.format, tensor{ input_layout.size.batch[0], 1, (int32_t)desc->top_k, 1 } };
	}

	std::string arg_max_min_inst::to_string(arg_max_min_node const& node)
	{
		auto desc = node.get_primitive();
		auto node_info = node.desc_to_json();
		auto axis = desc->with_axis ? "true" : "false";
		auto out_type = desc->output_type ? "max" : "min";

		std::stringstream primitive_description;

		json_composite conv_info;
		conv_info.add("top_k", desc->top_k);
		conv_info.add("with axis", axis);
		if (desc->with_axis)
			conv_info.add("axis", desc->axis);
		conv_info.add("output type", out_type);
		node_info->add("arg_max_min info", conv_info);
		node_info->dump(primitive_description);

		return primitive_description.str();
	}

	arg_max_min_inst::typed_primitive_inst(network_impl& network, arg_max_min_node const& node)
		: parent(network, node)
	{
	}
}
