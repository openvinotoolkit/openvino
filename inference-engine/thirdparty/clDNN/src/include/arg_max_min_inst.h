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
#pragma once
#include "api/CPP/arg_max_min.hpp"
#include "primitive_inst.h"

#include <memory>

namespace cldnn
{

	template <>
	struct typed_program_node<arg_max_min> : public typed_program_node_base<arg_max_min>
	{
		using parent = typed_program_node_base<arg_max_min>;

	public:
		typed_program_node(std::shared_ptr<primitive> prim, program_impl& prog)
			: parent(prim, prog)
		{
		}
        program_node& input() const { return get_dependency(0); }
	};

	using arg_max_min_node = typed_program_node<arg_max_min>;

	template <>
	class typed_primitive_inst<arg_max_min> : public typed_primitive_inst_base<arg_max_min>
	{
		using parent = typed_primitive_inst_base<arg_max_min>;

	public:
		static layout calc_output_layout(arg_max_min_node const& node);
		static std::string to_string(arg_max_min_node const& node);

	public:
		typed_primitive_inst(network_impl& network, arg_max_min_node const& node);
	};

	using arg_max_min_inst = typed_primitive_inst<arg_max_min>;

}
