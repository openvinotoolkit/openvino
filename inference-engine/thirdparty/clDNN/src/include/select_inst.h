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
#include <api/CPP/select.hpp>

#include "primitive_inst.h"

namespace cldnn
{
	template <>
	struct typed_program_node<select> : public typed_program_node_base<select>
	{
		using parent = typed_program_node_base<select>;

	public:
		using parent::parent;

        program_node& input(size_t idx = 0) const { return get_dependency(idx); }
		size_t inputs_count() const { return get_dependencies().size(); }
	};

	using select_node = typed_program_node<select>;

	template <>
	class typed_primitive_inst<select> : public typed_primitive_inst_base<select>
	{
		using parent = typed_primitive_inst_base<select>;

	public:
		static layout calc_output_layout(select_node const& node);
		static std::string to_string(select_node const& node);
		typed_primitive_inst(network_impl& network, select_node const& node);

	};

	using select_inst = typed_primitive_inst<select>;
}
