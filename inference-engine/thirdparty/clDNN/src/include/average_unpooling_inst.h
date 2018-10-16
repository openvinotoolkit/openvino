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
#pragma once
#include "api/CPP/average_unpooling.hpp"
#include "primitive_inst.h"

namespace cldnn
{

    template <>
    struct typed_program_node<average_unpooling> : public typed_program_node_base<average_unpooling>
    {
        using parent = typed_program_node_base<average_unpooling>;

    public:
        using parent::parent;
        decltype(auto) input() const { return get_dependency(0); }
    };

    using average_unpooling_node = typed_program_node<average_unpooling>;

    template <>
    class typed_primitive_inst<average_unpooling> : public typed_primitive_inst_base<average_unpooling>
    {
        using parent = typed_primitive_inst_base<average_unpooling>;

    public:
        typed_primitive_inst(network_impl& network, average_unpooling_node const& desc);
        static layout calc_output_layout(average_unpooling_node const& node);
        static std::string to_string(average_unpooling_node const& node);
    };

    using average_unpooling_inst = typed_primitive_inst<average_unpooling>;

}
