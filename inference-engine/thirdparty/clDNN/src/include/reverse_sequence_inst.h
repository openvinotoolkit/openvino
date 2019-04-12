/*
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
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "api/CPP/reverse_sequence.hpp"
#include "primitive_inst.h"

namespace  cldnn
{
    template <>
    struct typed_program_node<reverse_sequence> : public typed_program_node_base<reverse_sequence>
    {
        using parent = typed_program_node_base<reverse_sequence>;

    public:
        using parent::parent;

        program_node& input(size_t index = 0) const { return get_dependency(index); }
    };

    using reverse_sequence_node = typed_program_node<reverse_sequence>;

    template <>
    class typed_primitive_inst<reverse_sequence> : public typed_primitive_inst_base<reverse_sequence>
    {
        using parent = typed_primitive_inst_base<reverse_sequence>;

    public:
        static layout calc_output_layout(reverse_sequence_node const& node);
        static std::string to_string(reverse_sequence_node const& node);

    public:
        typed_primitive_inst(network_impl& network, reverse_sequence_node const& desc);
    };

    using reverse_sequence_inst = typed_primitive_inst<reverse_sequence>;
}
