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

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "api/CPP/reshape.hpp"
#include "primitive_inst.h"

namespace cldnn
{

template <>
struct typed_program_node<reshape> : public typed_program_node_base<reshape>
{
    using parent = typed_program_node_base<reshape>;
    typed_program_node(const std::shared_ptr<reshape> prim, program_impl& prog) : parent(prim, prog) { support_padding(true); }

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }

    bool is_in_place() const
    {
        if (this->is_output() || this->get_fused_activation_func() != activation_none)
            return false;
        return (!this->get_output_layout().data_padding && !input().get_output_layout(false).data_padding);
    }
};

using reshape_node = typed_program_node<reshape>;

template <>
class typed_primitive_inst<reshape> : public typed_primitive_inst_base<reshape>
{
    using parent = typed_primitive_inst_base<reshape>;

public:
    static layout calc_output_layout(reshape_node const& node);
    static std::string to_string(reshape_node const& node);

public:
    typed_primitive_inst(network_impl& network, reshape_node const& node);

private:
    void on_execute() override;

    void reuse_input();
};

using reshape_inst = typed_primitive_inst<reshape>;

}

