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
#include "api/CPP/crop.hpp"
#include "primitive_inst.h"

namespace cldnn
{

template <>
struct typed_program_node<crop> : public typed_program_node_base<crop>
{
private:
    using parent = typed_program_node_base<crop>;

public:
    using parent::parent;

    typed_program_node(const std::shared_ptr<crop> prim, program_impl& prog) : parent(prim, prog) { support_padding(true); }
    program_node& input() const { return get_dependency(0); }
};

using crop_node = typed_program_node<crop>;

template <>
class typed_primitive_inst<crop> : public typed_primitive_inst_base<crop>
{
    using parent = typed_primitive_inst_base<crop>;

public:
    static layout calc_output_layout(crop_node const& node);
    static std::string to_string(crop_node const& node);
    typed_primitive_inst(network_impl& network, crop_node const& node);

private:
    void on_execute() override;

    void reuse_input();
};

using crop_inst = typed_primitive_inst<crop>;
}
