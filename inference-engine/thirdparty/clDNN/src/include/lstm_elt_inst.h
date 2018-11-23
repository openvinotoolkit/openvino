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
#include "api/CPP/lstm.hpp"
#include "primitive_inst.h"

namespace cldnn
{
template <>
struct typed_program_node<lstm_elt> : public typed_program_node_base<lstm_elt>
{
    using parent = typed_program_node_base<lstm_elt>;

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }
    program_node& cell() const { return get_dependency(1); }
    bool cell_term() const { return !get_primitive()->cell.empty(); }
    int32_t offset_order() const { return get_primitive()->offset_order; }
    float clip() const {
        float clip_val = get_primitive()->clip;
        if (clip_val < 0)
            throw std::range_error("Clip value < 0");
        return clip_val; 
    }
    bool input_forget() const { return get_primitive()->input_forget; }
};

using lstm_elt_node = typed_program_node<lstm_elt>;

template <>
class typed_primitive_inst<lstm_elt> : public typed_primitive_inst_base<lstm_elt>
{
    using parent = typed_primitive_inst_base<lstm_elt>;

public:
    static layout calc_output_layout(lstm_elt_node const& node);
    static std::string to_string(lstm_elt_node const& node);

public:
    typed_primitive_inst(network_impl& network, lstm_elt_node const& node);

    memory_impl& cell_memory() const { return dep_memory(1); }
    bool cell_term() const { return !argument.cell.empty(); }
    int32_t offset_order() const { return argument.offset_order; }
    float clip() const {
        float clip_val = argument.clip;
        if (clip_val < 0)
            throw std::range_error("Clip value < 0");
        return clip_val;
    }
    bool input_forget() const { return argument.input_forget; }
};

using lstm_elt_inst = typed_primitive_inst<lstm_elt>;

}
