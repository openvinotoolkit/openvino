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
#include "mutable_data_inst.h"
#include "primitive_type_base.h"
#include "memory_impl.h"
#include <random>
#include "error_handler.h"
#include "json_object.h"

namespace cldnn
{
primitive_type_id mutable_data_type_id()
{
    static primitive_type_base<mutable_data> instance;
    return &instance;
}

namespace {
    memory_impl::ptr attach_or_copy_data(network_impl& network, memory_impl& mem)
    {
        auto& engine = network.get_engine();
        if (mem.is_allocated_by(engine))
            return &mem;

        memory_impl::ptr result = engine.allocate_memory(mem.get_layout());
        mem_lock<char> src(mem);
        mem_lock<char> dst(result);
        std::copy(src.begin(), src.end(), dst.begin());
        return result;
    }
}

mutable_data_node::typed_program_node(const std::shared_ptr<mutable_data> dprim, program_impl& prog)
    : parent(dprim, prog), mem(api_cast(dprim->mem.get()))
{
    recalc_output_layout(false);
    fill_memory();
}

void mutable_data_node::attach_memory(memory_impl& new_mem, bool invalidate_users_if_changed)
{
    mem = &new_mem;
    recalc_output_layout(invalidate_users_if_changed);
}

void mutable_data_node::fill_memory()
{
    auto prim = get_primitive();

    if (prim->fill_type == mutable_data::filler_type::no_fill)
        return;

    auto memory = mem.get();
    auto layout = memory->get_layout();
    if (layout.data_type != data_types::f32)
        CLDNN_ERROR_MESSAGE(id(), "only f32 data types can be filled");

    switch (prim->fill_type)
    {
    case mutable_data::filler_type::zero:
        fill_memory_constant(0.f);
        break;
    case mutable_data::filler_type::one:
        fill_memory_constant(1.f);
        break;
    case mutable_data::filler_type::xavier:
        fill_memory_xavier();
        break;
    default:
        break;
    }
}

void mutable_data_node::fill_memory_xavier()
{
    auto memory = mem.get();
    auto layout = memory->get_layout();
    auto n = layout.count() / layout.size.batch[0];
    float scale = float(sqrt(3.0f / (float)n));
    std::default_random_engine generator(0);

    mem_lock<float> lock(mem);
    auto out_ptr = lock.begin();
    std::uniform_real_distribution<float> distribution(-scale, scale);
    for (uint32_t i = 0; i < (uint32_t)layout.count(); i++)
        out_ptr[i] = distribution(generator);
}

void mutable_data_node::fill_memory_constant(float value)
{
    auto memory = mem.get();
    auto layout = memory->get_layout();
    mem_lock<float> lock(mem);
    auto out_ptr = lock.begin();

    for (uint32_t i = 0; i < (uint32_t)layout.count(); i++)
        out_ptr[i] = value;
}

std::string mutable_data_inst::to_string(mutable_data_node const& node)
{
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;
    
    node_info->dump(primitive_description);
    return primitive_description.str();
}

mutable_data_inst::typed_primitive_inst(network_impl& network, mutable_data_node const& node)
    : parent(network, node, *attach_or_copy_data(network, node.get_attached_memory()))
{
}

}
