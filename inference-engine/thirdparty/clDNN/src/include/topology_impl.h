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
#include "api/CPP/primitive.hpp"
#include "api/CPP/input_layout.hpp"
#include "api_impl.h"
#include "refcounted_obj.h"

#include <map>

namespace cldnn
{

typedef std::map<primitive_id, std::shared_ptr<primitive>> topology_map;

struct topology_impl : public refcounted_obj<topology_impl>
{
public:
    topology_impl(const topology_map& map = topology_map())
        : _primitives(map) 
    {}

    void add(std::shared_ptr<primitive> desc)
    {
        auto id = desc->id;
        auto itr = _primitives.find(id);
        if (itr != _primitives.end())
        {
            if (itr->second != desc)
                throw std::runtime_error("different primitive with id '" + id + "' exists already");

            //adding the same primitive more than once is not an error
            return;
        }
            
        _primitives.insert({ id, desc });
    }

    const auto& at(primitive_id id) const 
    {
        try
        {
            return _primitives.at(id);
        }
        catch (...)
        {
            throw std::runtime_error("Topology doesn't contain primtive: " + id);
        }
        
    }

    void change_input_layout(const primitive_id& id, layout new_layout)
    {
        auto& inp_layout = this->at(id);
        if (inp_layout->type != input_layout::type_id())
        {
            throw std::runtime_error("Primitive: " + id + " is not input_layout.");
        }
        auto inp_lay_prim = static_cast<input_layout*>(inp_layout.get());
        inp_lay_prim->change_layout(new_layout);
    }

    const topology_map& get_primitives() const { return _primitives; }

    const std::vector<primitive_id> get_primitives_id() const 
    {
        std::vector<primitive_id> prim_ids;
        for (const auto& prim : _primitives)
            prim_ids.push_back(prim.first);
        return prim_ids;
    }

private:
    topology_map _primitives;
};
}

API_CAST(::cldnn_topology, cldnn::topology_impl)
