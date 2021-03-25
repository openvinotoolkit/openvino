// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "api/primitive.hpp"
#include "api/input_layout.hpp"
#include "refcounted_obj.h"

#include <map>
#include <memory>
#include <vector>

namespace cldnn {

typedef std::map<primitive_id, std::shared_ptr<primitive>> topology_map;

struct topology_impl : public refcounted_obj<topology_impl> {
public:
    explicit topology_impl(const topology_map& map = topology_map()) : _primitives(map) {}

    void add(std::shared_ptr<primitive> desc) {
        auto id = desc->id;
        auto itr = _primitives.find(id);
        if (itr != _primitives.end()) {
            if (itr->second != desc)
                throw std::runtime_error("different primitive with id '" + id + "' exists already");

            // adding the same primitive more than once is not an error
            return;
        }

        _primitives.insert({id, desc});
    }

    const std::shared_ptr<primitive>& at(primitive_id id) const {
        try {
            return _primitives.at(id);
        } catch (...) {
            throw std::runtime_error("Topology doesn't contain primtive: " + id);
        }
    }

    void change_input_layout(const primitive_id& id, const layout& new_layout) {
        auto& inp_layout = this->at(id);
        if (inp_layout->type != input_layout::type_id()) {
            throw std::runtime_error("Primitive: " + id + " is not input_layout.");
        }
        auto inp_lay_prim = static_cast<input_layout*>(inp_layout.get());
        inp_lay_prim->change_layout(new_layout);
    }

    const topology_map& get_primitives() const { return _primitives; }

    const std::vector<primitive_id> get_primitives_id() const {
        std::vector<primitive_id> prim_ids;
        for (const auto& prim : _primitives) prim_ids.push_back(prim.first);
        return prim_ids;
    }

private:
    topology_map _primitives;
};
}  // namespace cldnn
