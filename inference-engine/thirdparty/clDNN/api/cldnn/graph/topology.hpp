// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "cldnn/primitives/primitive.hpp"
#include "cldnn/primitives/input_layout.hpp"

#include <map>
#include <memory>
#include <vector>

namespace cldnn {

typedef std::map<primitive_id, std::shared_ptr<primitive>> topology_map;

struct topology {
public:
    using ptr = std::shared_ptr<topology>;
    explicit topology(const topology_map& map = topology_map()) : _primitives(map) {}

    /// @brief Constructs topology containing primitives provided in argument(s).
    template <class... Args>
    explicit topology(const Args&... args) : topology() {
        add<Args...>(args...);
    }


    void add_primitive(std::shared_ptr<primitive> desc) {
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

    /// @brief Adds a primitive to topology.
    template <class PType>
    void add(PType const& desc) {
        add_primitive(std::static_pointer_cast<primitive>(std::make_shared<PType>(desc)));
    }

    /// @brief Adds primitives to topology.
    template <class PType, class... Args>
    void add(PType const& desc, Args const&... args) {
        add(desc);
        add<Args...>(args...);
    }


    const std::shared_ptr<primitive>& at(primitive_id id) const {
        try {
            return _primitives.at(id);
        } catch (...) {
            throw std::runtime_error("Topology doesn't contain primtive: " + id);
        }
    }

    void change_input_layout(const primitive_id& id, const layout& new_layout) {
        if (new_layout.format < format::any || new_layout.format >= format::format_num)
            throw std::invalid_argument("Unknown format of layout.");

        if (new_layout.data_type != data_types::f16 && new_layout.data_type != data_types::f32 &&
            new_layout.data_type != data_types::i8 && new_layout.data_type != data_types::bin &&
            new_layout.data_type != data_types::u8 && new_layout.data_type != data_types::i32 &&
            new_layout.data_type != data_types::i64)
            throw std::invalid_argument("Unknown data_type of layout.");

        auto& inp_layout = this->at(id);
        if (inp_layout->type != input_layout::type_id()) {
            throw std::runtime_error("Primitive: " + id + " is not input_layout.");
        }
        auto inp_lay_prim = static_cast<input_layout*>(inp_layout.get());
        inp_lay_prim->change_layout(new_layout);
    }

    const topology_map& get_primitives() const { return _primitives; }

    const std::vector<primitive_id> get_primitives_ids() const {
        std::vector<primitive_id> prim_ids;
        for (const auto& prim : _primitives) prim_ids.push_back(prim.first);
        return prim_ids;
    }

private:
    topology_map _primitives;
};
}  // namespace cldnn
