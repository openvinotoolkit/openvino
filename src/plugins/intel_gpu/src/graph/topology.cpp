// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/graph/topology.hpp"
#include <vector>
#include <memory>

namespace cldnn {

void topology::add_primitive(std::shared_ptr<primitive> desc) {
    auto id = desc->id;
    auto itr = _primitives.find(id);
    if (itr != _primitives.end()) {
        OPENVINO_ASSERT(itr->second == desc, "[GPU] Different primitive with id '" + id + "' exists already");

        // adding the same primitive more than once is not an error
        return;
    }

    _primitives.insert({id, desc});
}

const std::shared_ptr<primitive>& topology::at(primitive_id id) const {
    try {
        return _primitives.at(id);
    } catch (...) {
        throw std::runtime_error("Topology doesn't contain primtive: " + id);
    }
}

void topology::change_input_layout(const primitive_id& id, const layout& new_layout) {
    if (new_layout.format < format::any || new_layout.format >= format::format_num)
        throw std::invalid_argument("Unknown format of layout.");

    if (new_layout.data_type != data_types::f16 && new_layout.data_type != data_types::f32 &&
        new_layout.data_type != data_types::i8 && new_layout.data_type != data_types::u1 &&
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

const std::vector<primitive_id> topology::get_primitives_ids() const {
    std::vector<primitive_id> prim_ids;
    for (const auto& prim : _primitives) prim_ids.push_back(prim.first);
    return prim_ids;
}

}  // namespace cldnn
