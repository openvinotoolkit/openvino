// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "api/topology.hpp"
#include "topology_impl.h"
#include <vector>
#include <memory>

namespace cldnn {

topology::topology() : _impl(new topology_impl()) {}

const std::vector<primitive_id> topology::get_primitive_ids() const {
    return _impl->get_primitives_id();
}

void topology::change_input_layout(primitive_id id, const layout& new_layout) {
    if (new_layout.format < format::any || new_layout.format >= format::format_num)
        throw std::invalid_argument("Unknown format of layout.");

    if (new_layout.data_type != data_types::f16 && new_layout.data_type != data_types::f32 &&
        new_layout.data_type != data_types::i8 && new_layout.data_type != data_types::bin &&
        new_layout.data_type != data_types::u8 && new_layout.data_type != data_types::i32 &&
        new_layout.data_type != data_types::i64)
        throw std::invalid_argument("Unknown data_type of layout.");

    _impl->change_input_layout(id, new_layout);
}

void topology::add_primitive(std::shared_ptr<primitive> desc) {
    _impl->add(desc);
}

void topology::retain() {
    _impl->add_ref();
}

void topology::release() {
    _impl->release();
}

const std::shared_ptr<primitive>& topology::at(const primitive_id& id) const {
    return _impl->at(id);
}

}  // namespace cldnn
