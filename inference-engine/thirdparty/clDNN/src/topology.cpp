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

}  // namespace cldnn
