/*
// Copyright (c) 2020 Intel Corporation
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

#include "api/extract_image_patches.hpp"
#include "primitive_inst.h"

namespace cldnn {
template <>
struct typed_program_node<extract_image_patches> : public typed_program_node_base<extract_image_patches> {
    using parent = typed_program_node_base<extract_image_patches>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
};

using extract_image_patches_node = typed_program_node<extract_image_patches>;

template <>
class typed_primitive_inst<extract_image_patches> : public typed_primitive_inst_base<extract_image_patches> {
    using parent = typed_primitive_inst_base<extract_image_patches>;

public:
    static layout calc_output_layout(extract_image_patches_node const& node);
    static std::string to_string(extract_image_patches_node const& node);

public:
    typed_primitive_inst(network_impl& network, extract_image_patches_node const& desc);
};

using extract_image_patches_inst = typed_primitive_inst<extract_image_patches>;
}  // namespace cldnn
