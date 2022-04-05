// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "intel_gpu/primitives/extract_image_patches.hpp"
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
    typed_primitive_inst(network& network, extract_image_patches_node const& desc);
};

using extract_image_patches_inst = typed_primitive_inst<extract_image_patches>;
}  // namespace cldnn
