// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/extract_image_patches.hpp"
#include "primitive_inst.h"

namespace cldnn {

using extract_image_patches_node = typed_program_node<extract_image_patches>;

template <>
class typed_primitive_inst<extract_image_patches> : public typed_primitive_inst_base<extract_image_patches> {
    using parent = typed_primitive_inst_base<extract_image_patches>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(extract_image_patches_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(extract_image_patches_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(extract_image_patches_node const& node);

    typed_primitive_inst(network& network, extract_image_patches_node const& desc);
};

using extract_image_patches_inst = typed_primitive_inst<extract_image_patches>;
}  // namespace cldnn
