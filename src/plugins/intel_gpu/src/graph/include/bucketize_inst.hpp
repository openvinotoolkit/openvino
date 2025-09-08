// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/bucketize.hpp"
#include "primitive_inst.h"

namespace cldnn {

using bucketize_node = typed_program_node<bucketize>;

template <>
class typed_primitive_inst<bucketize> : public typed_primitive_inst_base<bucketize> {
public:
    using parent = typed_primitive_inst_base<bucketize>;
    using parent::parent;

    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(bucketize_node const& /*node*/, const kernel_impl_params& impl_param) {
        return forward_input0_shape<ShapeType>(impl_param);
    }

    static layout calc_output_layout(const bucketize_node& node, kernel_impl_params const& impl_param);
    static std::string to_string(const bucketize_node& node);
};

using bucketize_inst = typed_primitive_inst<bucketize>;

}  // namespace cldnn
