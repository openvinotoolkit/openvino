// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/shuffle_channels.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

using shuffle_channels_node = typed_program_node<shuffle_channels>;

template <>
class typed_primitive_inst<shuffle_channels> : public typed_primitive_inst_base<shuffle_channels> {
    using parent = typed_primitive_inst_base<shuffle_channels>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(shuffle_channels_node const& /*node*/, const kernel_impl_params& impl_param) {
        return forward_input0_shape<ShapeType>(impl_param);
    }

    static layout calc_output_layout(shuffle_channels_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(shuffle_channels_node const& node);

    typed_primitive_inst(network& network, shuffle_channels_node const& desc);
};

using shuffle_channels_inst = typed_primitive_inst<shuffle_channels>;
}  // namespace cldnn
