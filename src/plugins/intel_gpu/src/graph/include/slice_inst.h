// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <intel_gpu/primitives/slice.hpp>
#include "primitive_inst.h"
#include <intel_gpu/runtime/error_handler.hpp>

namespace cldnn {

using slice_node = typed_program_node<slice>;

template <>
class typed_primitive_inst<slice> : public typed_primitive_inst_base<slice> {
    using parent = typed_primitive_inst_base<slice>;
    using parent::parent;

public:
    static layout calc_output_layout(slice_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(slice_node const& node);

public:
    typed_primitive_inst(network& network, slice_node const& desc);
};

using slice_inst = typed_primitive_inst<slice>;

} // namespace cldnn
