// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <intel_gpu/primitives/slice.hpp>
#include "primitive_inst.h"
#include <intel_gpu/runtime/error_handler.hpp>

namespace cldnn {

template <>
struct typed_program_node<slice> : public typed_program_node_base<slice> {
    using parent = typed_program_node_base<slice>;

public:
    using parent::parent;

    program_node& input(std::size_t index = 0) const { return get_dependency(index); }
};

using slice_node = typed_program_node<slice>;

template <>
class typed_primitive_inst<slice> : public typed_primitive_inst_base<slice> {
    using parent = typed_primitive_inst_base<slice>;

public:
    static layout calc_output_layout(slice_node const& node);
    static std::string to_string(slice_node const& node);

public:
    typed_primitive_inst(network& network, slice_node const& desc);
};

using slice_inst = typed_primitive_inst<slice>;

} // namespace cldnn
