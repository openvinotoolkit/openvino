// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "api/pyramid_roi_align.hpp"
#include "primitive_inst.h"

#include <memory>
#include <string>

namespace cldnn {
template <>
struct typed_program_node<pyramid_roi_align> : public typed_program_node_base<pyramid_roi_align> {
    using parent = typed_program_node_base<pyramid_roi_align>;

public:
    typed_program_node(std::shared_ptr<primitive> prim, program_impl& prog) : parent(prim, prog) {}

    program_node& input() const { return get_dependency(0); }
    // program_node& boxes() const { return get_dependency(0); }
    program_node& P2() const { return get_dependency(1); }
    program_node& P3() const { return get_dependency(2); }
    program_node& P4() const { return get_dependency(3); }
    program_node& P5() const { return get_dependency(4); }
};

using pyramid_roi_align_node = typed_program_node<pyramid_roi_align>;

template <>
class typed_primitive_inst<pyramid_roi_align> : public typed_primitive_inst_base<pyramid_roi_align> {
    using parent = typed_primitive_inst_base<pyramid_roi_align>;

public:
    static layout calc_output_layout(pyramid_roi_align_node const& node);
    static std::string to_string(pyramid_roi_align_node const& node);
    typed_primitive_inst(network_impl& network, pyramid_roi_align_node const& node);

    memory_impl& input() const { return dep_memory(0); }
    memory_impl& P2() const { return dep_memory(1); }
    memory_impl& P3() const { return dep_memory(2); }
    memory_impl& P4() const { return dep_memory(3); }
    memory_impl& P5() const { return dep_memory(4); }
};

using pyramid_roi_align_inst = typed_primitive_inst<pyramid_roi_align>;
}  // namespace cldnn
