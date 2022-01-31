// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once
#include "intel_gpu/primitives/edgpsi.hpp"
#include "primitive_inst.h"

namespace cldnn {
template <>
struct typed_program_node<edgpsi> : public typed_program_node_base<edgpsi> {
    using parent = typed_program_node_base<edgpsi>;

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }

    program_node& im_info() const { return get_dependency(0); }
    program_node& anchors() const { return get_dependency(1); }
    program_node& deltas() const { return get_dependency(2); }
    program_node& scores() const { return get_dependency(3); }

    program_node& output_roi_scores_node() const { return get_dependency(4); }
};

using edgpsi_node = typed_program_node<edgpsi>;

template <>
class typed_primitive_inst<edgpsi> : public typed_primitive_inst_base<edgpsi> {
    using parent = typed_primitive_inst_base<edgpsi>;

public:
    static layout calc_output_layout(const edgpsi_node& node);
    static std::string to_string(const edgpsi_node& node);

    typed_primitive_inst(network& network, const edgpsi_node& node)
        : parent(network, node)
    {}

    memory::ptr output_roi_scores_memory() const { return dep_memory_ptr(4); }
};

using edgpsi_inst = typed_primitive_inst<edgpsi>;

}  // namespace cldnn
