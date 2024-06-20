// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/generate_proposals.hpp"
#include "primitive_inst.h"

namespace cldnn {
template <>
struct typed_program_node<generate_proposals>
        : public typed_program_node_base<generate_proposals> {
    using parent = typed_program_node_base<generate_proposals>;

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }

    program_node& anchors() const { return get_dependency(1); }
    program_node& deltas() const { return get_dependency(2); }
    program_node& scores() const { return get_dependency(3); }

    program_node& output_rois_scores_node() const { return get_dependency(4); }
    program_node& output_rois_nums_node() const { return get_dependency(5); }
};

using generate_proposals_node = typed_program_node<generate_proposals>;

template <>
class typed_primitive_inst<generate_proposals>
        : public typed_primitive_inst_base<generate_proposals> {
    using parent = typed_primitive_inst_base<generate_proposals>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(generate_proposals_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(const generate_proposals_node& node, kernel_impl_params const& impl_param);
    static std::string to_string(const generate_proposals_node& node);

    typed_primitive_inst(network& network, const generate_proposals_node& node)
            : parent(network, node)
    {}

    memory::ptr output_rois_scores_memory() const { return dep_memory_ptr(4); }
    memory::ptr output_rois_nums_memory() const { return dep_memory_ptr(5); }
};

using generate_proposals_inst = typed_primitive_inst<generate_proposals>;
}  // namespace cldnn
