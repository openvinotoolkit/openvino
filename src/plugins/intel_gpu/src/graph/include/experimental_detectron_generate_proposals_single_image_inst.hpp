// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/experimental_detectron_generate_proposals_single_image.hpp"
#include "primitive_inst.h"

namespace cldnn {
template <>
struct typed_program_node<experimental_detectron_generate_proposals_single_image>
        : public typed_program_node_base<experimental_detectron_generate_proposals_single_image> {
    using parent = typed_program_node_base<experimental_detectron_generate_proposals_single_image>;

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }

    program_node& anchors() const { return get_dependency(1); }
    program_node& deltas() const { return get_dependency(2); }
    program_node& scores() const { return get_dependency(3); }

    program_node& output_roi_scores_node() const { return get_dependency(4); }
};

using experimental_detectron_generate_proposals_single_image_node = typed_program_node<experimental_detectron_generate_proposals_single_image>;

template <>
class typed_primitive_inst<experimental_detectron_generate_proposals_single_image>
        : public typed_primitive_inst_base<experimental_detectron_generate_proposals_single_image> {
    using parent = typed_primitive_inst_base<experimental_detectron_generate_proposals_single_image>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(experimental_detectron_generate_proposals_single_image_node const&, const kernel_impl_params& impl_param);
    static layout calc_output_layout(const experimental_detectron_generate_proposals_single_image_node& node, kernel_impl_params const& impl_param);
    static std::string to_string(const experimental_detectron_generate_proposals_single_image_node& node);

    typed_primitive_inst(network& network, const experimental_detectron_generate_proposals_single_image_node& node)
        : parent(network, node)
    {}

    memory::ptr output_roi_scores_memory() const { return dep_memory_ptr(4); }
};

using experimental_detectron_generate_proposals_single_image_inst = typed_primitive_inst<experimental_detectron_generate_proposals_single_image>;

}  // namespace cldnn
