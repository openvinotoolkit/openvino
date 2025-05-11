// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <intel_gpu/primitives/roi_align.hpp>
#include "primitive_inst.h"

namespace cldnn {

template <>
struct typed_program_node<roi_align> : public typed_program_node_base<roi_align> {
    using parent = typed_program_node_base<roi_align>;
    typed_program_node(const std::shared_ptr<roi_align> prim, program& prog) : parent(prim, prog) {}

public:
    using parent::parent;
    program_node& input() const { return get_dependency(0); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using roi_align_node = typed_program_node<roi_align>;

template <>
class typed_primitive_inst<roi_align> : public typed_primitive_inst_base<roi_align> {
    using parent = typed_primitive_inst_base<roi_align>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(roi_align_node const& node, kernel_impl_params const& impl_param);
    static layout calc_output_layout(roi_align_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(roi_align_node const& node);

public:
    typed_primitive_inst(network& network, roi_align_node const& desc);
    memory::ptr rois_memory() const { return dep_memory_ptr(1); }
    memory::ptr batches_memory() const { return dep_memory_ptr(2); }
};

using roi_align_inst = typed_primitive_inst<roi_align>;

} // namespace cldnn
