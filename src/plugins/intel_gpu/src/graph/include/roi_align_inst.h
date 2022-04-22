// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <intel_gpu/primitives/roi_align.hpp>
#include "primitive_inst.h"
#include <intel_gpu/runtime/error_handler.hpp>

namespace cldnn {

template <>
struct typed_program_node<roi_align> : public typed_program_node_base<roi_align> {
    using parent = typed_program_node_base<roi_align>;

public:
    using parent::parent;

    program_node& input(std::size_t index = 0) const { return *get_dependency(index).first; }
};

using roi_align_node = typed_program_node<roi_align>;

template <>
class typed_primitive_inst<roi_align> : public typed_primitive_inst_base<roi_align> {
    using parent = typed_primitive_inst_base<roi_align>;

public:
    static layout calc_output_layout(roi_align_node const& node);
    static std::string to_string(roi_align_node const& node);

public:
    typed_primitive_inst(network& network, roi_align_node const& desc);
    memory::ptr rois_memory() const { return dep_memory_ptr(1); }
    memory::ptr batches_memory() const { return dep_memory_ptr(2); }
};

using roi_align_inst = typed_primitive_inst<roi_align>;

} // namespace cldnn
