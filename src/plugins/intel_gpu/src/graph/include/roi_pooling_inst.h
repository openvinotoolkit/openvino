// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/roi_pooling.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {
template <>
struct typed_program_node<roi_pooling> : public typed_program_node_base<roi_pooling> {
    using parent = typed_program_node_base<roi_pooling>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using roi_pooling_node = typed_program_node<roi_pooling>;

template <>
class typed_primitive_inst<roi_pooling> : public typed_primitive_inst_base<roi_pooling> {
    using parent = typed_primitive_inst_base<roi_pooling>;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(roi_pooling_node const& node, kernel_impl_params const& impl_param);
    static layout calc_output_layout(roi_pooling_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(roi_pooling_node const& node);

public:
    using parent::parent;

    memory::ptr rois_memory() const { return dep_memory_ptr(1); }
    memory::ptr trans_memory() const { return dep_memory_ptr(2); }
};

using roi_pooling_inst = typed_primitive_inst<roi_pooling>;

}  // namespace cldnn
