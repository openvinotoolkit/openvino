// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "implementation_manager.hpp"
#include "program_node.h"
#include "primitive_inst.h"

namespace cldnn {

shape_types ImplementationManager::get_shape_type(const kernel_impl_params& impl_params) {
    for (auto& in_shape : impl_params.input_layouts) {
        if (in_shape.is_dynamic()) {
            return shape_types::dynamic_shape;
        }
    }
    for (auto& out_shape : impl_params.output_layouts) {
        if (out_shape.is_dynamic()) {
            return shape_types::dynamic_shape;
        }
    }

    return shape_types::static_shape;
}

shape_types ImplementationManager::get_shape_type(const program_node& node) {
    for (auto& in_layout : node.get_input_layouts()) {
        if (in_layout.is_dynamic()) {
            return shape_types::dynamic_shape;
        }
    }
    for (auto& out_layout : node.get_output_layouts()) {
        if (out_layout.is_dynamic()) {
            return shape_types::dynamic_shape;
        }
    }

    return shape_types::static_shape;
}

bool ImplementationManager::is_supported(const program_node& node, const std::set<key_type>& supported_keys, shape_types supported_shape_type) {
    auto key_in = implementation_key()(!node.get_dependencies().empty() ? node.get_input_layout(0) : layout{ov::PartialShape{}, data_types::f32, format::any});
    if (!supported_keys.empty() && supported_keys.find(key_in) == supported_keys.end())
        return false;

    // calc_output_layouts() if layout is not valid looks redundant, but some tests fail w/o it due to
    // layout invalidation on get_input_layout() call
    auto key_out = implementation_key()(node.get_outputs_count() > 0
                                        ? node.is_valid_output_layout(0) ? node.get_output_layout(0) : node.calc_output_layouts()[0]
                                        : layout{ov::PartialShape{}, data_types::f32, format::any});
    if (!supported_keys.empty() && supported_keys.find(key_out) == supported_keys.end())
        return false;

    return true;
}

std::unique_ptr<primitive_impl> ImplementationManager::create(const program_node& node, const kernel_impl_params& params) const {
    if (auto impl = create_impl(node, params)) {
        update_impl(*impl, params);
        impl->set_node_params(node);
        impl->can_share_kernels = node.get_program().get_config().get_enable_kernels_reuse();
        return impl;
    }

    return nullptr;
}

std::unique_ptr<primitive_impl> ImplementationManager::create(const kernel_impl_params& params) const {
    if (auto impl = create_impl(params)) {
        update_impl(*impl, params);
        return impl;
    }

    return nullptr;
}

void ImplementationManager::update_impl(primitive_impl& impl, const kernel_impl_params& params) const {
    impl.set_dynamic((get_shape_type() & get_shape_type(params)) == shape_types::dynamic_shape);
    impl.m_manager = this;
}

} // namespace cldnn
