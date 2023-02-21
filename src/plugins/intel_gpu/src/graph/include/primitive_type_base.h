// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

#include "meta_utils.h"
#include "primitive_type.h"
#include "program_node.h"
#include "primitive_inst.h"
#include "intel_gpu/graph/network.hpp"
#include "impls/implementation_map.hpp"

#include <memory>
#include <string>

namespace cldnn {
template <class PType>
struct primitive_type_base : primitive_type {
    std::shared_ptr<cldnn::program_node> create_node(program& program,
                                                     const std::shared_ptr<primitive> prim) const override {
        OPENVINO_ASSERT(prim->type == this, "[GPU] primitive_type_base::create_node: primitive type mismatch");
        return std::make_shared<typed_program_node<PType>>(std::static_pointer_cast<PType>(prim), program);
    }

    std::shared_ptr<cldnn::primitive_inst> create_instance(network& network, const cldnn::program_node& node) const override {
        OPENVINO_ASSERT(node.type() == this, "[GPU] primitive_type_base::create_instance: primitive type mismatch");
        return std::make_shared<typed_primitive_inst<PType>>(network, node);
    }

    std::shared_ptr<cldnn::primitive_inst> create_instance(network& network) const override {
        return std::make_shared<typed_primitive_inst<PType>>(network);
    }

    // TODO: Should we get rid of engine type in impl map? Or we must pass internal build engine to get real ocl type?
    std::unique_ptr<primitive_impl> choose_impl(const cldnn::program_node& node) const override {
        return choose_impl(node, *node.get_kernel_impl_params());
    }

    std::unique_ptr<primitive_impl> choose_impl(const cldnn::program_node& node, const kernel_impl_params& runtime_params) const override {
        OPENVINO_ASSERT(node.type() == this, "[GPU] primitive_type_base::choose_impl: primitive type mismatch");
        auto factory = implementation_map<PType>::get(runtime_params, node.get_preferred_impl_type(), get_shape_type(runtime_params));
        auto impl = factory(node, runtime_params);
        impl->set_dynamic(get_shape_type(runtime_params) == shape_types::dynamic_shape);
        return impl;
    }

    bool does_an_implementation_exist(const cldnn::program_node& node) const override {
        return does_an_implementation_exist(node, *node.get_kernel_impl_params());
    }

    bool does_an_implementation_exist(const cldnn::program_node& node, const kernel_impl_params& impl_param) const override {
        OPENVINO_ASSERT(node.type() == this, "[GPU] primitive_type_base::does_an_implementation_exist: primitive type mismatch");

        return implementation_map<PType>::check(impl_param, node.get_preferred_impl_type(), shape_types::static_shape);
    }

    bool does_possible_implementation_exist(const cldnn::program_node& node) const override {
        return does_possible_implementation_exist(node, *node.get_kernel_impl_params());
    }

    bool does_possible_implementation_exist(const cldnn::program_node& node, const kernel_impl_params& impl_param) const override {
        OPENVINO_ASSERT(node.type() == this, "[GPU] primitive_type_base::does_possible_implementation_exist: primitive type mismatch");
        return implementation_map<PType>::check_io_eq(impl_param, node.get_preferred_impl_type(), shape_types::static_shape);
    }

    bool does_dynamic_implementation_exist(const cldnn::program_node& node) const override {
        return does_dynamic_implementation_exist(node, *node.get_kernel_impl_params());
    }

    bool does_dynamic_implementation_exist(const cldnn::program_node& node, const kernel_impl_params& impl_param) const override {
        OPENVINO_ASSERT(node.type() == this, "[GPU] primitive_type_base::does_possible_implementation_exist: primitive type mismatch");
        return implementation_map<PType>::check(impl_param, node.get_preferred_impl_type(), shape_types::dynamic_shape);
    }

    cldnn::layout calc_output_layout(const cldnn::program_node& node, const kernel_impl_params& impl_param) const override {
        OPENVINO_ASSERT(node.type() == this, "[GPU] primitive_type_base::calc_output_layout: primitive type mismatch");
        return typed_primitive_inst<PType>::calc_output_layout(node, impl_param);
    }

    std::vector<cldnn::layout> calc_output_layouts(const cldnn::program_node& node, const kernel_impl_params& impl_param) const override {
        OPENVINO_ASSERT(node.type() == this, "primitive_type_base::calc_output_layouts: primitive type mismatch");

        for (auto& t : impl_param.input_layouts) {
            GPU_DEBUG_TRACE_DETAIL << impl_param.desc->id << " input tensor: " << t.to_short_string() << std::endl;
        }

        auto res = typed_primitive_inst<PType>::template calc_output_layouts<ov::PartialShape>(node, impl_param);

        for (auto& t : res) {
            GPU_DEBUG_TRACE_DETAIL << impl_param.desc->id << " output tensor: " << t.to_short_string() << std::endl;
        }

        return res;
    }
    kernel_impl_params get_fake_aligned_params(kernel_impl_params const& orig_impl_param) const override {
        return typed_primitive_inst<PType>::get_fake_aligned_params(orig_impl_param);
    }
    std::vector<size_t> extend_input_shape_to_6d(kernel_impl_params const& orig_impl_param, int32_t input_idx) const override {
        return typed_primitive_inst<PType>::extend_input_shape_to_6d(orig_impl_param, input_idx);
    }
    std::vector<size_t> extend_output_shape_to_6d(kernel_impl_params const& orig_impl_param, int32_t output_idx) const override {
        return typed_primitive_inst<PType>::extend_output_shape_to_6d(orig_impl_param, output_idx);
    }
    std::string to_string(const cldnn::program_node& node) const override {
        OPENVINO_ASSERT(node.type() == this, "[GPU] primitive_type_base::to_string: primitive type mismatch");
        return typed_primitive_inst<PType>::to_string(node);
    }

    shape_types get_shape_type(const kernel_impl_params& impl_params) const {
        for (auto& in_shape : impl_params.input_layouts) {
            if (in_shape.is_dynamic()) {
                return shape_types::dynamic_shape;
            }
        }
        if (impl_params.get_output_layout().is_dynamic())
            return shape_types::dynamic_shape;

        return shape_types::static_shape;
    }
};

}  // namespace cldnn
