// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "impls/registry/registry.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/implementation_desc.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

#include "intel_gpu/runtime/utils.hpp"
#include "primitive_type.h"
#include "program_node.h"
#include "layout_optimizer.h"
#include "primitive_inst.h"
#include "intel_gpu/graph/network.hpp"
#include "impls/registry/implementation_manager.hpp"

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

    in_out_fmts_t query_preferred_formats(const cldnn::program_node& node, impl_types impl_type) const  override {
        OPENVINO_ASSERT(node.type() == this, "[GPU] primitive_type_base::query_preferred_formats: primitive type mismatch");
        auto shape_type = ImplementationManager::get_shape_type(node);
        if (auto factory = implementation_map<PType>::get(node.get_preferred_impl_type(), shape_type))
            return factory->query_formats(node);
        return {};
    }

    std::shared_ptr<ImplementationManager> choose_impl(const program_node& node, shape_types requested_shape_type) const override {
        OPENVINO_ASSERT(node.type() == this, "[GPU] primitive_type_base::choose_impl: primitive type mismatch");
        for (auto& impl : get_supported_implementations(node)) {
            impl_types impl_type = impl->get_impl_type();
            if ((node.get_forced_impl_type() & impl_type) != impl_type)
                continue;

            if (impl_type == impl_types::onednn && !node.get_program().get_layout_optimizer().get_optimization_attributes().use_onednn_impls)
                continue;

            shape_types supported_shape_type = impl->get_shape_type();
            if ((requested_shape_type & supported_shape_type) != requested_shape_type && requested_shape_type != shape_types::any)
                continue;

            return impl;
        }
        return nullptr;
    }

    std::unique_ptr<primitive_impl> create_impl(const program_node& node) const override {
        OPENVINO_ASSERT(node.type() == this, "[GPU] primitive_type_base::create_impl: primitive type mismatch");
        const auto params = node.get_kernel_impl_params();
        auto impl = choose_impl(node, ImplementationManager::get_shape_type(*params));

        const auto& p = node.get_primitive();
        OPENVINO_ASSERT(impl != nullptr, "[GPU] Can't choose implementation for ", node.id(), " node (type=", p->type_string(), ")\n",
                                         "[GPU] Original name: ", p->origin_op_name, "\n",
                                         "[GPU] Original type: ", p->origin_op_type_name, "\n");
        return impl->create(node, *params);
    }

    std::shared_ptr<ImplementationManager> get_best_impl(impl_types requested_impl_type, shape_types requested_shape_type) const override {
        const auto& all_impls = get_all_implementations();
        for (auto& impl : all_impls) {
            impl_types impl_type = impl->get_impl_type();
            if ((requested_impl_type & impl_type) != impl_type)
                continue;

            shape_types supported_shape_type = impl->get_shape_type();
            if ((requested_shape_type & supported_shape_type) != requested_shape_type)
                continue;

            return impl;
        }

        return nullptr;
    }

     std::set<impl_types> get_available_impl_types(const cldnn::program_node& node) const override {
        OPENVINO_ASSERT(node.type() == this, "[GPU] primitive_type_base::get_available_impl_types: primitive type mismatch");
        auto supported_impls = get_supported_implementations(node);
        std::set<impl_types> supported_impl_types;
        for (const auto& impl : supported_impls) {
            supported_impl_types.insert(impl->get_impl_type());
        }

        return supported_impl_types;
    }

    std::shared_ptr<ImplementationManager> get(const ov::DiscreteTypeInfo& type_info) const override {
        for (auto& impl : get_all_implementations()) {
            if (impl->get_type_info() == type_info)
                return impl;
        }
        return nullptr;
    }

    std::vector<std::shared_ptr<ImplementationManager>> get_supported_implementations(const program_node& node) const override {
        OPENVINO_ASSERT(node.type() == this, "[GPU] primitive_type_base::get_supported_implementations: primitive type mismatch");
        const auto& all_impls = get_all_implementations();
        std::vector<std::shared_ptr<ImplementationManager>> supported_list;

        auto forced_impl_type = node.get_forced_impl_type();
        for (auto& impl : all_impls) {
            // Ignore impl validation if it was forced. Mainly used in unit tests
            if (forced_impl_type != impl_types::any && forced_impl_type == impl->get_impl_type()) {
                supported_list.push_back(impl);
            } else if (forced_impl_type == impl_types::any && impl->validate(node)) {
                supported_list.push_back(impl);
            }
        }

        return supported_list;
    }

    const std::vector<std::shared_ptr<ImplementationManager>>& get_all_implementations() const override {
        return ov::intel_gpu::Registry<PType>::get_implementations();
    }

    bool has_impl_for(const cldnn::program_node& node) const override {
        OPENVINO_ASSERT(node.type() == this, "[GPU] primitive_type_base::has_impl_for: primitive type mismatch");
        return has_impl_for(node, impl_types::any, shape_types::any);
    }

    bool has_impl_for(const cldnn::program_node& node, impl_types requested_impl_type) const override {
        OPENVINO_ASSERT(node.type() == this, "[GPU] primitive_type_base::has_impl_for: primitive type mismatch");
        return has_impl_for(node, requested_impl_type, shape_types::any);
    }

    bool has_impl_for(const cldnn::program_node& node, shape_types requested_shape_type) const override {
        OPENVINO_ASSERT(node.type() == this, "[GPU] primitive_type_base::has_impl_for: primitive type mismatch");
        return has_impl_for(node, impl_types::any, requested_shape_type);
    }

    bool has_impl_for(const cldnn::program_node& node, impl_types requested_impl_type, shape_types requested_shape_type) const override {
        OPENVINO_ASSERT(node.type() == this, "[GPU] primitive_type_base::has_impl_for: primitive type mismatch");
        const auto& all_impls = get_all_implementations();
        auto forced_impl_type = node.get_forced_impl_type();
        for (auto& impl : all_impls) {
            impl_types impl_type = impl->get_impl_type();
            if (requested_impl_type != impl_types::any && (requested_impl_type & impl_type) != impl_type)
                continue;

            shape_types supported_shape_type = impl->get_shape_type();
            if (requested_shape_type != shape_types::any && (requested_shape_type & supported_shape_type) != requested_shape_type)
                continue;

            if (forced_impl_type != impl_types::any) {
                // in case if we have forced impl, we don't do validation
                // and skip all other impl types here
                if (forced_impl_type == impl->get_impl_type())
                    return true;
                continue;
            } else {
                if (impl_type == impl_types::onednn && !node.get_program().get_layout_optimizer().get_optimization_attributes().use_onednn_impls)
                    continue;

                if (!impl->validate(node))
                    continue;

                return true;
            }
        }

        return false;
    }

    cldnn::layout calc_output_layout(const cldnn::program_node& node, const kernel_impl_params& impl_param) const override {
        OPENVINO_ASSERT(node.type() == this, "[GPU] primitive_type_base::calc_output_layout: primitive type mismatch");
        for (auto& t : impl_param.input_layouts) {
            GPU_DEBUG_TRACE_DETAIL << impl_param.desc->id << " input tensor: " << t.to_short_string() << std::endl;
        }
        auto res = typed_primitive_inst<PType>::calc_output_layout(node, impl_param);

        GPU_DEBUG_TRACE_DETAIL << impl_param.desc->id << " output tensor: " << res.to_short_string() << std::endl;
        return res;
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

    std::string to_string(const cldnn::program_node& node) const override {
        OPENVINO_ASSERT(node.type() == this, "[GPU] primitive_type_base::to_string: primitive type mismatch");
        return typed_primitive_inst<PType>::to_string(node);
    }
};

}  // namespace cldnn
