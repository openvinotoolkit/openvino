// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/implementation_desc.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "openvino/core/except.hpp"

#include <functional>
#include <memory>
#include <tuple>

namespace cldnn {

using in_out_fmts_t = std::pair<std::vector<format::type>, std::vector<format::type>>;

struct primitive_impl;

struct program_node;
template <class PType>
struct typed_program_node;

using key_type = std::tuple<data_types, format::type>;
struct implementation_key {
    key_type operator()(const layout& proposed_layout) {
        return std::make_tuple(proposed_layout.data_type, proposed_layout.format);
    }
};

#define OV_GPU_PRIMITIVE_IMPL(TYPE_NAME)                                                  \
    _OPENVINO_HIDDEN_METHOD static const ::ov::DiscreteTypeInfo& get_type_info_static() { \
        static ::ov::DiscreteTypeInfo type_info_static{TYPE_NAME};                   \
        type_info_static.hash();                                                          \
        return type_info_static;                                                          \
    }                                                                                     \
    const ::ov::DiscreteTypeInfo& get_type_info() const override { return get_type_info_static(); }

using ValidateFunc = std::function<bool(const program_node& node)>;
struct ImplementationManager {
public:
    std::unique_ptr<primitive_impl> create(const program_node& node, const kernel_impl_params& params) const;
    std::unique_ptr<primitive_impl> create(const kernel_impl_params& params) const;
    bool validate(const program_node& node) const {
        if (!validate_impl(node))
            return false;
        if (m_vf) {
            return m_vf(node);
        }

        return true;
    }

    virtual const ov::DiscreteTypeInfo& get_type_info() const = 0;
    virtual std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const = 0;
    virtual std::unique_ptr<primitive_impl> create_impl(const kernel_impl_params& params) const { OPENVINO_NOT_IMPLEMENTED; }
    virtual bool validate_impl(const program_node& node) const { return true; }
    virtual bool support_shapes(const kernel_impl_params& param) const { return true; }
    virtual in_out_fmts_t query_formats(const program_node& node) const { OPENVINO_NOT_IMPLEMENTED; }

    ImplementationManager(impl_types impl_type, shape_types shape_type, ValidateFunc vf = nullptr)
        : m_impl_type(impl_type)
        , m_shape_type(shape_type)
        , m_vf(vf) {}
    virtual ~ImplementationManager() = default;

    static shape_types get_shape_type(const program_node& node);
    static shape_types get_shape_type(const kernel_impl_params& params);

    impl_types get_impl_type() const { return m_impl_type; }
    shape_types get_shape_type() const { return m_shape_type; }

protected:
    static bool is_supported(const program_node& node, const std::set<key_type>& supported_keys, shape_types shape_type);
    impl_types m_impl_type;
    shape_types m_shape_type;
    ValidateFunc m_vf;

    void update_impl(primitive_impl& impl, const kernel_impl_params& params) const;
};

template <typename primitive_kind>
struct ImplementationManagerLegacy : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL(typeid(primitive_kind).name())

    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override {
        if (m_factory) {
            return m_factory(static_cast<const typed_program_node<primitive_kind>&>(node), params);
        }

        OPENVINO_NOT_IMPLEMENTED;
    }
    bool validate_impl(const program_node& node) const override {
        return ImplementationManager::is_supported(node, m_keys, m_shape_type);
    }

    bool support_shapes(const kernel_impl_params& params) const override {
        return true;
    }

    in_out_fmts_t query_formats(const program_node& node) const override {
        return {};
    }

    using simple_factory_type = std::function<std::unique_ptr<primitive_impl>(const typed_program_node<primitive_kind>&, const kernel_impl_params&)>;
    ImplementationManagerLegacy(simple_factory_type factory, impl_types impl_type, shape_types shape_type, std::set<key_type> keys)
        : ImplementationManager(impl_type, shape_type, nullptr)
        , m_factory(factory)
        , m_keys(keys) {
            add_keys_with_any_layout();
        }

    ImplementationManagerLegacy(const ImplementationManagerLegacy* other, ValidateFunc vf)
        : ImplementationManager(other->m_impl_type, other->m_shape_type, vf)
        , m_factory(other->m_factory)
        , m_keys(other->m_keys) {
            add_keys_with_any_layout();
        }

    ImplementationManagerLegacy() = default;

private:
    simple_factory_type m_factory;
    std::set<key_type> m_keys;

    void add_keys_with_any_layout() {
        std::set<data_types> supported_types;
        for (auto& key : m_keys) {
            supported_types.insert(std::get<0>(key));
        }
        for (auto& dt : supported_types) {
            m_keys.insert({dt, format::any});
        }
    }
};

}  // namespace cldnn
