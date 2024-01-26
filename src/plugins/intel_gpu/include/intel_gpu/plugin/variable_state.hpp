// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/ivariable_state.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "intel_gpu/runtime/shape_predictor.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include <functional>
#include <unordered_map>

namespace ov {
namespace intel_gpu {
class RemoteContextImpl;

struct VariableStateInfo {
    VariableStateInfo(const std::string& id, const cldnn::layout& layout, ov::element::Type_t user_specified_type = ov::element::undefined)
        : m_id(id)
        , m_layout(layout)
        , m_user_specified_type(user_specified_type)
        , m_primitives() {}

    std::string m_id;
    cldnn::layout m_layout;
    ov::element::Type m_user_specified_type;
    std::set<const cldnn::primitive*> m_primitives;
};

class GPUVariableState : public ov::IVariableState {
public:
    GPUVariableState(const std::string& id, std::shared_ptr<RemoteContextImpl> context) : ov::IVariableState(id), m_context(context) {}
    virtual cldnn::memory::ptr get_memory() const = 0;
    virtual const cldnn::layout& get_layout() const = 0;
    virtual bool is_set() const = 0;
    virtual void set_layout(const cldnn::layout& new_layout) = 0;
    virtual void set_memory(const cldnn::memory::ptr& new_mem, const cldnn::layout& actual_layout) = 0;
    virtual size_t get_actual_mem_size() const = 0;

    void set() { m_is_set = true; }

protected:
    bool m_is_set = false;
    std::shared_ptr<RemoteContextImpl> m_context;
};

class VariableState : public GPUVariableState {
public:
    VariableState(const VariableStateInfo& info, std::shared_ptr<RemoteContextImpl> context, ShapePredictor::Ptr shape_predictor);
    using Ptr = std::shared_ptr<VariableState>;

    void reset() override;
    void set_state(const ov::SoPtr<ov::ITensor>& state) override;
    ov::SoPtr<ov::ITensor> get_state() const override;

    cldnn::memory::ptr get_memory() const override;
    const cldnn::layout& get_layout() const override;
    bool is_set() const override;

    void set_layout(const cldnn::layout& new_layout) override;
    void set_memory(const cldnn::memory::ptr& new_mem, const cldnn::layout& actual_layout) override;
    size_t get_actual_mem_size() const override {
        return actual_size;
    }

    const cldnn::layout& get_initial_layout() const {
        return m_initial_layout;
    }

protected:
    cldnn::layout m_layout;
    ov::element::Type m_user_specified_type;
    std::shared_ptr<cldnn::ShapePredictor> m_shape_predictor;

    cldnn::memory::ptr m_memory = nullptr;
    size_t actual_size = 0;

    const cldnn::layout m_initial_layout;

    void update_device_buffer();
    ov::element::Type get_user_specified_type() const;
};

using VariablesMap = std::unordered_map<std::string, std::shared_ptr<GPUVariableState>>;
using VariablesInfoMap = std::unordered_map<std::string, VariableStateInfo>;

}  // namespace intel_gpu
}  // namespace ov
