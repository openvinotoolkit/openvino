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
        , m_user_specified_type(user_specified_type) {}

    std::string m_id;
    cldnn::layout m_layout;
    ov::element::Type m_user_specified_type;
};

class VariableState : public ov::IVariableState {
public:
    VariableState(const VariableStateInfo& info, std::shared_ptr<RemoteContextImpl> context, std::shared_ptr<cldnn::ShapePredictor> shape_predictor);
    using Ptr = std::shared_ptr<VariableState>;

    void reset() override;
    void set_state(const ov::SoPtr<ov::ITensor>& state) override;
    ov::SoPtr<ov::ITensor> get_state() const override;

    cldnn::memory::ptr get_memory() const;
    const cldnn::layout& get_layout() const;
    bool is_set() const;
    void set();
    void set_layout(const cldnn::layout& new_layout);
    void set_memory(const cldnn::memory::ptr& new_mem, const cldnn::layout& actual_layout);
    size_t get_actual_mem_size() const {
        return actual_size;
    }

private:
    cldnn::layout m_layout;
    ov::element::Type m_user_specified_type;
    std::shared_ptr<RemoteContextImpl> m_context;
    std::shared_ptr<cldnn::ShapePredictor> m_shape_predictor;
    bool m_is_set = false;
    cldnn::memory::ptr m_memory = nullptr;
    size_t actual_size = 0;

    const cldnn::layout m_initial_layout;

    void update_device_buffer();
    ov::element::Type get_user_specified_type() const;
};

using VariablesMap = std::unordered_map<std::string, VariableState::Ptr>;
using VariablesInfoMap = std::unordered_map<std::string, VariableStateInfo>;

}  // namespace intel_gpu
}  // namespace ov
