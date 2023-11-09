// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "intel_gpu/runtime/layout.hpp"
#include "intel_gpu/runtime/shape_predictor.hpp"
#include "openvino/runtime/ivariable_state.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include <functional>
#include <unordered_map>

namespace ov {
namespace intel_gpu {

struct VariableStateInfo {
    VariableStateInfo(const std::string& id, const cldnn::layout& layout) : m_id(id), m_layout(layout) {}

    std::string m_id;
    cldnn::layout m_layout;
};

class VariableState : public ov::IVariableState {
public:
    VariableState(const VariableStateInfo& info, cldnn::engine& engine, cldnn::ShapePredictor& shape_predictor);
    using Ptr = std::shared_ptr<VariableState>;

    void reset() override;
    void set_state(const ov::SoPtr<ov::ITensor>& state) override;
    ov::SoPtr<ov::ITensor> get_state() const override;

    cldnn::memory::ptr get_memory() const;
    const cldnn::layout& get_layout() const;
    bool is_set() const;
    void set();
    void set_layout(const cldnn::layout& new_layout);

private:
    cldnn::layout m_layout;
    cldnn::engine& m_engine;
    cldnn::ShapePredictor& m_shape_predictor;
    bool m_is_set = false;
    cldnn::memory::ptr m_memory = nullptr;
    size_t actual_size = 0;

    void update_device_buffer();
};

using VariablesMap = std::unordered_map<std::string, VariableState::Ptr>;
using VariablesInfoMap = std::unordered_map<std::string, VariableStateInfo>;

}  // namespace intel_gpu
}  // namespace ov
