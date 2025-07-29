// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <regex>

#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/ivariable_state.hpp"

namespace ov {
namespace npuw {

class VariableState final : public ov::IVariableState {
public:
    explicit VariableState(const std::string& name, const ov::SoPtr<ov::ITensor>& tensor) : ov::IVariableState(name) {
        m_state = tensor;
        clear_state_updated();
    }

    virtual void set_state(const ov::SoPtr<ov::ITensor>& newState) override {
        m_state = newState;
        m_state_updapted = true;
    }

    virtual void reset() override {
        std::memset(m_state->data(), 0, m_state->get_byte_size());
    }

    ~VariableState() override = default;

    bool is_state_updated() const {
        return m_state_updapted;
    }

    void clear_state_updated() {
        m_state_updapted = false;
    }

private:
    bool m_state_updapted;

};

struct LoRANames {
    static constexpr const char* MatMul_A = "MatMul\\.A";
    static constexpr const char* MatMul_B = "MatMul\\.B";
    static constexpr const char* MatMul_alpha = "MatMul\\.alpha";
};

static bool matchStringWithLoRAPattern(const std::string& input, const std::string& pattern_suffix) {
    std::string pattern = "^lora_state.*" + pattern_suffix + "$";
    std::regex regex_pattern(pattern);

    return std::regex_match(input, regex_pattern);
};

static bool matchLoRAMatMulAString(const std::string& input) {
    return matchStringWithLoRAPattern(input, LoRANames::MatMul_A);
};

static bool matchLoRAMatMulBString(const std::string& input) {
    return matchStringWithLoRAPattern(input, LoRANames::MatMul_B);
};

static bool matchLoRAMatMulAlphaString(const std::string& input) {
    return matchStringWithLoRAPattern(input, LoRANames::MatMul_alpha);
};

}  // namespace npuw
}  // namespace ov
