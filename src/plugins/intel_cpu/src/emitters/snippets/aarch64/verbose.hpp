// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef SNIPPETS_DEBUG_CAPS
#    include <string>

#    pragma once

namespace ov::intel_cpu {
namespace aarch64 {
class jit_emitter;
}
struct jit_emitter_info_t {
    jit_emitter_info_t() = default;
    jit_emitter_info_t(const jit_emitter_info_t& rhs) = default;
    jit_emitter_info_t& operator=(const jit_emitter_info_t& rhs) = default;

    const char* c_str() const {
        return str_.c_str();
    }
    bool is_initialized() const {
        return is_initialized_;
    }

    void init(const aarch64::jit_emitter* emitter);

private:
    std::string str_;
    bool is_initialized_ = false;
};

std::string get_emitter_type_name(const aarch64::jit_emitter* emitter);

}  // namespace ov::intel_cpu

#endif  // SNIPPETS_DEBUG_CAPS
