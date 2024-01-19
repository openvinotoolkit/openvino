// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef SNIPPETS_DEBUG_CAPS

#pragma once

#include <sstream>

namespace ov {
namespace intel_cpu {
class jit_emitter;
struct jit_emitter_info_t {
    jit_emitter_info_t() = default;
    jit_emitter_info_t(const jit_emitter_info_t &rhs)
        : str_(rhs.str_), is_initialized_(rhs.is_initialized_) {}
    jit_emitter_info_t &operator=(const jit_emitter_info_t &rhs) {
        is_initialized_ = rhs.is_initialized_;
        str_ = rhs.str_;
        return *this;
    }

    const char *c_str() const { return str_.c_str(); }
    bool is_initialized() const { return is_initialized_; }

    void init(const jit_emitter *emitter);

private:
    std::string str_;
    bool is_initialized_ = false;
};

std::string get_emitter_type_name(const jit_emitter* emitter);

}   // namespace intel_cpu
}   // namespace ov

#endif