// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "openvino/core/except.hpp"
#include "snippets/target_machine.hpp"

namespace ov::intel_cpu {

// A small helper that wraps a platform-specific JIT generator and exposes
// a uniform CompiledSnippet interface. This reduces duplication across
// x64, aarch64 and riscv64 backends.
template <typename JitGeneratorT>
class CompiledSnippetCPUCommon : public ov::snippets::CompiledSnippet {
public:
    explicit CompiledSnippetCPUCommon(std::unique_ptr<JitGeneratorT> h) : h_compiled(std::move(h)) {
        OPENVINO_ASSERT(h_compiled && h_compiled->jit_ker(), "Got invalid jit generator or kernel was not compiled");
    }

    [[nodiscard]] const uint8_t* get_code() const override {
        return h_compiled->jit_ker();
    }
    [[nodiscard]] size_t get_code_size() const override {
        return h_compiled->getSize();
    }
    [[nodiscard]] bool empty() const override {
        return get_code_size() == 0;
    }

private:
    const std::unique_ptr<const JitGeneratorT> h_compiled;
};

}  // namespace ov::intel_cpu
