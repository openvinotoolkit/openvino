// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef SNIPPETS_DEBUG_CAPS
#    include <string>

#    pragma once

#    include "emitters/snippets/common/verbose_utils.hpp"

namespace ov::intel_cpu::aarch64 {

class jit_emitter;

struct jit_emitter_info_t : public snippets_common::jit_emitter_info_base {
    void init(const void* emitter) override;
};

}  // namespace ov::intel_cpu::aarch64

#endif  // SNIPPETS_DEBUG_CAPS
