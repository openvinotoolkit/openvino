// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/plugin/x64/jit_emitter.hpp"


namespace ov {
namespace intel_cpu {

// All emitters for dynamic operations should be derived from this class.
// This should be done to distinguish between static and dynamic emitters.
class jit_snippets_dynamic_emitter {
public:
    virtual ~jit_snippets_dynamic_emitter() = default;
};

}   // namespace intel_cpu
}   // namespace ov
