// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "variable_state.hpp"

namespace ov {
namespace npuw {

struct LoRANames {
    static constexpr const char* MatMul_A = "MatMul\\.A";
    static constexpr const char* MatMul_B = "MatMul\\.B";
    static constexpr const char* MatMul_alpha = "MatMul\\.alpha";
};

}  // namespace npuw
}  // namespace ov
