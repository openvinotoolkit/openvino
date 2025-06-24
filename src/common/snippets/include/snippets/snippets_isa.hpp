// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ov::snippets::isa {
#define OV_OP(a, b) using b::a;
#undef OV_OP
}  // namespace ov::snippets::isa
