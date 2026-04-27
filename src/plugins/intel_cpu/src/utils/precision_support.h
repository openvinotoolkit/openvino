// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu {

enum class Aarch64Int8Isa {
    scalar_reference,
    neon,
    neon_dotprod,
    neon_i8mm,
    sve,
    sve_i8mm,
    sve2_i8mm
};

bool hasHardwareSupport(const ov::element::Type& precision);
ov::element::Type defaultFloatPrecision();
bool hasIntDotProductSupport();
bool hasInt8MMSupport();
bool hasSVESupport();
bool hasSVE2Support();
Aarch64Int8Isa resolveAarch64Int8Isa(bool has_asimd, bool has_dotprod, bool has_i8mm, bool has_sve, bool has_sve2);
Aarch64Int8Isa getAarch64Int8Isa();
bool isSVEInt8Isa(Aarch64Int8Isa isa);
const char* aarch64Int8IsaName(Aarch64Int8Isa isa);

}  // namespace ov::intel_cpu
