// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/model.hpp"

namespace ov {
namespace npuw {
namespace kokoro {

/// Guard the Divide(imag, real) node produced by the OV PyTorch frontend's
/// decomposition of aten::angle against NaN when both inputs are zero
/// (IEEE 754: 0/0 = NaN).
///
/// OpenVINO has no native angle or atan2 op, so during model import the
/// PyTorch frontend translates aten::angle into a chain of OV ops:
///   Divide(imag, real) -> Atan -> Select (quadrant correction)
/// The Divide is not guarded against the (0, 0) case, so when both real
/// and imag parts of a spectral bin are zero, it produces NaN which
/// cascades through the entire decoder.
///
/// This transform finds Divide nodes whose friendly name contains "aten::angle" and
/// inserts a guard on the denominator:
///
///   real_is_zero  = Equal(real, 0)
///   imag_is_zero  = Equal(imag, 0)
///   both_zero     = LogicalAnd(real_is_zero, imag_is_zero)
///   guarded_real  = Select(both_zero, epsilon, real)
///   result        = Divide(imag, guarded_real)   // safe: 0/eps -> 0 -> Atan(0) = 0
///
/// Should be called on Model B before NPUW partitioning/compilation.
void guard_angle_divide(std::shared_ptr<ov::Model>& model);

}  // namespace kokoro
}  // namespace npuw
}  // namespace ov
