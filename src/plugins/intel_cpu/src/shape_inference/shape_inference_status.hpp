// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ov::intel_cpu {

enum class ShapeInferStatus {
    success,  ///< shapes were successfully calculated
    skip      ///< shape inference was skipped.
    ///< This status is used when the implementation was expectedly not able to compute defined output shape
    ///< e.g. in the case of internal dynamism.
};

}  // namespace ov::intel_cpu
