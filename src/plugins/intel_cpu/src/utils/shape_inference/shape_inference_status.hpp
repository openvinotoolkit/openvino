// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ov {
namespace intel_cpu {

enum class ShapeInferStatus {
    update, // shapes were successfully calculated
    skip // shape infer was skipped
};

}  // namespace intel_cpu
}  // namespace ov
