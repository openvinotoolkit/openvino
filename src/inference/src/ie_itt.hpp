// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Defines openvino domains for tracing
 * @file ie_itt.hpp
 */

#pragma once

#include <openvino/itt.hpp>

namespace InferenceEngine {
namespace itt {
namespace domains {
OV_ITT_DOMAIN(IE_LT);
}  // namespace domains
}  // namespace itt
}  // namespace InferenceEngine

namespace ov {
namespace itt {
namespace domains {
OV_ITT_DOMAIN(IE);
OV_ITT_DOMAIN(IE_RT);
}  // namespace domains
}  // namespace itt
}  // namespace ov
