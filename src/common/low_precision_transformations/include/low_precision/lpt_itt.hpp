// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Defines openvino domains for tracing
 * @file lpt_itt.hpp
 */

#pragma once


#include "openvino/itt.hpp"

namespace ov {
namespace pass {
namespace low_precision {
namespace itt {
namespace domains {

OV_ITT_DOMAIN(LPT);
OV_ITT_DOMAIN(LPT_LT);

} // namespace domains
} // namespace itt
} // namespace low_precision
} // namespace pass
} // namespace ov
