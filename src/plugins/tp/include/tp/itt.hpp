// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief ITT domains of the TP plugin. The transformations' (ov::tp::transforms) live with them, the collectives'
 *        (ov::tp::collective) in the device plugin.
 * @file itt.hpp
 */

#pragma once

#include "openvino/itt.hpp"

namespace ov {
namespace tp {
namespace itt {
namespace domains {
OV_ITT_DOMAIN(tp_plugin, "ov::tp::plugin");
OV_ITT_DOMAIN(tp_inference, "ov::tp::inference");
}  // namespace domains
}  // namespace itt
}  // namespace tp
}  // namespace ov
