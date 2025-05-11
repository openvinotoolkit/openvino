// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Defines openvino domains for tracing
 * @file itt.h
 */

#pragma once

#include <openvino/itt.hpp>
namespace ov {
namespace auto_plugin {
namespace itt {
namespace domains {
    OV_ITT_DOMAIN(AutoPlugin);
}
} // namespace itt
} // namespace auto_plugin
} // namespace ov
