// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Defines openvino domains for tracing
 * @file itt.h
 */

#pragma once

#include <openvino/itt.hpp>

namespace ov {
namespace intel_cpu {
namespace itt {
namespace domains {
    OV_ITT_DOMAIN(intel_cpu);
    OV_ITT_DOMAIN(MKLDNN_LT);
}
}
}
}
