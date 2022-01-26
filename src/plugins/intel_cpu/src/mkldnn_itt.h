// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Defines openvino domains for tracing
 * @file mkldnn_itt.h
 */

#pragma once

#include <openvino/itt.hpp>

namespace MKLDNNPlugin {
namespace itt {
namespace domains {
    OV_ITT_DOMAIN(MKLDNNPlugin);
    OV_ITT_DOMAIN(MKLDNN_LT);
}
}
}
