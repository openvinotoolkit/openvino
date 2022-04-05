// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Defines openvino domains for tracing
 * @file gna_itt.hpp
 */

#pragma once

#include <openvino/itt.hpp>

namespace GNAPluginNS {
namespace itt {
namespace domains {
    OV_ITT_DOMAIN(GNAPlugin);
    OV_ITT_DOMAIN(GNA_LT);
}
}
}
