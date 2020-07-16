// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header that defines advanced related properties for VPU plugins.
 * These properties should be used in SetConfig() and LoadNetwork() methods of plugins
 *
 * @file
 */

#pragma once

#include <string>

#define DECLARE_MYRIAD_CONFIG_KEY(name) static constexpr auto MYRIAD_##name = "MYRIAD_" #name
#define DECLARE_MYRIAD_CONFIG_VALUE(name) static constexpr auto MYRIAD_##name = "MYRIAD_" #name

#define DECLARE_MYRIAD_METRIC_KEY(name, ...) DECLARE_METRIC_KEY(MYRIAD_##name, __VA_ARGS__)
