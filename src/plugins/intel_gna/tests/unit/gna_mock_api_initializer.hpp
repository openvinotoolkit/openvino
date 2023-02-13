// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <gna2-common-api.h>

#include "cstdint"
#include "gna_mock_api.hpp"
#include "vector"

class GnaMockApiInitializer {
    GNACppApi _mock_api;
    std::vector<std::vector<uint8_t>> _mocked_gna_memory;
    Gna2DeviceVersion _gna_device_version = Gna2DeviceVersion::Gna2DeviceVersionSoftwareEmulation;
    bool _create_model = true;

public:
    void init();
    void set_gna_device_version(const Gna2DeviceVersion val);
    void set_create_model(const bool val);
};
