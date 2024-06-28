// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/runtime/core.hpp>

// create dummy network for tests
std::shared_ptr<ov::Model> buildSingleLayerSoftMaxNetwork();

std::shared_ptr<ov::Model> createModelWithLargeSize();

// class encapsulated Platform getting from environmental variable
class PlatformEnvironment {
public:
    static const std::string PLATFORM;
};
