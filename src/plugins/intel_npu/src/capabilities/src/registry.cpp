//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include <cassert>
#include <cstdint>
#include <memory>

#include "meta.hpp"
#include "requirements.hpp"
#include "vpu_capabilities.h"
#include "registry.hpp"
#include "concepts.hpp"

namespace compat {

// namespace {

// void registerNPU4Capabilities(Registry& registry) {
//     registry.registerCapability<WeightlessBlob>();
// }

// }  // namespace

const Registry& Registry::get() {
    static Registry registry;
    return registry;
}

Registry::Registry() {
    Registry::registerCapability<WeightlessBlob>(_hwCapabilities);
    Registry::registerCapability<BatchSize>(_swCapabilities);
    // registerNPU4Capabilities(*this);
}

bool Registry::isRegistered(uint16_t type) const {
    return _hwCapabilities.count(type) == 1 || _swCapabilities.count(type) == 1;
}

std::unique_ptr<const concepts::Capability> Registry::parseHW(uint16_t type, const uint8_t* bytes) const {
    // assert(isRegistered(type));
    return _hwCapabilities.count(type) == 1 ? _hwCapabilities.at(type)(bytes) : nullptr;
}

std::unique_ptr<const concepts::Capability> Registry::parseSW(uint16_t type, const uint8_t* bytes) const {
    // assert(isRegistered(type));
    // printf("!!! Registry::parseSW(%u, %p) !!!\n", type, bytes);
    return _swCapabilities.count(type) == 1 ? _swCapabilities.at(type)(bytes) : nullptr;
}

}  // namespace compat
