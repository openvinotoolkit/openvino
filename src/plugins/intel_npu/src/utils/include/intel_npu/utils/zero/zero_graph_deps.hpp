// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>

#include "intel_npu/utils/zero/zero_device.hpp"
#include "intel_npu/utils/zero/zero_driver.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"

namespace intel_npu {

/**
 * @brief What the compiler adapters, the parser and the graphs need from level zero.
 * @details The driver gives the graph extension table, the versions and the capabilities, the device gives its handle
 * and its compiler properties. The context is only here for what the adapters hand to the utils (command lists, the
 * command queue pool, tensors), until those take narrower inputs too. Empty when there is no driver, which is how
 * offline compilation is expressed.
 */
struct ZeroGraphDeps {
    std::shared_ptr<ZeroDriver> driver;
    std::shared_ptr<ZeroDevice> device;
    std::shared_ptr<ZeroContext> context;

    ZeroGraphDeps() = default;
    ZeroGraphDeps(std::nullptr_t) {}
    // Implicit on purpose, so that code that has the context can pass it where the dependencies are expected
    ZeroGraphDeps(const std::shared_ptr<ZeroContext>& zeroContext)
        : driver(zeroContext ? zeroContext->getZeroDriver() : nullptr),
          device(zeroContext ? zeroContext->getZeroDriver()->getDefaultDevice() : nullptr),
          context(zeroContext) {}

    explicit operator bool() const {
        return driver != nullptr;
    }
};

}  // namespace intel_npu
