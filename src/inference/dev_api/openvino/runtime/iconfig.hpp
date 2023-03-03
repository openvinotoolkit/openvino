// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief OpenVINO Runtime CompiledModel interface
 * @file openvino/runtime/iconfig.hpp
 */

#pragma once

#include <string>

#include "openvino/core/any.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {

/**
 * @brief Plugin implements all logic for get/set_property inside
 * class can contain plugin specific fields to improve the performance
 */
class IConfig {
public:
    // get property
    virtual ov::Any get(const std::string& name) const = 0;
    virtual ov::Any get(const ov::PropertyName name) const = 0;

    // set property
    virtual void set(const std::string& name, const ov::AnyMap& value) = 0;
    virtual void set(const ov::PropertyName& name, const ov::AnyMap& value) = 0;

    // Change access to read only for property
    virtual void ro(const std::string& name) = 0;
    virtual void ro(const ov::PropertyName& name) = 0;

    // Change access to read write for property
    virtual void rw(const std::string& name) = 0;
    virtual void rw(const ov::PropertyName& name) = 0;

    // Plugin configuration parameters
    //
    // int deviceId = 0;
    // bool perfCount = true;
    // ov::threading::IStreamsExecutor::Config _streamsExecutorConfig;
    // ...
};

}  // namespace ov
