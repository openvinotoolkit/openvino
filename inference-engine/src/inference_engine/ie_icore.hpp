// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for ICore interface
 * @file ie_icore.hpp
 */
#pragma once

#include <memory>
#include <string>
#include "ie_plugin_ptr.hpp"
#include "cpp_interfaces/ie_itask_executor.hpp"

namespace InferenceEngine {

class ITaskExecutor;

/**
 * @brief Minimal ICore interface to allow plugin to get information from Core Inference Engine class.
 */
class ICore {
public:
    /**
     * @brief Returns global to Inference Engine class task executor
     * @return Reference to task executor
     */
    virtual std::shared_ptr<ITaskExecutor> GetTaskExecutor() const = 0;

    /**
     * @brief Returns reference to plugin by a device name
     * @param deviceName - a name of device
     * @return Reference to plugin
     */
    virtual InferenceEnginePluginPtr GetPluginByName(const std::string & deviceName) const = 0;

    /**
     * @brief Default virtual destructor
     */
    virtual ~ICore() = default;
};

}  // namespace InferenceEngine
