// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for ICore interface
 * @file ie_icore.hpp
 */

#pragma once

#include <array>
#include <memory>
#include <string>

#include <ie_plugin_ptr.hpp>
#include "threading/ie_itask_executor.hpp"

namespace InferenceEngine {

/**
 * @interface ICore
 * @brief Minimal ICore interface to allow plugin to get information from Core Inference Engine class.
 * @ingroup ie_dev_api_plugin_api
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
    virtual InferenceEnginePluginPtr GetPluginByName(const std::string& deviceName) const = 0;

    /**
     * @brief Reads IR xml and bin (with the same name) files
     * @param model string with IR
     * @param weights shared pointer to constant blob with weights
     * @return CNNNetwork
     */
    virtual CNNNetwork ReadNetwork(const std::string& model, const Blob::CPtr& weights) const = 0;

    /**
     * @brief Reads IR xml and bin files
     * @param modelPath path to IR file
     * @param binPath path to bin file, if path is empty, will try to read bin file with the same name as xml and
     * if bin file with the same name was not found, will load IR without weights.
     * @return CNNNetwork
     */
    virtual CNNNetwork ReadNetwork(const std::string& modelPath, const std::string& binPath) const = 0;

    /**
     * @brief Default virtual destructor
     */
    virtual ~ICore() = default;
};

/**
 * @brief Type of magic value
 * @ingroup ie_dev_api_plugin_api
 */
using ExportMagic = std::array<char, 4>;

/**
 * @brief Magic number used by ie core to identify exported network with plugin name
 * @ingroup ie_dev_api_plugin_api
 */
constexpr static const ExportMagic exportMagic = {{0x1, 0xE, 0xE, 0x1}};

}  // namespace InferenceEngine
