// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/common/igraph.hpp"

namespace intel_npu {

class ICompilerAdapter {
public:
    virtual std::shared_ptr<IGraph> compile(const std::shared_ptr<const ov::Model>& model,
                                            const FilteredConfig& config) const = 0;

    /**
     * @brief Compiles the model, weights separation enabled.
     * @details The result of compilation will be a binary object that does not contain a significant portion of
     * weights. The binary object will include two types of schedules: weights initialization and the operations of the
     * main graph. In order to run inference on this weightless blob, the original weights will need to be provided as
     * inputs to the weights initialization schedule. Running this will output the processed weights, that can then be
     * fed to the main schedule and therefore enable it to run predictions.
     *
     * @param model The model that will be compiled.
     * @param config Will be passed to the compiler. Additionally, the "SEPARATE_WEIGHTS_VERSION" option will determine
     * which weights separation implementation will be used. See the weights separation specific methods within
     * "icompiler.hpp".
     * @return A "WeightlessGraph" type of object.
     */
    virtual std::shared_ptr<IGraph> compileWS(std::shared_ptr<ov::Model>&& model,
                                              const FilteredConfig& config) const = 0;

    virtual ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>& model,
                                      const FilteredConfig& config) const = 0;
    virtual uint32_t get_version() const = 0;
    virtual std::vector<std::string> get_supported_options() const = 0;
    virtual bool is_option_supported(std::string optName, std::optional<std::string> optValue = std::nullopt) const = 0;

    virtual ~ICompilerAdapter() = default;
};

}  // namespace intel_npu
