// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "intel_npu/common/filtered_config.hpp"
#include "openvino/core/model.hpp"
#include "openvino/runtime/common.hpp"
#include "openvino/runtime/profiling_info.hpp"
#include "openvino/runtime/tensor.hpp"

namespace intel_npu {

/**
 * @brief The compiler-in-plugin surface consumed by PluginCompilerAdapter.
 *
 * Deliberately expressed only in OpenVINO and standard types - no VCL, no Level Zero - so
 * PluginCompilerAdapter depends on this abstraction rather than on VCLCompilerImpl directly.
 */
class IVCLCompiler {
public:
    virtual ~IVCLCompiler() = default;

    /**
     * @brief Transforms a network from the OpenVINO model representation to a format executable
     * by a NPU device.
     * @return a pair containing an ov::Tensor object with the compiled model (blob) and an optional
     *         string with runtime requirements for the blob
     */
    virtual std::pair<ov::Tensor, std::optional<std::string>> compile(const std::shared_ptr<const ov::Model>& model,
                                                                      const FilteredConfig& config) const = 0;

    /**
     * @brief Compiles the model, weights separation enabled. All init schedules along with the main
     * one are compiled in the same scope.
     * @return A pair containing one ov::Tensor for each init schedule, followed by another one
     * corresponding to the main part, and an optional compatibility string for the compiled blobs.
     */
    virtual std::pair<std::vector<ov::Tensor>, std::optional<std::string>> compileWsOneShot(
        const std::shared_ptr<ov::Model>& model,
        const FilteredConfig& config) const = 0;

    /**
     * @brief Sequential compilation of Init(s) and Main ("stateless compiler" approach).
     * @param callNumber Tells the compiler which Init (or Main) to return; the plugin does not know
     * the total number of Init schedules.
     */
    virtual std::pair<ov::Tensor, std::optional<std::string>> compileWsIterative(
        const std::shared_ptr<ov::Model>& model,
        const FilteredConfig& config,
        size_t callNumber) const = 0;

    /**
     * @brief Returns information about supported layers of the network passed.
     */
    virtual ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>& model,
                                      const FilteredConfig& config) const = 0;

    /**
     * @brief Returns the compiler version.
     * @return composite uint32_t value: MSB 16 bits = Major version, LSB 16 bits = Minor version.
     */
    virtual uint32_t get_version() const = 0;

    /**
     * @brief Returns the options the compiler supports, already tokenised.
     * @note The VCL calling convention (a char buffer with trailing NULs) is deliberately kept out
     * of this interface; implementations do the trimming and tokenisation themselves.
     */
    virtual std::vector<std::string> get_supported_options() const = 0;

    /**
     * @brief Checks whether the given option and value are supported by the compiler.
     */
    virtual bool is_option_supported(const std::string& option,
                                     const std::optional<std::string>& optValue = std::nullopt) const = 0;

    /**
     * @brief Decodes raw profiling output produced by a run of the given network.
     */
    virtual std::vector<ov::ProfilingInfo> process_profiling_output(const std::vector<uint8_t>& profData,
                                                                    const std::vector<uint8_t>& network) const = 0;
};

}  // namespace intel_npu
