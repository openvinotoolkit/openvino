// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Compiler Interface

#pragma once

#include <ze_api.h>
#include <ze_graph_ext.h>

#include <type_traits>
#include <utility>

#include "intel_npu/common/npu.hpp"
#include "intel_npu/config/config.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "ze_graph_ext_wrappers_interface.hpp"

namespace intel_npu {

class DriverCompilerAdapter final : public ICompilerAdapter {
public:
    DriverCompilerAdapter(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct);

    std::shared_ptr<IGraph> compile(const std::shared_ptr<const ov::Model>& model, const Config& config) const override;

    std::shared_ptr<IGraph> parse(std::vector<uint8_t> network, const Config& config) const override;

    ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>& model, const Config& config) const override;

private:
    /**
     * @brief Serialize input / output information to string format.
     * @details Format:
     * --inputs_precisions="0:<input1Precision> [1:<input2Precision>]"
     * --inputs_layouts="0:<input1Layout> [1:<input2Layout>]"
     * --outputs_precisions="0:<output1Precision>"
     * --outputs_layouts="0:<output1Layout>"
     *
     * For older compiler versions, the name of the inputs/outputs may be used instead of their indices.
     *
     * Since the layout information is no longer an important part of the metadata values when using the 2.0 OV
     * API, the layout fields shall be filled with default values in order to assure the backward compatibility
     * with the driver.
     */
    std::string serializeIOInfo(const std::shared_ptr<const ov::Model>& model, const bool useIndices) const;

    SerializedIR serializeIR(const std::shared_ptr<const ov::Model>& model,
                             ze_graph_compiler_version_info_t compilerVersion,
                             const uint32_t supportedOpsetVersion) const;

    std::string serializeConfig(const Config& config, ze_graph_compiler_version_info_t compilerVersion) const;

    std::shared_ptr<ZeroInitStructsHolder> _zeroInitStruct;
    std::shared_ptr<ZeGraphExtWrappersInterface> _zeGraphExt;

    ze_device_graph_properties_t _deviceGraphProperties = {};

    Logger _logger;
};

}  // namespace intel_npu
