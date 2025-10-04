// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <ze_graph_ext.h>

#include <iostream>
#include <mutex>
#include <string>

#include "intel_npu/config/config.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "openvino/core/model.hpp"
#include "ze_graph_ext.h"

namespace intel_npu {

using SerializedIR = std::pair<size_t, std::shared_ptr<uint8_t>>;

/**
 * @brief Contain all required transformation on OpenVINO model in case for external compiler usage and
 *  providing forward compatibility (OV model with opset N+M, external compiler with opset N)
 */
namespace driver_compiler_utils {

class IRSerializerBase {
public:
    IRSerializerBase(const std::shared_ptr<const ov::Model>& origModel,
                     const ze_graph_compiler_version_info_t compilerVersion,
                     const uint32_t supportedOpset = 11);

    virtual SerializedIR serialize() = 0;

protected:
    Logger _logger;
    std::shared_ptr<ov::Model> _model = nullptr;
    ze_graph_compiler_version_info_t _compilerVersion;
    uint32_t _supportedOpset = 11;
};

class IRSerializerWithWeightsCopy : public IRSerializerBase {
public:
    IRSerializerWithWeightsCopy(const std::shared_ptr<const ov::Model>& origModel,
                                const ze_graph_compiler_version_info_t compilerVersion,
                                const uint32_t supportedOpset = 11);

    SerializedIR serialize() override;

private:
    /**
     * @brief Serialize OpenVINO model to target buffer
     */
    void serializeModelToBuffer(uint8_t* xml, uint8_t* weights);

    /**
     * @brief Serialize OpenVINO model to target stream
     */
    void serializeModelToStream(std::ostream& xml, std::ostream& weights);

    /**
     * @brief Get size of xml and weights from model
     */
    void countModelSize();

    size_t _xmlSize = 0;
    size_t _weightsSize = 0;
};

class IRSerializerWithoutWeightsCopy : public IRSerializerBase {
public:
    IRSerializerWithoutWeightsCopy(const std::shared_ptr<const ov::Model>& origModel,
                                   const ze_graph_compiler_version_info_t compilerVersion,
                                   const uint32_t supportedOpset = 11);

    SerializedIR serialize() override;

private:
    void serializeModelToBuffer(uint8_t* buffer);

    void serializeModelToStream(std::ostream& stream);

    void countModelSize();

    uint64_t _serializedModelSize = 0;
};

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
SerializedIR serializeIR(const std::shared_ptr<const ov::Model>& model,
                         ze_graph_compiler_version_info_t compilerVersion,
                         const uint32_t supportedOpsetVersion,
                         const bool useBetterModelSerialization = true);

std::string serializeIOInfo(const std::shared_ptr<const ov::Model>& model, const bool useIndices);

std::string serializeConfig(const Config& config,
                            ze_graph_compiler_version_info_t compilerVersion,
                            bool turboSupported = false);

static std::mutex rtInfoMutex;

}  // namespace driver_compiler_utils
}  // namespace intel_npu
