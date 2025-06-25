// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <iostream>
#include <string>
#include <vector>

#include "custom_stream_buffer.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "openvino/pass/manager.hpp"
#include "ze_graph_ext_wrappers.hpp"

/**
 * @brief Contain all required transformation on OpenVINO model in case for external compiler usage and
 *  providing forward compatibility (OV model with opset N+M, external compiler with opset N)
 */
namespace intel_npu::driver_compiler_utils {

class IRSerializer {
public:
    IRSerializer(const std::shared_ptr<const ov::Model>& origModel, const uint32_t supportedOpset = 11);

    size_t getXmlSize() const {
        return _xmlSize;
    }

    size_t getWeightsSize() const {
        return _weightsSize;
    }

    /**
     * @brief Serialize OpenVINO model to target buffer
     */
    void serializeModelToBuffer(uint8_t* xml, uint8_t* weights);

private:
    /**
     * @brief Serialize OpenVINO model to target stream
     */
    void serializeModelToStream(std::ostream& xml, std::ostream& weights);

    /**
     * @brief Get size of xml and weights from model
     */
    void countModelSize();

    Logger _logger;
    std::shared_ptr<ov::Model> _model = nullptr;
    uint32_t _supportedOpset = 11;
    size_t _xmlSize = 0;
    size_t _weightsSize = 0;
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
std::string serializeIOInfo(const std::shared_ptr<const ov::Model>& model, const bool useIndices);

SerializedIR serializeIR(const std::shared_ptr<const ov::Model>& model,
                         ze_graph_compiler_version_info_t compilerVersion,
                         const uint32_t supportedOpsetVersion);

std::string serializeConfig(const Config& config,
                            ze_graph_compiler_version_info_t compilerVersion,
                            bool turboSupported = false);

// uint8_t* allocateBlob(uint64_t size) {
//     uint8_t* ptr = static_cast<uint8_t*>(std::calloc(static_cast<size_t>(size), sizeof(uint8_t)));

//     if (ptr == nullptr) {
//         throw std::runtime_error("Memory allocation failed in allocateBlob!");
//     }

//     return ptr;
// }

// void deallocateBlob(uint8_t* ptr) {
//     if (ptr == nullptr) {
//         throw std::runtime_error("Pointer is nullptr in deallocateBlob!");
//     }

//     free(ptr);
// }

}  // namespace intel_npu::driver_compiler_utils
