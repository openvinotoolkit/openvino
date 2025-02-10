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

/**
 * @brief Contain all required transformation on OpenVINO model in case for external compiler usage and
 *  providing forward compatibility (OV model with opset N+M, external compiler with opset N)
 */
namespace intel_npu::driver_compiler_utils {

class IRSerializer {
public:
    IRSerializer(const std::shared_ptr<const ov::Model>& origModel,
                 const uint32_t supportedOpset = 11,
                 const bool commonPassesApplied = false);

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

    /**
     * @brief Indicates whether or not the OV common passes have already been applied on the model.
     * @details This attribute will be stored inside model's runtime information in order to be sent to the compiler.
     * The compiler will thus know if applying the same passes on its end is required.
     */
    bool _commonPassesApplied;
};

}  // namespace intel_npu::driver_compiler_utils
