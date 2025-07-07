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
#include "openvino/pass/serialize.hpp"

/**
 * @brief Contain all required transformation on OpenVINO model in case for external compiler usage and
 *  providing forward compatibility (OV model with opset N+M, external compiler with opset N)
 */
namespace intel_npu::driver_compiler_utils {

class IRSerializer {
public:
    IRSerializer(const std::shared_ptr<const ov::Model>& origModel,
                 const uint32_t supportedOpset = 11,
                 ov::pass::WeightsMapWrapper* weightsMapWrapper = nullptr);

    size_t getXmlSize() const {
        if (_weightsMapWrapper) {
            // Use stingstream to get xml if we use weights map
            return _xmlString.size() + 1;
        } else {
            // Use custom stream buffer to get xml size
            return _xmlSize;
        }
    }

    size_t getWeightsSize() const {
        if (_weightsMapWrapper) {
            // Store pointer to buffer if we use weights map
            return sizeof(void*);
        } else {
            // Store weights content to buffer
            return _weightsSize;
        }
    }

    /**
     * @brief Serialize OpenVINO model to target buffer
     */
    void serializeModelToBuffer(uint8_t* xml, uint8_t* weights);

private:
    /**
     * @brief Serialize OpenVINO model to target stream
     */
    void serializeModelToStream(std::ostream& xml,
                                std::ostream& weights,
                                ov::pass::WeightsMapWrapper* weightsMapWrapper = nullptr);

    /**
     * @brief Get size of xml and weights from model
     */
    void countModelSize();

    Logger _logger;
    std::shared_ptr<ov::Model> _model = nullptr;
    uint32_t _supportedOpset = 11;
    size_t _xmlSize = 0;
    size_t _weightsSize = 0;
    ov::pass::WeightsMapWrapper* _weightsMapWrapper = nullptr;
    std::string _xmlString;
};

}  // namespace intel_npu::driver_compiler_utils
