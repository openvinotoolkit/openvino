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

namespace intel_npu::driver_compiler_utils {

class IRSerializer {
public:
    IRSerializer(const std::shared_ptr<const ov::Model>& origModel);

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
    size_t _xmlSize = 0;
    size_t _weightsSize = 0;
};

}  // namespace intel_npu::driver_compiler_utils
