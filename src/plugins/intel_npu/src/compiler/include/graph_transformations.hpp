// Copyright (C) 2018-2024 Intel Corporation
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
namespace intel_npu::driverCompilerAdapter {

class IR {
public:
    IR(const std::shared_ptr<const ov::Model>& origModel, uint32_t supportedVersionByCompiler = 11);

    std::istream& getXml() {
        return _xmlStream;
    }

    std::istream& getWeights() {
        return _weightsStream;
    }

private:
    /**
     * @brief Serialize OpenVINO model to IR, get xml and bin data
     */
    void serializeOVModelToIR(std::shared_ptr<ov::Model> model, uint32_t supportedVersionByCompiler);

    // Streams for normal model
    std::stringstream _xmlStream;

    // Use custom stream buffer for weights to support 2G+ files on Windows
    CustomStreamBuf _weightsCache;
    std::iostream _weightsStream;
};

}  // namespace intel_npu::driverCompilerAdapter
