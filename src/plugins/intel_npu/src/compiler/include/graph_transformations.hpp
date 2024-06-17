// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <fstream>
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
    ~IR();

    std::istream& getXml() {
        if (!_isLargeModel) {
            return _xml;
        } else {
            return _xmlStream;
        }
    }

    std::istream& getWeights() {
        if (!_isLargeModel) {
            return _weights;
        } else {
            return _weightsStream;
        }
    }

private:
    bool isLargeModel() const {
        return _isLargeModel;
    }

    /**
     * @brief Serialize OpenVINO model to IR
     */
    void serializeToIR(uint32_t supportedVersionByCompiler);

    Logger _logger;
    std::shared_ptr<ov::Model> _model;

    std::stringstream _xml;
    std::stringstream _weights;

    // Streams that use local stream_buf
    bool _isLargeModel;
    CustomStreamBuf _xmlCache;
    CustomStreamBuf _weightsCache;
    std::iostream _xmlStream;
    std::iostream _weightsStream;
};

}  // namespace intel_npu::driverCompilerAdapter
