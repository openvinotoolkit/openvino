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

// TODO interface and inheritance
class IRSerializerBase {
public:
    IRSerializerBase(const std::shared_ptr<const ov::Model>& origModel,
                     const uint16_t compilerMajorVersion,
                     const uint16_t compilerMinorVersion,
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
                                const uint16_t compilerMajorVersion,
                                const uint16_t compilerMinorVersion,
                                const uint32_t supportedOpset = 11)
        : IRSerializerBase(origModel, compilerMajorVersion, compilerMinorVersion, supportedOpset){};

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
                                   const uint16_t compilerMajorVersion,
                                   const uint16_t compilerMinorVersion,
                                   const uint32_t supportedOpset = 11)
        : IRSerializerBase(origModel, compilerMajorVersion, compilerMinorVersion, supportedOpset){};

    SerializedIR serialize() override;

private:
    void serializeModelToBuffer(uint8_t* buffer);

    void serializeModelToStream(std::ostream& stream);

    /**
     * @brief Get size of xml and weights from model
     */
    void countModelSize();

    size_t _serializedModelSize = 0;
};

}  // namespace intel_npu::driver_compiler_utils
