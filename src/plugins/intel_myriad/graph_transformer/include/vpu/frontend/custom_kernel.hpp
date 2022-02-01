// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pugixml.hpp>
#include <ie_common.h>

#include <vpu/utils/enums.hpp>
#include <vpu/utils/small_vector.hpp>

namespace vpu {

namespace ie = InferenceEngine;

VPU_DECLARE_ENUM(CustomParamType,
    Input,
    Output,
    Data,
    LocalData,
    InputBuffer,
    OutputBuffer,
    Int,
    Float)

VPU_DECLARE_ENUM(CustomDataFormat,
                 BYXF = 0,  // NHWC used in most software layers
                 BFYX = 1,  // NCHW used if HW module is enabled
                 YXF = 2,   // HWC used in most software layers
                 FYX = 3,   // CHW used if HW module is enabled
                 BF = 4,    // NC layout
                 Any = 5,   // doesn't really matter
                 None = 6)

VPU_DECLARE_ENUM(CustomDimSource, Input, Output)

struct CustomKernel final {
    struct KernelParam final {
        CustomParamType type = CustomParamType::Input;
        CustomDataFormat format = CustomDataFormat::Any;
        std::string argName;
        int portIndex = -1;
        std::string irSource;
        std::string bufferSizeRule;
        CustomDimSource dimSource;
        int dimIdx = -1;
    };

private:
    std::string _configDir;
    int _maxShaves = 0;
    std::string _kernelBinary;
    SmallVector<KernelParam> _kernelParams;
    SmallVector<std::string> _globalGridSizeRules;
    SmallVector<std::string> _localGridSizeRules;
    SmallVector<std::string> _parameters;
    int _kernelId = 0;

    CustomDimSource _wgDimSource = CustomDimSource::Input;
    int _wgDimIdx = -1;

    int _inputDataCount = 0;

public:
    explicit CustomKernel(const pugi::xml_node& node, std::string configDir);

    void processParametersNode(const pugi::xml_node& node);
    void processWorkSizesNode(const pugi::xml_node& node);

    int maxShaves() const { return _maxShaves; }
    const std::string& kernelBinary() const { return _kernelBinary; }
    SmallVector<KernelParam> bindings() const { return _kernelParams; }
    SmallVector<std::string> globalGridSizeRules() const { return _globalGridSizeRules; }
    SmallVector<std::string> localGridSizeRules() const { return _localGridSizeRules; }
    SmallVector<std::string> parameters() const { return _parameters; }
    int kernelId() const { return _kernelId; }
    CustomDimSource dimSource() const { return _wgDimSource; }
    int dimSourceIndex() const { return _wgDimIdx; }
    int inputDataCount() const { return _inputDataCount; }
};

} // namespace vpu
