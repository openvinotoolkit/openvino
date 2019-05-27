// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <functional>

#include <details/caseless.hpp>

#include <pugixml.hpp>

#include <vpu/utils/enums.hpp>

namespace vpu {

namespace ie = InferenceEngine;

VPU_DECLARE_ENUM(CustomDataFormat,
    BYXF = 0,  // HWC used in most software layers
    BFYX = 1,  // CHW used if HW module is enabled
    Any  = 2,  // doesn't really matter
    None = 3
)

VPU_DECLARE_ENUM(CustomParamType,
    Input,
    Output,
    Data,
    Int,
    Float
)

class CustomLayer final {
public:
    using Ptr = std::shared_ptr<CustomLayer>;

    struct KernelParam final {
        CustomParamType type = CustomParamType::Input;
        CustomDataFormat format = CustomDataFormat::Any;
        std::string argName;
        int portIndex = -1;
        std::string irSource;
    };

    static ie::details::caseless_map<std::string, CustomLayer::Ptr> loadFromFile(
                const std::string& configFile,
                bool canBeMissed = false);

    const std::string& kernelBinary() const { return _kernelBinary; }

    int kernelAddress(int idx = 1) const;

    const std::vector<KernelParam>& bindings() const { return _kernelParams; }
    const std::vector<std::string>& parameters() const { return _parameters; }

    const std::vector<std::string>& globalSizeRules() const { return _globalSizeRules; }
    const std::vector<std::string>& localSizeRules() const { return _localSizeRules; }

    int inputDimSourceIndex() { return _wgDimInputIdx; }

private:
    explicit CustomLayer(const std::string& dirname) : _configDir(dirname) {}

    void loadSingleLayer(const pugi::xml_node& node);
    void processKernelNode(const pugi::xml_node& node);
    void processParametersNode(const pugi::xml_node& node);
    void processWorkSizesNode(const pugi::xml_node& node);

    static bool isLegalSizeRule(const std::string& rule);
    static CustomDataFormat formatFromString(const std::string& str);

private:
    std::string _configDir;
    std::string _layerName;
    std::string _kernelEntry;
    std::string _kernelBinary;

    std::vector<KernelParam> _kernelParams;
    std::vector<std::string> _globalSizeRules;
    std::vector<std::string> _localSizeRules;
    std::vector<std::string> _parameters;

    std::map<uint32_t, uint32_t, std::greater<uint32_t>> _kernelAddress;

    int _wgDimInputIdx = 0;
};

};  // namespace vpu
