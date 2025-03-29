// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <memory>
#include <string>
#include <vector>
#include <map>
#include "pugixml.hpp"
#include "intel_gpu/runtime/format.hpp"

namespace ov::intel_gpu {

using CustomLayerPtr = std::shared_ptr<class CustomLayer>;
using CustomLayerMap = std::map<std::string, CustomLayerPtr>;
class CustomLayer{
public:
    static void LoadFromFile(
        const std::string configFile,
        CustomLayerMap& customLayers,
        bool can_be_missed = false);

    typedef enum {
        Input,
        Output,
        Data,
    } ParamType;
    struct KerenlParam {
        KerenlParam() :type(Input), paramIndex(-1), portIndex(-1),
                       format(cldnn::format::any) {}
        ParamType type;
        int paramIndex;
        int portIndex;
        std::string blobName;
        cldnn::format format;
    };

    struct KernelDefine {
        std::string name;
        std::string param;
        std::string default_value;
        std::string prefix;
        std::string postfix;
    };

    const std::string& Name()const { return m_layerName; }
    const std::string& KernelSource()const { return m_kernelSource; }
    const std::string& KernelEntry()const { return m_kernelEntry; }
    const std::vector<KernelDefine>& Defines()const { return m_defines; }
    const std::string& CompilerOptions()const { return m_compilerOptions; }
    const std::vector<std::string>& GlobalSizeRules()const { return m_globalSizeRules; }
    const std::vector<std::string>& LocalSizeRules()const { return m_localSizeRules; }
    const std::vector<KerenlParam>& KernelParams()const { return m_kernelParams; }
    int InputDimSourceIndex() { return m_wgDimInputIdx; }

protected:
    CustomLayer() : m_wgDimInputIdx(0) {}
    explicit CustomLayer(const std::string dirname) : m_configDir(dirname), m_wgDimInputIdx(0) {}

    bool Error() const { return m_ErrorMessage.length() > 0; }
    void LoadSingleLayer(const pugi::xml_node& node);
    void ProcessKernelNode(const pugi::xml_node& node);
    void ProcessBuffersNode(const pugi::xml_node& node);
    void ProcessCompilerOptionsNode(const pugi::xml_node& node);
    void ProcessWorkSizesNode(const pugi::xml_node& node);
    static bool IsLegalSizeRule(const std::string& rule);
    static cldnn::format FormatFromString(const std::string& str);

    std::string m_configDir;
    std::string m_layerName;
    std::string m_kernelSource;
    std::string m_kernelEntry;
    std::vector<KernelDefine> m_defines;  // <name , take value from> <x,y> --> #define x value_of
    std::string m_compilerOptions;
    int m_wgDimInputIdx;
    std::vector<std::string> m_globalSizeRules;
    std::vector<std::string> m_localSizeRules;
    std::vector<KerenlParam> m_kernelParams;
    std::string m_ErrorMessage;
};

}  // namespace ov::intel_gpu
