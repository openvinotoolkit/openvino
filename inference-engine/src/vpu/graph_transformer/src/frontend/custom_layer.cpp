// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/custom_layer.hpp>
#include <vpu/utils/numeric.hpp>

#include <climits>

#include <map>
#include <fstream>
#include <streambuf>
#include <tuple>
#include <utility>
#include <string>
#include <vector>

#if defined(__linux__) || defined (__APPLE__)
# include <dlfcn.h>
#endif

#ifdef _WIN32
# include <windows.h>
#endif

#include <bitset>

#include <description_buffer.hpp>
#include <xml_parse_utils.h>
#include <caseless.hpp>

#include <vpu/utils/simple_math.hpp>
#include <vpu/utils/error.hpp>
#include <cstring>

namespace vpu {

namespace {

void assertExactlyOneOccurrence(const pugi::xml_node &node, const SmallVector<std::string>& children) {
    for (const auto &name : children) {
        const auto& child = node.child(name.c_str());
        VPU_THROW_UNLESS(!child.empty(), "Required parameter %s is not found", name);
        VPU_THROW_UNLESS(child.next_sibling(name.c_str()).empty(),
            "Found several definitions of the parameter %s", name);
    }
}

void assertOneOrMoreOccurrence(const pugi::xml_node &node, const SmallVector<std::string>& children) {
    for (const auto& name : children) {
        const auto& child = node.child(name.c_str());
        VPU_THROW_UNLESS(!child.empty(),
            "Required parameter %s is not found", name);
    }
}

void assertZeroOrOneOccurrence(const pugi::xml_node& node, const SmallVector<std::string>& childNames) {
    for (const auto& name : childNames) {
        const auto& child = node.child(name.c_str());
        VPU_THROW_UNLESS(!child.empty() || child.next_sibling(name.c_str()).empty(),
            "Found several definitions of the parameter %s", name);
    }
}

void assertNoEmptyAttributes(const pugi::xml_node& customLayer) {
    const auto checkAttributes = [&customLayer](const pugi::xml_node& node) {
        for (const auto& attr : node.attributes()) {
            VPU_THROW_UNLESS(strlen(attr.value()) != 0,
                "Wrong custom layer XML: Custom layer %s has node <%s> with an empty attribute %s",
                customLayer.attribute("name").value(), node.name(), attr.name());
        }
    };

    checkAttributes(customLayer);

    for (const auto& child : customLayer.children()) {
        assertNoEmptyAttributes(child);
    }
}

}  // namespace

ie::details::caseless_map<std::string, std::vector<CustomLayer::Ptr>> CustomLayer::loadFromFile(
        const std::string& configFile,
        bool canBeMissed) {
    pugi::xml_document xmlDoc;
    pugi::xml_parse_result res = xmlDoc.load_file(configFile.c_str());

    if (res.status != pugi::status_ok) {
        if (canBeMissed) {
            // Config file might not exist - like global config, for example.
            return {};
        } else {
            VPU_THROW_FORMAT("Failed to load custom layer configuration file %s : %s at offset %s",
                configFile, res.description(), res.offset);
        }
    }

#ifdef _WIN32
    char path[MAX_PATH];
    auto abs_path_ptr = _fullpath(path, configFile.c_str(), MAX_PATH);
#elif defined(__linux__) || defined(__APPLE__)
    char path[PATH_MAX];
    auto abs_path_ptr = realpath(configFile.c_str(), path);
#endif

    VPU_THROW_UNLESS(abs_path_ptr != nullptr,
        "Failed to load custom layer configuration file %s : can't get canonicalized absolute path", configFile);

    std::string abs_file_name(path);

    // Try extracting directory from config path.
    auto dir_split_pos = abs_file_name.find_last_of("/\\");
    auto colon_pos = abs_file_name.find_first_of(':');
    auto first_slash_pos = abs_file_name.find_first_of('/');

    // If path is absolute.
    std::string dir_path;
    if (dir_split_pos != std::string::npos && (colon_pos != std::string::npos || first_slash_pos == 0)) {
        dir_path = abs_file_name.substr(0, dir_split_pos);
    } else {
        VPU_THROW_EXCEPTION
            << "Failed to load custom layer configuration file " << configFile
            << " : path is not valid";
    }

    auto out = ie::details::caseless_map<std::string, std::vector<CustomLayer::Ptr>> {};
    for (auto r = xmlDoc.document_element(); r; r = r.next_sibling()) {
        auto layerPtr = std::make_shared<CustomLayer>(dir_path, r);
        out[layerPtr->_layerName].push_back(std::move(layerPtr));
    }

    return out;
}


CustomLayer::CustomLayer(std::string configDir, const pugi::xml_node& customLayer) : _configDir(std::move(configDir)) {
    const auto cmp = ie::details::CaselessEq<std::string>{};
    const auto nodeName = customLayer.name();
    VPU_THROW_UNLESS(cmp(nodeName, "CustomLayer"),
        "Wrong custom layer XML : Node is not CustomLayer, but %s",  nodeName);

    const auto nodeType = XMLParseUtils::GetStrAttr(customLayer, "type");
    VPU_THROW_UNLESS(cmp(nodeType, "MVCL"),
        "Wrong custom layer XML : Type is not MVCL, but %s", nodeType);

    const auto version = XMLParseUtils::GetIntAttr(customLayer, "version");
    VPU_THROW_UNLESS(version == 1, "Wrong custom layer XML : only version 1 is supported");

    _layerName = XMLParseUtils::GetStrAttr(customLayer, "name");

    assertNoEmptyAttributes(customLayer);

    assertZeroOrOneOccurrence(customLayer, {"Where"});
    const auto whereNode = customLayer.child("Where");
    for (auto where : whereNode.attributes()) {
        _whereParams[where.name()] = where.value();
    }

    assertOneOrMoreOccurrence(customLayer, {"Kernel"});
    auto kernelNodes = [&] {
        auto nodes = SmallVector<pugi::xml_node>{};
        FOREACH_CHILD(kernel, customLayer, "Kernel") {
            assertExactlyOneOccurrence(kernel, {"Parameters", "WorkSizes"});
            assertOneOrMoreOccurrence(kernel, {"Source"});
            nodes.push_back(kernel);
        }
        return nodes;
    }();

    if (kernelNodes.size() == 1) {
        _kernels.emplace_back(kernelNodes.front(), _configDir);
    } else {
        auto stageOrder = std::map<int, CustomKernel>{};
        for (auto& kernel : kernelNodes) {
            const auto stageAttr = kernel.attribute("stage");
            VPU_THROW_UNLESS(stageAttr, "Error while binding %s custom layer: for multi-kernel binding, "
                "each kernel should be provided with 'stage' attribute.", _layerName);

            const auto stageNum = std::stod(stageAttr.value());
            VPU_THROW_UNLESS(stageOrder.find(static_cast<int>(stageNum)) == stageOrder.end(),
                "Error while binding %s custom layer: found duplicating stage id.", _layerName);

            stageOrder.emplace(static_cast<int>(stageNum), CustomKernel{kernel, _configDir});
        }

        VPU_THROW_UNLESS(!stageOrder.empty(),
            "Error while binding %s custom layer: No kernels are found.", _layerName);
        VPU_THROW_UNLESS(stageOrder.begin()->first == 0,
            "Error while binding %s custom layer: Stage 0 is not found.", _layerName);
        VPU_THROW_UNLESS(stageOrder.rbegin()->first == stageOrder.size() - 1,
            "Error while binding %s custom layer: Kernels should have stage id from 0 to N.", _layerName);

        for (auto& stage : stageOrder) {
            _kernels.push_back(std::move(stage.second));
        }
    }

    const auto addPorts = [](std::map<int, CustomDataFormat>& ports, const CustomKernel::KernelParam& newEdge) {
        const auto layerInput = ports.find(newEdge.portIndex);
        if (layerInput == ports.end()) {
            ports.emplace(newEdge.portIndex, newEdge.format);
        } else if (newEdge.format == CustomDataFormat::Any) {
            return;
        } else if (layerInput->second == CustomDataFormat::Any) {
            layerInput->second = newEdge.format;
        }
    };

    for (const auto& kernel : _kernels) {
        for (const auto& binding : kernel.bindings()) {
            if (binding.type == CustomParamType::Input) {
                addPorts(_inputs, binding);
            }
            if (binding.type == CustomParamType::Output) {
                addPorts(_outputs, binding);
            }
        }
    }
}

bool CustomLayer::isLegalSizeRule(const std::string& rule, std::map<std::string, std::string> layerParams) {
    {
        auto sizes = SmallVector<std::pair<std::string, std::string>> {
            { "b", "1" }, { "B", "1" },
            { "f", "1" }, { "F", "1" },
            { "y", "1" }, { "Y", "1" },
            { "x", "1" }, { "X", "1" },
        };

        std::move(begin(sizes), end(sizes), inserter(layerParams, end(layerParams)));
    }

    MathExpression expr;
    expr.setVariables(layerParams);

    try {
        expr.parse(rule);
    } catch (...) {
        return false;
    }

    return true;
}

CustomDataFormat CustomLayer::formatFromLayout(const InferenceEngine::Layout& layout) {
    const auto layoutToFormat = std::map<ie::Layout, CustomDataFormat> {
        { ie::NCHW , CustomDataFormat::BFYX },
        { ie::NHWC , CustomDataFormat::BYXF },
        { ie::CHW , CustomDataFormat::FYX },
        { ie::NC , CustomDataFormat::BF },
        { ie::ANY , CustomDataFormat::Any }
    };

    const auto it = layoutToFormat.find(layout);
    VPU_THROW_UNLESS(it != layoutToFormat.end(), "Tensor node has an invalid format %s", layout);
    return it->second;
}

bool CustomLayer::meetsWhereRestrictions(const std::map<std::string, std::string>& params) const {
    const auto cmp = ie::details::CaselessEq<std::string>{};

    for (const auto& where : _whereParams) {
        const auto restrictedParam = [&](const std::pair<std::string, std::string>& param) {
            return param.first == where.first;
        };

        const auto param = std::find_if(begin(params), end(params), restrictedParam);
        if (param == params.end()) {
            return false;
        }

        const auto& restriction = where.second;
        const auto number = parseNumber<float>(param->second);

        const auto meetsRestriction = [&] {
            // compare non-number restrictions (ex. kernel="3,3")
            if (!number.hasValue()) {
                return cmp(param->second, restriction);
            } else {
                if (restriction[0] == '>' && restriction[1] == '=') {
                    const auto to_compare = std::stof(restriction.substr(2, std::string::npos));
                    return number.get() >= to_compare;
                } else if (restriction[0] == '<' && restriction[1] == '=') {
                    const auto to_compare = std::stof(restriction.substr(2, std::string::npos));
                    return number.get() <= to_compare;
                } else if (restriction[0] == '>') {
                    const auto to_compare = std::stof(restriction.substr(1, std::string::npos));
                    return number.get() > to_compare;
                } else if (restriction[0] == '<') {
                    const auto to_compare = std::stof(restriction.substr(1, std::string::npos));
                    return number.get() < to_compare;
                } else if (restriction[0] == '!' && restriction[1] == '=') {
                    const auto to_compare = std::stof(restriction.substr(2, std::string::npos));
                    return number.get() != to_compare;
                }
                return number.get() == std::stof(restriction);
            }
        }();

        if (!meetsRestriction) {
            return false;
        }
    }
    return true;
}

}  // namespace vpu
