// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/custom_layer.hpp"

#include "intel_gpu/plugin/simple_math.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "openvino/core/except.hpp"
#include "openvino/util/xml_parse_utils.hpp"

#include <climits>
#include <fstream>
#include <map>
#include <streambuf>

#ifdef _WIN32
# ifndef NOMINMAX
#  define NOMINMAX
# endif
# include <windows.h>
#endif

using namespace ov::util::pugixml;

#define CheckAndReturnError(cond, errorMsg) \
    if (cond) { std::stringstream ss; ss << errorMsg; m_ErrorMessage = ss.str(); return; }
#define CheckNodeTypeAndReturnError(node, type) \
    CheckAndReturnError((std::string(node.name()).compare(type)), "Wrong node! expected: " << #type << " found: " << node.name())
#define CheckStrAttrAndReturnError(node, attr, value) \
    CheckAndReturnError(get_str_attr(node, attr, "").compare(value), "Wrong attribute value! expected: " << value << " found: " << get_str_attr(node, attr, ""))
#define CheckIntAttrAndReturnError(node, attr, value) \
    CheckAndReturnError(get_int_attr(node, attr, -1) != (value), "Wrong attribute value! expected: " << value << " found: " << get_int_attr(node, attr, -1))

namespace ov::intel_gpu {

void CustomLayer::LoadSingleLayer(const pugi::xml_node & node) {
    // Root checks
    CheckNodeTypeAndReturnError(node, "CustomLayer");
    CheckStrAttrAndReturnError(node, "type", "SimpleGPU");
    CheckIntAttrAndReturnError(node, "version", 1);
    m_layerName = get_str_attr(node, "name", "");
    CheckAndReturnError(m_layerName.length() == 0, "Missing Layer name in CustomLayer");

    // Process child nodes
    ProcessKernelNode(node.child("Kernel"));
    ProcessBuffersNode(node.child("Buffers"));
    ProcessCompilerOptionsNode(node.child("CompilerOptions"));
    ProcessWorkSizesNode(node.child("WorkSizes"));
}

void CustomLayer::ProcessKernelNode(const pugi::xml_node & node) {
    CheckNodeTypeAndReturnError(node, "Kernel");
    CheckAndReturnError(m_kernelSource.length() > 0, "Multiple definition of Kernel");
    m_kernelEntry = get_str_attr(node, "entry", "");
    CheckAndReturnError(m_kernelEntry.length() == 0, "No Kernel entry in layer: " << get_str_attr(node.parent(), "name"));

    // Handle Source nodes
    FOREACH_CHILD(sourceNode, node, "Source") {
        // open file
        std::string filename = m_configDir + "/" + get_str_attr(sourceNode, "filename", "");
        std::ifstream inputFile(filename);
        CheckAndReturnError(!inputFile.is_open(), "Couldn't open kernel file: " << filename);

        // read to string
        std::string fileContent;
        inputFile.seekg(0, std::ios::end);
        fileContent.reserve(inputFile.tellg());
        inputFile.seekg(0, std::ios::beg);

        fileContent.assign((std::istreambuf_iterator<char>(inputFile)),
            std::istreambuf_iterator<char>());

        // append to source string
        m_kernelSource.append("\n// Custom Layer Kernel " + filename + "\n\n");
        m_kernelSource.append(fileContent);
    }

    // Handle Define nodes
    FOREACH_CHILD(defineNode, node, "Define") {
        KernelDefine kd;
        kd.name = get_str_attr(defineNode, "name", "");
        CheckAndReturnError((kd.name.length() == 0), "Missing name for define node");
        kd.param = get_str_attr(defineNode, "param", "");
        kd.default_value = get_str_attr(defineNode, "default", "");
        std::string type = get_str_attr(defineNode, "type", "");
        if (type.compare("int[]") == 0 || type.compare("float[]") == 0) {
            kd.prefix = "(" + type + ") {";
            kd.postfix = "}";
        }
        m_defines.push_back(kd);
    }
}

void CustomLayer::ProcessBuffersNode(const pugi::xml_node & node) {
    CheckNodeTypeAndReturnError(node, "Buffers");
    FOREACH_CHILD(tensorNode, node, "Tensor") {
        KerenlParam kp;
        kp.format = FormatFromString(get_str_attr(tensorNode, "format", "BFYX"));
        CheckAndReturnError(kp.format == cldnn::format::format_num, "Tensor node has an invalid format: " << get_str_attr(tensorNode, "format"));
        kp.paramIndex = get_int_attr(tensorNode, "arg-index", -1);
        CheckAndReturnError(kp.paramIndex == -1, "Tensor node has no arg-index");
        kp.portIndex = get_int_attr(tensorNode, "port-index", -1);
        CheckAndReturnError(kp.portIndex == -1, "Tensor node has no port-index");
        std::string typeStr = get_str_attr(tensorNode, "type");
        if (typeStr.compare("input") == 0) {
            kp.type = ParamType::Input;
        } else if (typeStr.compare("output") == 0) {
            kp.type = ParamType::Output;
        } else {
            CheckAndReturnError(true, "Tensor node has an invalid type: " << typeStr);
        }
        m_kernelParams.push_back(kp);
    }
    FOREACH_CHILD(dataNode, node, "Data") {
        KerenlParam kp;
        kp.type = ParamType::Data;
        kp.paramIndex = get_int_attr(dataNode, "arg-index", -1);
        CheckAndReturnError(kp.paramIndex == -1, "Data node has no arg-index");
        kp.blobName = get_str_attr(dataNode, "name", "");
        CheckAndReturnError(kp.blobName.empty(), "Data node has no name");
        m_kernelParams.push_back(kp);
    }
}

void CustomLayer::ProcessCompilerOptionsNode(const pugi::xml_node & node) {
    if (node.empty()) {
        return;  // Optional node doesn't exist
    }
    CheckNodeTypeAndReturnError(node, "CompilerOptions");
    CheckAndReturnError(m_compilerOptions.length() > 0, "Multiple definition of CompilerOptions");
    m_compilerOptions = get_str_attr(node, "options", "");
}

void CustomLayer::ProcessWorkSizesNode(const pugi::xml_node & node) {
    if (node.empty()) {
        return;  // Optional node doesn't exist
    }
    CheckNodeTypeAndReturnError(node, "WorkSizes");

    m_wgDimInputIdx = -1;
    std::string dim_src_string = node.attribute("dim").as_string("");
    if (!dim_src_string.empty() && "output" != dim_src_string) {
        // try to locate index separator
        auto pos = dim_src_string.find_first_of(',');
        auto flag = dim_src_string.substr(0, pos);
        CheckAndReturnError(("input" != flag), "Invalid WG dim source: " << flag);

        int input_idx = 0;
        if (pos != std::string::npos) {
            // user explicitly set input index in config
            auto input_idx_string = dim_src_string.substr(pos + 1, std::string::npos);
            input_idx = std::stoi(input_idx_string);
        }
        CheckAndReturnError((input_idx < 0), "Invalid input tensor index: " << input_idx);
        m_wgDimInputIdx = input_idx;
    }

    std::string gws = node.attribute("global").as_string("");
    while (!gws.empty()) {
        auto pos = gws.find_first_of(',');
        auto rule = gws.substr(0, pos);
        CheckAndReturnError(!IsLegalSizeRule(rule), "Invalid WorkSize: " << rule);
        m_globalSizeRules.push_back(rule);
        if (pos == std::string::npos) {
            gws.clear();
        } else {
            gws = gws.substr(pos + 1, std::string::npos);
        }
    }

    std::string lws = node.attribute("local").as_string("");
    while (!lws.empty()) {
        auto pos = lws.find_first_of(',');
        auto rule = lws.substr(0, pos);
        CheckAndReturnError(!IsLegalSizeRule(rule), "Invalid WorkSize: " << rule);
        m_localSizeRules.push_back(rule);
        if (pos == std::string::npos) {
            lws.clear();
        } else {
            lws = lws.substr(pos + 1, std::string::npos);
        }
    }
}

bool CustomLayer::IsLegalSizeRule(const std::string & rule) {
    SimpleMathExpression expr;
    expr.SetVariables({
        { 'b', 1 }, { 'B', 1 },
        { 'f', 1 }, { 'F', 1 },
        { 'y', 1 }, { 'Y', 1 },
        { 'x', 1 }, { 'X', 1 },
    });
    if (!expr.SetExpression(rule)) {
        return false;
    }

    try {
        expr.Evaluate();
    } catch (std::exception&) {
        return false;
    }
    return true;
}

cldnn::format CustomLayer::FormatFromString(const std::string & str) {
    static const std::map<std::string, cldnn::format> FormatNameToType = {
        { "BFYX" , cldnn::format::bfyx },
        { "bfyx" , cldnn::format::bfyx },

        { "BYXF" , cldnn::format::byxf },
        { "byxf" , cldnn::format::byxf },

        { "FYXB" , cldnn::format::fyxb },
        { "fyxb" , cldnn::format::fyxb },

        { "YXFB" , cldnn::format::yxfb },
        { "yxfb" , cldnn::format::yxfb },

        { "ANY" , cldnn::format::any },
        { "any" , cldnn::format::any },
    };
    auto it = FormatNameToType.find(str);
    if (it != FormatNameToType.end())
        return it->second;
    else
        return cldnn::format::format_num;
}

void CustomLayer::LoadFromFile(const std::string configFile, CustomLayerMap& customLayers, bool can_be_missed) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "CustomLayer::LoadFromFile");
    pugi::xml_document xmlDoc;
    pugi::xml_parse_result res = xmlDoc.load_file(configFile.c_str());
    if (res.status != pugi::status_ok) {
        if (can_be_missed) {
            // config file might not exist - like global config, for example
            return;
        } else {
            OPENVINO_THROW("Error loading custom layer configuration file: ", configFile, ", ", res.description(), " at offset ", res.offset);
        }
    }

#ifdef _WIN32
    char path[MAX_PATH];
    char* abs_path_ptr = _fullpath(path, configFile.c_str(), MAX_PATH);
#elif __linux__
    char path[PATH_MAX];
    char* abs_path_ptr = realpath(configFile.c_str(), path);
#else
#error "Intel GPU plugin: unknown target system"
#endif
    if (abs_path_ptr == nullptr) {
        OPENVINO_THROW("Error loading custom layer configuration file: ", configFile, ", ", "Can't get canonicalized absolute pathname.");
    }

    std::string abs_file_name(path);
    // try extracting directory from config path
    std::string dir_path;
    std::size_t dir_split_pos = abs_file_name.find_last_of("/\\");
    std::size_t colon_pos = abs_file_name.find_first_of(":");
    std::size_t first_slash_pos = abs_file_name.find_first_of("/");

    if (dir_split_pos != std::string::npos &&
       (colon_pos != std::string::npos || first_slash_pos == 0)) {
        // path is absolute
        dir_path = abs_file_name.substr(0, dir_split_pos);
    } else {
        OPENVINO_THROW("Error loading custom layer configuration file: ", configFile, ", ", "Path is not valid");
    }

    for (auto r = xmlDoc.document_element(); r; r = r.next_sibling()) {
        CustomLayerPtr layer = std::make_shared<CustomLayer>(CustomLayer(dir_path));
        layer->LoadSingleLayer(r);
        if (layer->Error()) {
            customLayers.clear();
            OPENVINO_THROW(layer->m_ErrorMessage);
        } else {
            customLayers[layer->Name()] = layer;
        }
    }
}

}  // namespace ov::intel_gpu
