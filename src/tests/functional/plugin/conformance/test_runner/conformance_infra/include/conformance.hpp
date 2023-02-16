// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once


#include <list>
#include <algorithm>

#include "openvino/opsets/opset.hpp"

#include "common_test_utils/file_utils.hpp"

namespace ov {
namespace test {
namespace conformance {
extern const char* targetDevice;
extern const char *targetPluginName;

extern std::vector<std::string> IRFolderPaths;
extern std::vector<std::string> disabledTests;
extern std::list<std::string> dirList;

extern ov::AnyMap pluginConfig;

inline ov::AnyMap readPluginConfig(const std::string &configFilePath) {
    if (!CommonTestUtils::fileExists(configFilePath)) {
        std::string msg = "Input directory (" + configFilePath + ") doesn't not exist!";
        throw std::runtime_error(msg);
    }
    ov::AnyMap config;
    std::ifstream file(configFilePath);
    if (file.is_open()) {
        std::string buffer;
        while (getline(file, buffer)) {
            if (buffer.find("#") == std::string::npos && !buffer.empty()) {
                auto configElements = CommonTestUtils::splitStringByDelimiter(buffer, " ");
                if (configElements.size() != 2) {
                    throw std::runtime_error("Incorrect line to get config item: " + buffer + "\n. Example: \"PLUGIN_CONFIG_KEY=PLUGIN_CONFIG_VALUE\"");
                }
                config.emplace(configElements.front(), configElements.back());
            }
        }
    } else {
        std::string msg = "Error in opening file: " + configFilePath;
        throw std::runtime_error(msg);
    }
    file.close();
    return config;
}

static std::map<std::string, std::set<std::string>> get_unique_ops() {
    std::map<std::string, std::set<std::string>> res;
    for (const auto& opset_pair : get_available_opsets()) {
        std::string opset_name = opset_pair.first;
        const OpSet& opset = opset_pair.second();
        for (const auto& op : opset.get_type_info_set()) {
            std::string op_version = op.get_version(), opset_name = "opset";
            auto pos = op_version.find(opset_name);
            if (pos != std::string::npos) {
                op_version = op_version.substr(pos + opset_name.size());
            }
            if (res.find(op.name) == res.end()) {
                // W/A for existing directories (without op version)
                res.insert({op.name, {""}});
            }
            res[op.name].insert(op_version);
        }
    }
    return res;
}

static std::set<std::string> get_element_type_names() {
    std::vector<ov::element::Type> element_types = { ov::element::Type_t::f64,
                                                     ov::element::Type_t::f32,
                                                     ov::element::Type_t::f16,
                                                     ov::element::Type_t::bf16,
                                                     ov::element::Type_t::i64,
                                                     ov::element::Type_t::i32,
                                                     ov::element::Type_t::i16,
                                                     ov::element::Type_t::i8,
                                                     ov::element::Type_t::i4,
                                                     ov::element::Type_t::u64,
                                                     ov::element::Type_t::u32,
                                                     ov::element::Type_t::u16,
                                                     ov::element::Type_t::u8,
                                                     ov::element::Type_t::u4,
                                                     ov::element::Type_t::u1,
                                                     ov::element::Type_t::boolean,
                                                     ov::element::Type_t::dynamic
                                                   };
    std::set<std::string> result;
    for (const auto& element_type : element_types) {
        std::string element_name = element_type.get_type_name();
        std::transform(element_name.begin(), element_name.end(), element_name.begin(),
                       [](unsigned char symbol){ return std::tolower(symbol); });
        result.insert(element_name);
    }
    return result;
}

static auto unique_ops = get_unique_ops();
static auto element_type_names = get_element_type_names();

inline std::vector<std::string> getModelPaths(const std::vector<std::string>& conformance_ir_paths,
                                              const std::string& opName = "Other") {
    // This is required to prevent re-scan folders each call in case there is nothing found
    static bool listPrepared = false;
    if (!listPrepared) {
        // Looking for any applicable files in a folders
        for (const auto& conformance_ir_path : conformance_ir_paths) {
            std::vector<std::string> tmp_buf;
            if (CommonTestUtils::directoryExists(conformance_ir_path)) {
                tmp_buf =
                    CommonTestUtils::getFileListByPatternRecursive({conformance_ir_path}, {std::regex(R"(.*\.xml)")});
            } else if (CommonTestUtils::fileExists(conformance_ir_path)) {
                tmp_buf = CommonTestUtils::readListFiles({conformance_ir_path});
            } else {
                continue;
            }
            //Save it in a list
            dirList.insert(dirList.end(), tmp_buf.begin(), tmp_buf.end());
        }
        listPrepared = true;
    }

    std::vector<std::string> result;
    if (!opName.empty() && opName != "Other") {
        for (const auto& op_version : unique_ops[opName]) {
            std::string final_op_name = op_version == "" ? opName : opName + "-" + op_version;
            std::string strToFind = CommonTestUtils::FileSeparator + final_op_name + CommonTestUtils::FileSeparator;
            auto it = dirList.begin();
            while (it != dirList.end()) {
                if (it->find(strToFind) != std::string::npos) {
                    result.push_back(*it);
                    it = dirList.erase(it);
                } else {
                    ++it;
                }
            }
        }
    } else if (opName == "Other") {
        // For "Undefined" operation name - run all applicable files in "Undefined" handler
        result.insert(result.end(), dirList.begin(), dirList.end());
    }
    return result;
}

}  // namespace conformance
}  // namespace test
}  // namespace ov
