// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once


#include <list>
#include <algorithm>

#include "openvino/opsets/opset.hpp"

#include "common_test_utils/file_utils.hpp"
#include "functional_test_utils/node_utils.hpp"

namespace ov {
namespace test {
namespace conformance {
extern const char* targetDevice;
extern const char *targetPluginName;
extern const char* refCachePath;

extern std::vector<std::string> IRFolderPaths;
extern std::vector<std::string> disabledTests;

extern ov::AnyMap pluginConfig;

enum ShapeMode {
    DYNAMIC,
    STATIC,
    BOTH
};

extern ShapeMode shapeMode;

inline ov::AnyMap readPluginConfig(const std::string &configFilePath) {
    if (!ov::test::utils::fileExists(configFilePath)) {
        std::string msg = "Input directory (" + configFilePath + ") doesn't not exist!";
        throw std::runtime_error(msg);
    }
    ov::AnyMap config;
    std::ifstream file(configFilePath);
    if (file.is_open()) {
        std::string buffer;
        while (getline(file, buffer)) {
            if (buffer.find("#") == std::string::npos && !buffer.empty()) {
                auto configElements = ov::test::utils::splitStringByDelimiter(buffer, " ");
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

static std::set<std::string> get_element_type_names() {
    std::vector<ov::element::Type> element_types = { ov::element::Type_t::f64,
                                                     ov::element::Type_t::f32,
                                                     ov::element::Type_t::f16,
                                                     ov::element::Type_t::bf16,
                                                     ov::element::Type_t::nf4,
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
                                                     ov::element::Type_t::dynamic,
                                                     ov::element::Type_t::undefined,
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

static auto unique_ops = ov::test::utils::get_unique_ops();
static auto element_type_names = get_element_type_names();

inline std::string get_ref_path(const std::string& model_path) {
    if (std::string(refCachePath) == "") {
        return "";
    }
    if (!ov::test::utils::directoryExists(refCachePath)) {
        ov::test::utils::createDirectoryRecursive(refCachePath);
    }
    std::string path_to_cache = refCachePath + std::string(ov::test::utils::FileSeparator);
    std::string ref_name = model_path.substr(model_path.rfind(ov::test::utils::FileSeparator) + 1);
    ref_name = ov::test::utils::replaceExt(ref_name, "bin");
    path_to_cache += ref_name;
    return path_to_cache;
}

// vector<ir_path, ref_path>
inline std::vector<std::pair<std::string, std::string>>
getModelPaths(const std::vector<std::string>& conformance_ir_paths,
              const std::string& opName = "undefined") {
    // This is required to prevent re-scan folders each call in case there is nothing found
    // {{ op_name, {irs} }}
    static std::unordered_map<std::string, std::vector<std::pair<std::string, std::string>>> op_filelist;
    if (op_filelist.empty()) {
        for (const auto& op_name : unique_ops) {
            op_filelist.insert({op_name.first, {}});
        }
        op_filelist.insert({"undefined", {}});
        std::vector<std::string> filelist;
        // Looking for any applicable files in a folders
        for (const auto& conformance_ir_path : conformance_ir_paths) {
            std::vector<std::string> tmp_buf;
            if (ov::test::utils::directoryExists(conformance_ir_path)) {
                tmp_buf =
                    ov::test::utils::getFileListByPatternRecursive({conformance_ir_path}, {std::regex(R"(.*\.xml)")});
            } else if (ov::test::utils::fileExists(conformance_ir_path)) {
                tmp_buf = ov::test::utils::readListFiles({conformance_ir_path});
            } else {
                continue;
            }
            //Save it in a list, first value - path, second - amout of tests with this path
            for (auto& val : tmp_buf) {
                bool is_op = false;
                for (const auto& path_item : ov::test::utils::splitStringByDelimiter(val, ov::test::utils::FileSeparator)) {
                    auto tmp_path_item = path_item;
                    auto pos = tmp_path_item.find('-');
                    if (pos != std::string::npos) {
                        tmp_path_item = tmp_path_item.substr(0, pos);
                    }
                    if (op_filelist.find(tmp_path_item) != op_filelist.end()) {
                        op_filelist[tmp_path_item].push_back({val, get_ref_path(val)});
                        is_op = true;
                        break;
                    }
                }
                if (!is_op) {
                    op_filelist["undefined"].push_back({val, get_ref_path(val)});
                }
            }
        }
    }

    if (op_filelist.find(opName) != op_filelist.end()) {
        return op_filelist[opName];
    }
    return {};
}

}  // namespace conformance
}  // namespace test
}  // namespace ov
