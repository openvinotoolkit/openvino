// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "op_conformance_utils/utils/file.hpp"
#include "functional_test_utils/node_utils.hpp"
#include "conformance.hpp"

namespace ov {
namespace test {
namespace op_conformance {


static auto unique_ops = ov::test::utils::get_unique_ops();

inline std::string get_ref_path(const std::string& model_path) {
    if (std::string(conformance::refCachePath) == "") {
        return "";
    }
    if (!ov::util::directory_exists(conformance::refCachePath)) {
        ov::util::create_directory_recursive(conformance::refCachePath);
    }
    std::string path_to_cache = conformance::refCachePath + std::string(ov::test::utils::FileSeparator);
    std::string ref_name = model_path.substr(model_path.rfind(ov::test::utils::FileSeparator) + 1);
    ref_name = ov::util::replace_extension(ref_name, "bin");
    path_to_cache += ref_name;
    return path_to_cache;
}

// vector<ir_path, ref_path>
inline std::vector<std::pair<std::string, std::string>>
get_model_paths(const std::vector<std::string>& conformance_ir_paths,
                const std::string& operation_name = "undefined") {
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
            if (ov::util::directory_exists(conformance_ir_path)) {
                tmp_buf = ov::util::get_filelist_recursive({conformance_ir_path}, {std::regex(R"(.*\.xml)")});
            } else if (ov::util::file_exists(conformance_ir_path)) {
                tmp_buf = ov::util::read_lst_file({conformance_ir_path});
            } else {
                continue;
            }
            //Save it in a list, first value - path, second - amout of tests with this path
            for (auto& val : tmp_buf) {
                bool is_op = false;
#ifdef _WIN32
                for (auto it = val.begin(); it != val.end(); ++it) {
                    if (*it == '/')
                        val.replace(it, it + 1, ov::test::utils::FileSeparator);
                }
#endif
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

    if (op_filelist.find(operation_name) != op_filelist.end()) {
        return op_filelist[operation_name];
    }
    return {};
}

} // namespace op_conformance
} // namespace test
} // namespace ov