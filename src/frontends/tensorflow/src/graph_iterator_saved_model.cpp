// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_iterator_saved_model.hpp"

#include <stdlib.h>

#include <fstream>
#include <string>

#include "openvino/core/type/element_type.hpp"
#include "ov_tensorflow/tensor_bundle.pb.h"
#include "ov_tensorflow/trackable_object_graph.pb.h"

namespace ov {
namespace frontend {
namespace tensorflow {

bool GraphIteratorSavedModel::is_valid_signature(const ::tensorflow::SignatureDef& signature) const {
    for (const auto& it : signature.inputs()) {
        if (it.second.name().empty())
            return false;
    }
    for (const auto& it : signature.outputs()) {
        if (it.second.name().empty())
            return false;
    }
    return true;
}

bool GraphIteratorSavedModel::is_supported(const std::string& path) {
    if (ov::util::directory_exists(path)) {
        FRONT_END_GENERAL_CHECK(util::file_exists(ov::util::path_join({path, "saved_model.pb"})),
                                "Could not open the file: \"",
                                ov::util::path_join({path, "saved_model.pb"}),
                                '"');
        return true;
    } else {
        return false;
    }
}

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
bool GraphIteratorSavedModel::is_supported(const std::wstring& path) {
    return ov::util::directory_exists(path) && ov::util::file_exists(ov::util::path_join_w({path, L"saved_model.pb"}));
}
#endif

template <>
std::basic_string<char> get_saved_model_name<char>() {
    return "/saved_model.pb";
}
template <>
std::basic_string<char> get_variables_index_name<char>() {
    return "/variables/variables.index";
}

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
template <>
std::basic_string<wchar_t> get_saved_model_name<wchar_t>() {
    return L"/saved_model.pb";
}
template <>
std::basic_string<wchar_t> get_variables_index_name<wchar_t>() {
    return L"/variables/variables.index";
}
#endif

std::vector<std::string> GraphIteratorSavedModel::split_tags(const std::string tags) const {
    std::vector<std::string> tag_list = {};
    std::size_t len = tags.length();
    if (len == 0) {
        return tag_list;
    }
    std::string tag = "";
    std::size_t last_delimeter_pos = 0;
    std::size_t delimeter_pos = std::string::npos;
    while ((delimeter_pos = tags.find_first_of(",", last_delimeter_pos)) != std::string::npos) {
        tag = tags.substr(last_delimeter_pos, delimeter_pos - last_delimeter_pos);
        tag_list.push_back(tag);
        last_delimeter_pos = delimeter_pos + 1;
    }
    if (last_delimeter_pos != std::string::npos) {
        if (last_delimeter_pos < len) {
            tag = tags.substr(last_delimeter_pos);
        } else {
            tag = "";
        }
        tag_list.push_back(tag);
    }
    return tag_list;
}

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
