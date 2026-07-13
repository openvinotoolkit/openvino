// Copyright (C) 2018-2026 Intel Corporation
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

bool GraphIteratorSavedModel::is_supported(const std::filesystem::path& path) {
    if (ov::util::directory_exists(path)) {
        FRONT_END_GENERAL_CHECK(util::file_exists(path / "saved_model.pb"),
                                "Could not open the file: ",
                                path / "saved_model.pb");
        return true;
    } else {
        return false;
    }
}

std::filesystem::path get_saved_model_name() {
    return "saved_model.pb";
}

std::filesystem::path get_variables_index_name() {
    return std::filesystem::path("variables") / "variables.index";
}

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
