// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_iterator_meta.hpp"

#include <stdlib.h>

#include <fstream>
#include <string>

#include "openvino/core/type/element_type.hpp"
#include "ov_tensorflow/tensor_bundle.pb.h"
#include "ov_tensorflow/trackable_object_graph.pb.h"

namespace ov {
namespace frontend {
namespace tensorflow {

bool GraphIteratorMeta::is_valid_signature(const ::tensorflow::SignatureDef& signature) const {
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

template <>
std::basic_string<char> get_variables_index_name<char>(const std::string name) {
    return name + ".index";
}

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
template <>
std::basic_string<wchar_t> get_variables_index_name<wchar_t>(const std::wstring name) {
    return name + L".index";
}
#endif

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
