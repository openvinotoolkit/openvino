// Copyright (C) 2018-2026 Intel Corporation
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

std::filesystem::path get_variables_index_name(const std::filesystem::path& name) {
    return std::filesystem::path(name) += ".index";
}

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
