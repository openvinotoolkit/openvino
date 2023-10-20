// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_iterator_meta.hpp"

#include <stdlib.h>

#include <fstream>
#include <string>

#include "openvino/core/type/element_type.hpp"
#include "tensor_bundle.pb.h"
#include "trackable_object_graph.pb.h"

namespace ov {
namespace frontend {
namespace tensorflow {

bool GraphIteratorMeta::is_valid_signature(const ::ov_tensorflow::SignatureDef& signature) const {
    const std::map<::ov_tensorflow::DataType, ov::element::Type> types{
        {::ov_tensorflow::DataType::DT_BOOL, ov::element::boolean},
        {::ov_tensorflow::DataType::DT_INT16, ov::element::i16},
        {::ov_tensorflow::DataType::DT_INT32, ov::element::i32},
        {::ov_tensorflow::DataType::DT_INT64, ov::element::i64},
        {::ov_tensorflow::DataType::DT_HALF, ov::element::f16},
        {::ov_tensorflow::DataType::DT_FLOAT, ov::element::f32},
        {::ov_tensorflow::DataType::DT_DOUBLE, ov::element::f64},
        {::ov_tensorflow::DataType::DT_UINT8, ov::element::u8},
        {::ov_tensorflow::DataType::DT_INT8, ov::element::i8},
        {::ov_tensorflow::DataType::DT_BFLOAT16, ov::element::bf16},
        {::ov_tensorflow::DataType::DT_STRING, ov::element::dynamic}};

    for (const auto& it : signature.inputs()) {
        if (it.second.name().empty() || types.find(it.second.dtype()) == types.end())
            return false;
    }
    for (const auto& it : signature.outputs()) {
        if (it.second.name().empty() || types.find(it.second.dtype()) == types.end())
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
