// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_iterator_saved_model.hpp"

#include <stdlib.h>

#include <fstream>
#include <string>

#include "openvino/core/type/element_type.hpp"
#include "tensor_bundle.pb.h"
#include "trackable_object_graph.pb.h"

namespace ov {
namespace frontend {
namespace tensorflow {

bool GraphIteratorSavedModel::is_valid_signature(const ::tensorflow::SignatureDef& signature) const {
    const std::map<::tensorflow::DataType, ov::element::Type> types{
        {::tensorflow::DataType::DT_BOOL, ov::element::boolean},
        {::tensorflow::DataType::DT_INT16, ov::element::i16},
        {::tensorflow::DataType::DT_INT32, ov::element::i32},
        {::tensorflow::DataType::DT_INT64, ov::element::i64},
        {::tensorflow::DataType::DT_HALF, ov::element::f16},
        {::tensorflow::DataType::DT_FLOAT, ov::element::f32},
        {::tensorflow::DataType::DT_DOUBLE, ov::element::f64},
        {::tensorflow::DataType::DT_UINT8, ov::element::u8},
        {::tensorflow::DataType::DT_INT8, ov::element::i8},
        {::tensorflow::DataType::DT_BFLOAT16, ov::element::bf16},
        {::tensorflow::DataType::DT_STRING, ov::element::dynamic}};

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

bool GraphIteratorSavedModel::is_supported(const std::string& path) {
    return ov::util::directory_exists(path) && ov::util::file_exists(ov::util::path_join({path, "saved_model.pb"}));
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

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
