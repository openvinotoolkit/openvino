// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "decoder_flatbuffer.h"
#include "schema_generated.h"

namespace ov {
namespace frontend {
namespace tensorflow_lite {

namespace {
const std::map<tflite::TensorType, ov::element::Type> &TYPE_MAP() {
    static const std::map<tflite::TensorType, ov::element::Type> type_map{
            {TensorType_FLOAT32,        element::f32},
            {TensorType_FLOAT16,        element::f16},
            {TensorType_INT32,          element::i32},
            {TensorType_UINT8,          element::u8},
            {TensorType_INT64,          element::i64},
            {TensorType_BOOL,           element::boolean},
            {TensorType_INT16,          element::i16},
            {TensorType_INT8,           element::i8},
            {TensorType_FLOAT64,        element::f64},
            {TensorType_UINT64,         element::u64},
            {TensorType_UINT32,         element::u32},
            {TensorType_UINT16,         element::u16},
            {TensorType_INT4,           element::i4},
//          {TensorType_STRING,       element::string},
//          {TensorType_COMPLEX64,    element::complex64},
//          {TensorType_COMPLEX128,     element::complex128},
//          {TensorType_RESOURCE,       element::resource},
//          {TensorType_VARIANT,        element::variant},
    };
    return type_map;
}

ov::element::Type get_ov_type(const tflite::TensorType& tf_type) {
    const auto& mapping = TYPE_MAP();
    if (mapping.find(tf_type) == mapping.end()) {
        FRONT_END_THROW("Unexpected type");
    }
    return mapping.at(tf_type);
}

ov::PartialShape get_ov_shape(const flatbuffers::Vector<int32_t>* tf_shape) {
    return ov::Shape{tf_shape->begin(), tf_shape->end()};
}

}
}
}
}