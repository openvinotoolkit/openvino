// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "attr_value.pb.h"
#include "node_def.pb.h"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/tensor.hpp"
#include "tensor.pb.h"
#include "tensor_shape.pb.h"
#include "types.pb.h"

namespace ov {
namespace frontend {
namespace tensorflow {

ov::element::Type get_ov_type(const ::tensorflow::DataType& type);

ov::Any unpack_tensor_proto(const ::tensorflow::TensorProto& tensor_proto);

ov::Any unpack_tensor_proto(const ::tensorflow::TensorProto& tensor_proto,
                            const ::tensorflow::TensorShapeProto& tensor_shape,
                            const ::tensorflow::DataType& tensor_type);

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
