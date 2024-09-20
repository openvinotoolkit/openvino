// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/descriptor/tensor.hpp"

namespace ov {
namespace descriptor {

// To change Tensor element type please change the Parameter type.
OPENVINO_API
void set_element_type(Tensor& tensor, const element::Type& elemenet_type);

// To change Tensor type please change the Parameter type.
OPENVINO_API
void set_tensor_type(Tensor& tensor, const element::Type& element_type, const PartialShape& pshape);

OPENVINO_DEPRECATED("get_ov_tensor_legacy_name() is deprecated. Please don't use this function.")
OPENVINO_API
std::string get_ov_tensor_legacy_name(const Tensor& tensor);

OPENVINO_DEPRECATED("set_ov_tensor_legacy_name() is deprecated. Please don't use this function.")
OPENVINO_API
void set_ov_tensor_legacy_name(Tensor& tensor, const std::string& tensor_name);

}  // namespace descriptor
}  // namespace ov
