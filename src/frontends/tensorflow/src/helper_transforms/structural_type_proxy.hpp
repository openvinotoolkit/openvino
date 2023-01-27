// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Purpose of this file is to define help proxies to represent structural data types that
// may be represented with multiple tensors in lowered-model. For example, tensor of strings
// can be represented as a single tensor if shape is scalar or as 3 tensors with it is not a scalar.
// Ragged tensors add two additional tensors to store indices for a ragged dimension. And so on --
// the exact representation depends on the type and will be defined explicitly.

// Helpers are defined as templated classes because they are used for description classes and for real values.

#pragma once

#include "openvino/core/type/non_tensor_type.hpp"
#include "openvino/frontend/tensorflow/visibility.hpp"


namespace ov {
namespace frontend {
namespace tensorflow {

}
}
}
