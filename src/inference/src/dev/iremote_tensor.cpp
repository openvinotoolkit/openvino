// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/iremote_tensor.hpp"

#include <memory>

#include "ie_blob.h"
#include "ie_ngraph_utils.hpp"
#include "ie_remote_blob.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {

IRemoteTensor::~IRemoteTensor() = default;

}  // namespace ov
