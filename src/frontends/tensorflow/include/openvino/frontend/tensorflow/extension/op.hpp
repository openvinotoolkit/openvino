// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "openvino/frontend/extension/op.hpp"
#include "openvino/frontend/tensorflow/extension/conversion.hpp"

namespace ov::frontend::tensorflow {

template <typename OVOpType = void>
using OpExtension = ov::frontend::OpExtensionBase<ov::frontend::tensorflow::ConversionExtension, OVOpType>;

}  // namespace ov::frontend::tensorflow
