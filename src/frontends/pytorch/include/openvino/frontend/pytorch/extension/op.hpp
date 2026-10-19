// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "openvino/frontend/extension/op.hpp"
#include "openvino/frontend/pytorch/extension/conversion.hpp"

namespace ov::frontend::pytorch {

template <typename OVOpType = void>
using OpExtension = ov::frontend::OpExtensionBase<ov::frontend::pytorch::ConversionExtension, OVOpType>;

}  // namespace ov::frontend::pytorch
