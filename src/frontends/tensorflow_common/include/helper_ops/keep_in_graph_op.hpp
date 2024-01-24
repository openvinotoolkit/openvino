// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "helper_ops/internal_operation.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

class OPENVINO_API KeepInGraphOp : public InternalOperation {
public:
    OPENVINO_OP("KeepInGraphOp", "ov::frontend::tensorflow", InternalOperation);

    KeepInGraphOp() = default;

    KeepInGraphOp(const std::string& op_type_name, const std::shared_ptr<DecoderBase>& decoder = nullptr)
        : InternalOperation(decoder, OutputVector{}, 1, op_type_name) {}
};
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
