// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/opsets/experimental_opset1.hpp"

#include <mutex>

#include "openvino/op/experimental/block_sparse_attention.hpp"
#include "openvino/op/experimental/draft_block_mask.hpp"

namespace ov::op::experimental {

const ov::OpSet& get_opset1() {
    static OpSet opset("experimental_opset1");
    static std::once_flag flag;
    std::call_once(flag, [&]() {
        opset.insert<ov::op::experimental::v1::BlockSparseAttention>();
        opset.insert<ov::op::experimental::v1::DraftBlockMask>();
    });
    return opset;
}

}  // namespace ov::op::experimental