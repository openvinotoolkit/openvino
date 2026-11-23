// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/sparse_pa.hpp"

#include <string>

#include "qsa_op_utils.hpp"

namespace ov::intel_gpu::op {

using namespace qsa_util;

bool SparsePA::validate_config(const Config& c, std::string& reason) {
    if (!validate_sparse_attention_config(c, reason)) {
        return false;
    }
    if (c.page_size != 0 && c.page_size % c.key_block_size != 0) {
        reason = "key_block_size must divide the page size";
        return false;
    }
    if (c.kv_quantized()) {
        if (!validate_kv_quant(c.key_quant, true, c.page_size, c.k_head_size, c.key_block_size, reason) ||
            !validate_kv_quant(c.value_quant, false, c.page_size, c.v_head_size, c.key_block_size, reason)) {
            return false;
        }
    }
    return true;
}

}  // namespace ov::intel_gpu::op
