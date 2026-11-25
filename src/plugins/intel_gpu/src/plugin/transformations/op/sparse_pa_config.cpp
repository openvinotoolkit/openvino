// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/sparse_pa_config.hpp"

#include <string>

#include "qsa_op_utils.hpp"

namespace ov::intel_gpu::op {

using namespace qsa_util;

bool SparsePAConfig::validate(std::string& reason) const {
    if (!validate_sparse_attention_config(*this, reason)) {
        return false;
    }
    if (page_size != 0 && page_size % key_block_size != 0) {
        reason = "key_block_size must divide the page size";
        return false;
    }
    if (kv_quantized()) {
        if (!validate_kv_quant(key_quant, true, page_size, k_head_size, key_block_size, reason) ||
            !validate_kv_quant(value_quant, false, page_size, v_head_size, key_block_size, reason)) {
            return false;
        }
    }
    return true;
}

}  // namespace ov::intel_gpu::op
