/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/
#include "cpu/binary_injector_utils.hpp"
#include "common/primitive.hpp"
#include "common/primitive_attr.hpp"
#include "oneapi/dnnl/dnnl_types.h"

namespace dnnl {
namespace impl {
namespace cpu {
namespace binary_injector_utils {

std::vector<const void *> prepare_binary_args(const post_ops_t &post_ops,
        const exec_ctx_t &ctx, const unsigned first_arg_idx_offset) {
    std::vector<const void *> post_ops_binary_rhs_arg_vec;
    post_ops_binary_rhs_arg_vec.reserve(post_ops.entry_.size());

    unsigned idx = first_arg_idx_offset;
    for (const auto &post_op : post_ops.entry_) {
        if (post_op.is_binary() || post_op.is_depthwise() || post_op.is_quantization()) {
            post_ops_binary_rhs_arg_vec.emplace_back(CTX_IN_MEM(const void *,
                    DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | DNNL_ARG_SRC_1));
        }
        ++idx;
    }

    post_ops_binary_rhs_arg_vec.shrink_to_fit();

    return post_ops_binary_rhs_arg_vec;
}

bool bcast_strategy_present(
        const std::vector<broadcasting_strategy_t> &post_ops_bcasts,
        const broadcasting_strategy_t bcast_strategy) {
    for (const auto &post_op_bcast : post_ops_bcasts)
        if (post_op_bcast == bcast_strategy) return true;
    return false;
}

std::vector<broadcasting_strategy_t> extract_bcast_strategies(
        const std::vector<dnnl_post_ops::entry_t> &post_ops,
        const memory_desc_wrapper &dst_md) {
    std::vector<broadcasting_strategy_t> post_ops_bcasts;
    post_ops_bcasts.reserve(post_ops.size());
    for (const auto &post_op : post_ops)
        if (post_op.is_binary())
            post_ops_bcasts.emplace_back(get_rhs_arg_broadcasting_strategy(
                    post_op.binary.src1_desc, dst_md));
    return post_ops_bcasts;
}

} // namespace binary_injector_utils
} // namespace cpu
} // namespace impl
} // namespace dnnl
