// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transpose.hpp"

#include <utility>
#include <vector>

#include "openvino/core/parallel.hpp"

namespace ov::intel_cpu {

TransposeExecutor::TransposeExecutor(ExecutorContext::CPtr context) : context(std::move(context)) {}

jit_permute_config_params TransposeExecutor::prepareParams(const PermuteParams& params) {
    jit_permute_config_params jcp = {};
    VectorDims src_block_order = params.src_block_order;
    VectorDims src_block_strides(params.src_block_dims.size(), 1);
    VectorDims dst_block_strides(params.dst_block_dims.size(), 1);
    for (int i = params.src_block_dims.size() - 2; i >= 0; i--) {
        src_block_strides[i] = src_block_strides[i + 1] * params.src_block_dims[i + 1];
    }
    for (int i = params.dst_block_dims.size() - 2; i >= 0; i--) {
        dst_block_strides[i] = dst_block_strides[i + 1] * params.dst_block_dims[i + 1];
    }

    VectorDims new_dst_block_strides = dst_block_strides;
    VectorDims new_dst_block_order = params.dst_block_order;
    VectorDims new_dst_block_dims = params.dst_block_dims;
    VectorDims new_src_block_strides(dst_block_strides.size());
    VectorDims mask(dst_block_strides.size());

    VectorDims tmp_order;
    for (size_t i = 0; i < params.dst_block_order.size(); i++) {
        tmp_order.push_back(params.order[params.dst_block_order[i]]);
    }

    for (int i = tmp_order.size() - 1; i >= 0; i--) {
        int pos = std::distance(std::find(src_block_order.rbegin(), src_block_order.rend(), tmp_order[i]),
                                src_block_order.rend() - 1);
        if (pos != -1) {
            new_src_block_strides[i] = src_block_strides[pos];
            src_block_order.erase(src_block_order.begin() + pos);
            src_block_strides.erase(src_block_strides.begin() + pos);
            mask[i] = 0;
        } else {
            new_src_block_strides[i] =
                new_src_block_strides[tmp_order.size() - 1] * params.dst_block_dims[tmp_order.size() - 1];
            mask[i] = 1;
            mask[tmp_order.size() - 1] = 1;
        }
    }
    if (!src_block_order.empty()) {
        int pos = std::distance(tmp_order.begin(), std::find(tmp_order.begin(), tmp_order.end(), src_block_order[0]));
        new_src_block_strides.insert(new_src_block_strides.begin() + pos, src_block_strides[0]);
        new_dst_block_strides.insert(
            new_dst_block_strides.begin() + pos,
            new_dst_block_strides[pos] * params.src_block_dims[params.src_block_dims.size() - 1]);
        new_dst_block_order.insert(new_dst_block_order.begin() + pos, new_dst_block_order[pos]);
        new_dst_block_dims.insert(new_dst_block_dims.begin() + pos + 1,
                                  params.src_block_dims[params.src_block_dims.size() - 1]);
        new_dst_block_dims[pos] = div_up(new_dst_block_dims[pos], new_dst_block_dims[pos + 1]);
        mask.insert(mask.begin() + pos + 1, 1);
        mask[pos] = 1;
    }

    VectorDims sorted_src_strides;
    VectorDims sorted_dst_strides;
    VectorDims sorted_order;
    VectorDims sorted_dst_dims;

    //  support dynamic batch
    int batch_ord = std::distance(params.order.begin(), std::find(params.order.begin(), params.order.end(), 0));
    int batch_count = 0;
    int batch_pos = 0;
    for (size_t i = 0; i < new_dst_block_order.size(); i++) {
        if (static_cast<int>(new_dst_block_order[i]) == batch_ord) {
            batch_count++;
            batch_pos = i;
        }
    }
    if (batch_count == 1) {
        sorted_src_strides.push_back(new_src_block_strides[batch_pos]);
        sorted_dst_strides.push_back(new_dst_block_strides[batch_pos]);
        sorted_order.push_back(new_dst_block_order[batch_pos]);
        sorted_dst_dims.push_back(new_dst_block_dims[batch_pos]);
        jcp.supported_dynamic_batch = true;
    }

    int n2 = 0;
    for (size_t i = 0; i < mask.size(); i++) {
        if (mask[i] == 0) {
            n2++;
            if (batch_count == 1 && static_cast<int>(new_dst_block_order[i]) == batch_ord) {
                continue;
            }
            sorted_src_strides.push_back(new_src_block_strides[i]);
            sorted_dst_strides.push_back(new_dst_block_strides[i]);
            sorted_order.push_back(new_dst_block_order[i]);
            sorted_dst_dims.push_back(new_dst_block_dims[i]);
        }
    }
    for (size_t i = 0; i < mask.size(); i++) {
        if (mask[i] == 1) {
            sorted_src_strides.push_back(new_src_block_strides[i]);
            sorted_dst_strides.push_back(new_dst_block_strides[i]);
            sorted_order.push_back(new_dst_block_order[i]);
            sorted_dst_dims.push_back(new_dst_block_dims[i]);
        }
    }

    int max_threads = parallel_get_max_threads();
    const int n_max = 3;  //  max count dims for parallel
    int n = 0;
    int work_amount = sorted_dst_dims[0];
    for (size_t i = 1; i < sorted_dst_dims.size() && n < n_max; i++) {
        n++;
        if (work_amount >= 4 * max_threads) {  //  4 * max_threads is a specially selected value for best performance
            break;
        }
        work_amount *= sorted_dst_dims[i];
    }

    jcp.src_strides = sorted_src_strides;
    jcp.dst_strides = sorted_dst_strides;
    jcp.dst_block_dims = sorted_dst_dims;
    jcp.n = std::min(n, n2);
    jcp.ndims = sorted_order.size();
    jcp.data_size = params.data_size;

    return jcp;
}

}  // namespace ov::intel_cpu
