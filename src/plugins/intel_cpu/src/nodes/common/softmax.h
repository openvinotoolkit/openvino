// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <memory>

#include "defs.h"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace intel_cpu {

struct jit_uni_softmax_kernel;

static inline void softmax_many_batches(const float* src_data, float* dst_data, int B, int C, int H, int W) {
    ov::parallel_for(B * H * W, [&](size_t i) {
        const float* psrc = src_data + (i / (H * W)) * C * H * W - (i / (H * W)) * H * W;
        float* pdst = dst_data + (i / (H * W)) * C * H * W - (i / (H * W)) * H * W;

        float max = psrc[i];
        for (int c = 0; c < C; c++) {
            float val = psrc[c * H * W + i];
            if (val > max)
                max = val;
        }

        float expSum = 0;
        for (int c = 0; c < C; c++) {
            pdst[c * H * W + i] = std::exp(psrc[c * H * W + i] - max);
            expSum += pdst[c * H * W + i];
        }

        for (int c = 0; c < C; c++) {
            pdst[c * H * W + i] = pdst[c * H * W + i] / expSum;
        }
    });
}

class SoftmaxGeneric {
public:
    SoftmaxGeneric(ov::element::Type inpPrc, ov::element::Type outPrc);

    void execute(const uint8_t* src_data, uint8_t* dst_data, int B, int C, int H, int W);

private:
    template <typename in_data_t, typename out_data_t>
    void calculate(const in_data_t* src_data, out_data_t* dst_data, int B, int C, int H, int W);

private:
    int block_size;
    ov::element::Type input_prec, output_prec;
    std::shared_ptr<jit_uni_softmax_kernel> softmax_kernel;
};

}  // namespace intel_cpu
}  // namespace ov
