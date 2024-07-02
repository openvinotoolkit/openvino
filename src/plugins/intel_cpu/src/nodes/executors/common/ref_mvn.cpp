// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ref_mvn.hpp"
#include "openvino/core/parallel.hpp"

namespace ov {
namespace intel_cpu {

MVNRefExecutor::MVNRefExecutor(const MVNAttrs& mvnAttrs):MVNExecutorBase(mvnAttrs) {}

void MVNRefExecutor::exec(const uint8_t *src_data, uint8_t *dst_data, const void *post_ops_data_, const VectorDims& shape5d) {
        mvn_ref(src_data, dst_data, shape5d);
    }

void MVNRefExecutor::mvn_ref(const uint8_t* src_data, uint8_t* dst_data, const VectorDims& shape5d) {
        const float *src_data_ptr = reinterpret_cast<const float *>(src_data);
        float *dst_data_ptr = reinterpret_cast<float *>(dst_data);
        const size_t N = shape5d[0];
        const size_t C = shape5d[1];
        const size_t D = shape5d[2];
        const size_t H = shape5d[3];
        const size_t W = shape5d[4];

        size_t C1 = H * W;
        size_t C2 = C1 * D;
        size_t C3 = C2 * C;

        parallel_for(N, [&](int b) {
            size_t cb = b * C3;
            if (mvnAttrs.execAcrossChannels_) {
                // Parallel sum for each channel for mean
                float C3inv = 1.f / static_cast<float>(C3);
                float mean_temp = 0.0f;

                mean_temp = parallel_sum(C, mean_temp, [&](size_t c)->float {
                    float mean_internal = 0.0f;
                    size_t cc = cb + c * C2;
                    for (size_t sp = 0lu; sp < C2; sp++) {
                        mean_internal += src_data_ptr[cc + sp];
                    }
                    return mean_internal;
                });

                float mean = mean_temp * C3inv;

                if (mvnAttrs.normalizeVariance_) {
                    // parallel sum for each channel for variance
                    float variance_temp = 0.0f;
                    variance_temp = parallel_sum(C, variance_temp, [&](size_t c)->float {
                        float variance_internal = 0.0f;
                        size_t cc = cb + c * C2;
                        for (size_t sp = 0lu; sp < C2; sp++) {
                            variance_internal += (src_data_ptr[cc + sp] - mean) * (src_data_ptr[cc + sp] - mean);
                        }
                        return variance_internal;
                    });

                    float variance = 1.f;
                    if (mvnAttrs.epsMode_ == INSIDE_SQRT)
                        variance = 1.f / sqrtf(variance_temp * C3inv + mvnAttrs.epsValue_);
                    else if (mvnAttrs.epsMode_ == OUTSIDE_SQRT)
                        variance = 1.f / (sqrtf(variance_temp * C3inv) + mvnAttrs.epsValue_);

                    parallel_for(C, [&](int c) {
                        size_t cc = cb + c * C2;
                        for (size_t sp = 0lu; sp < C2; sp++) {
                            dst_data_ptr[cc + sp] = (src_data_ptr[cc + sp] - mean) * variance;
                        }
                    });
                } else {
                    parallel_for(C, [&](int c) {
                        size_t cc = cb + c * C2;
                        for (size_t sp = 0lu; sp < C2; sp++) {
                            dst_data_ptr[cc + sp] = src_data_ptr[cc + sp] - mean;
                        }
                    });
                }
            } else {  // per channel
                float C2inv = 1.f / static_cast<float>(C2);
                parallel_for(C, [&](size_t c) {
                    // mean for this channel
                    float mean = 0.f;
                    size_t cc = cb + c * C2;
                    for (size_t sp = 0lu; sp < C2; sp++) {
                        mean += src_data_ptr[cc + sp];
                    }
                    mean *= C2inv;

                    if (mvnAttrs.normalizeVariance_) {
                        // variance for this channel
                        float variance = 0.f;
                        for (size_t sp = 0lu; sp < C2; sp++) {
                            variance += (src_data_ptr[cc + sp] - mean) * (src_data_ptr[cc + sp] - mean);
                        }

                        if (mvnAttrs.epsMode_ == INSIDE_SQRT)
                            variance = 1.f / sqrtf(variance * C2inv + mvnAttrs.epsValue_);
                        else if (mvnAttrs.epsMode_ == OUTSIDE_SQRT)
                            variance = 1.f / (sqrtf(variance * C2inv) + mvnAttrs.epsValue_);

                        // mvn for this channel
                        for (size_t sp = 0lu; sp < C2; sp++) {
                            dst_data_ptr[cc + sp] = (src_data_ptr[cc + sp] - mean) * variance;
                        }
                    } else {
                        // mvn for this channel
                        for (size_t sp = 0lu; sp < C2; sp++) {
                            dst_data_ptr[cc + sp] = src_data_ptr[cc + sp] - mean;
                        }
                    }
                });
            }
        });
    }

}   // namespace intel_cpu
}   // namespace ov