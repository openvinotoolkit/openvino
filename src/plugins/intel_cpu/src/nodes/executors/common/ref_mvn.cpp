// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ref_mvn.hpp"

#include <any>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "cpu_types.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/common/cpu_convert.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/mvn_config.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/element_type.hpp"
#include "post_ops.hpp"

namespace ov::intel_cpu {

// Helper function for ceiling division
static inline size_t div_up(size_t a, size_t b) {
    return (a + b - 1) / b;
}

MVNRefExecutor::MVNRefExecutor(MVNAttrs mvnAttrs, MemoryArgs memory, ExecutorContext::CPtr contextPtr)
    : attrs(std::move(mvnAttrs)),
      memoryArgs(std::move(memory)),
      context(std::move(contextPtr)) {
    // Initialize from constructor parameters
    auto srcDesc = memoryArgs.at(ARG_SRC_0)->getDescPtr();
    auto dstDesc = memoryArgs.at(ARG_DST)->getDescPtr();

    OPENVINO_ASSERT(srcDesc && dstDesc, "Invalid memory descriptors for MVNRefExecutor");

    src_data_size = srcDesc->getPrecision().size();
    dst_data_size = dstDesc->getPrecision().size();
}

bool MVNRefExecutor::supports([[maybe_unused]] const MVNConfig& config) {
    // Reference implementation supports all configurations
    return true;
}

// Static local implementation as requested in review
static void mvn_ref_impl(const MVNAttrs& attrs,
                         const MemoryArgs& memoryArgs,
                         const uint8_t* src_data,
                         uint8_t* dst_data,
                         const VectorDims& shape5d) {
    const auto& src_prc = memoryArgs.at(ARG_SRC_0)->getDesc().getPrecision();
    const auto& dst_prc = memoryArgs.at(ARG_DST)->getDesc().getPrecision();

    const size_t N = shape5d[0];
    const size_t C = shape5d[1];
    const size_t D = shape5d[2];
    const size_t H = shape5d[3];
    const size_t W = shape5d[4];
    const size_t total_size = N * C * D * H * W;

    // Determine block size for blocked layouts
    size_t blk_size = 1;
    if (attrs.layout == mvn_block) {
        // Get block size from memory descriptor
        const auto srcDesc = memoryArgs.at(ARG_SRC_0)->getDescPtr();
        if (srcDesc && srcDesc->hasLayoutType(LayoutType::nCsp16c)) {
            blk_size = 16;
        } else if (srcDesc && srcDesc->hasLayoutType(LayoutType::nCsp8c)) {
            blk_size = 8;
        }
    }

    // Check if we need conversion or can work directly with the data
    const float* src_data_ptr = nullptr;
    float* dst_data_ptr = nullptr;
    std::vector<float> src_float;
    std::vector<float> dst_float;

    if (src_prc == ov::element::f32) {
        // No conversion needed for input
        src_data_ptr = reinterpret_cast<const float*>(src_data);
    } else {
        // Convert input to float for intermediate calculations
        src_float.resize(total_size);
        cpu_convert(src_data, src_float.data(), src_prc, ov::element::f32, total_size);
        src_data_ptr = src_float.data();
    }

    if (dst_prc == ov::element::f32) {
        // No conversion needed for output
        dst_data_ptr = reinterpret_cast<float*>(dst_data);
    } else {
        // We'll need to convert output later
        dst_float.resize(total_size);
        dst_data_ptr = dst_float.data();
    }

    if (attrs.execAcrossChannels_) {
        parallel_for(N, [&](int b) {
            const size_t data_size = C * D * H * W;

            // Pre-calculate strides for better optimization
            const size_t spatial_size = D * H * W;
            const size_t CB = div_up(C, blk_size);  // Number of channel blocks
            const size_t batch_stride = C * spatial_size;

            // Calculate mean
            double mean = 0;
            if (attrs.layout == mvn_planar) {
                // NCDHW/NCHW layout - planar format
                // Optimize memory access with sequential reading
                size_t idx = b * batch_stride;
                for (size_t c = 0; c < C; c++) {
                    const float* src_ptr = &src_data_ptr[idx];
                    for (size_t i = 0; i < spatial_size; i++) {
                        mean += src_ptr[i];
                    }
                    idx += spatial_size;
                }
            } else if (attrs.layout == mvn_block) {
                // Blocked layout (e.g., nChw16c)
                for (size_t cb = 0; cb < CB; cb++) {
                    for (size_t d = 0; d < D; d++) {
                        for (size_t h = 0; h < H; h++) {
                            for (size_t w = 0; w < W; w++) {
                                // Handle channel tail: when C is not multiple of block size
                                size_t c_in_blk = (cb == CB - 1 && C % blk_size != 0) ? (C % blk_size) : blk_size;
                                for (size_t c = 0; c < c_in_blk; c++) {
                                    const size_t idx = b * CB * D * H * W * blk_size + cb * D * H * W * blk_size +
                                                       d * H * W * blk_size + h * W * blk_size + w * blk_size + c;
                                    mean += src_data_ptr[idx];
                                }
                            }
                        }
                    }
                }
            } else {
                // NDHWC/NHWC layout - channel last format
                // Optimize for better cache locality and vectorization
                const size_t b_offset = b * spatial_size * C;
                size_t idx = b_offset;
                for (size_t d = 0; d < D; d++) {
                    for (size_t h = 0; h < H; h++) {
                        for (size_t w = 0; w < W; w++) {
                            // Process C elements sequentially for better cache usage
                            const float* src_ptr = &src_data_ptr[idx];
                            for (size_t c = 0; c < C; c++) {
                                mean += src_ptr[c];
                            }
                            idx += C;
                        }
                    }
                }
            }
            mean /= data_size;

            // Calculate variance (if needed) and normalize
            if (attrs.normalizeVariance_) {
                double variance = 0;
                if (attrs.layout == mvn_planar) {
                    size_t idx = b * batch_stride;
                    const auto mean_f = static_cast<float>(mean);
                    for (size_t c = 0; c < C; c++) {
                        const float* src_ptr = &src_data_ptr[idx];
                        for (size_t i = 0; i < spatial_size; i++) {
                            double diff = src_ptr[i] - mean_f;
                            variance += diff * diff;
                        }
                        idx += spatial_size;
                    }
                } else if (attrs.layout == mvn_block) {
                    // Blocked layout variance calculation
                    for (size_t cb = 0; cb < CB; cb++) {
                        for (size_t d = 0; d < D; d++) {
                            for (size_t h = 0; h < H; h++) {
                                for (size_t w = 0; w < W; w++) {
                                    // Handle channel tail: when C is not multiple of block size
                                    size_t c_in_blk = (cb == CB - 1 && C % blk_size != 0) ? (C % blk_size) : blk_size;
                                    for (size_t c = 0; c < c_in_blk; c++) {
                                        const size_t idx = b * CB * D * H * W * blk_size + cb * D * H * W * blk_size +
                                                           d * H * W * blk_size + h * W * blk_size + w * blk_size + c;
                                        double diff = src_data_ptr[idx] - mean;
                                        variance += diff * diff;
                                    }
                                }
                            }
                        }
                    }
                } else {
                    const size_t b_offset = b * spatial_size * C;
                    size_t idx = b_offset;
                    for (size_t d = 0; d < D; d++) {
                        for (size_t h = 0; h < H; h++) {
                            for (size_t w = 0; w < W; w++) {
                                const float* src_ptr = &src_data_ptr[idx];
                                for (size_t c = 0; c < C; c++) {
                                    double diff = src_ptr[c] - mean;
                                    variance += diff * diff;
                                }
                                idx += C;
                            }
                        }
                    }
                }
                variance /= data_size;

                double sigma = 0.0;
                sigma = attrs.epsMode_ == INSIDE_SQRT ? std::sqrt(variance + attrs.epsValue_)
                                                      : std::sqrt(variance) + attrs.epsValue_;
                const double inv_sigma = 1.0 / sigma;

                // Normalize
                if (attrs.layout == mvn_planar) {
                    size_t idx = b * batch_stride;
                    const auto mean_f = static_cast<float>(mean);
                    const auto inv_sigma_f = static_cast<float>(inv_sigma);
                    for (size_t c = 0; c < C; c++) {
                        float* dst_ptr = &dst_data_ptr[idx];
                        const float* src_ptr = &src_data_ptr[idx];
                        // Vectorizable loop
                        for (size_t i = 0; i < spatial_size; i++) {
                            dst_ptr[i] = (src_ptr[i] - mean_f) * inv_sigma_f;
                        }
                        idx += spatial_size;
                    }
                } else if (attrs.layout == mvn_block) {
                    // Blocked layout normalization
                    for (size_t cb = 0; cb < CB; cb++) {
                        for (size_t d = 0; d < D; d++) {
                            for (size_t h = 0; h < H; h++) {
                                for (size_t w = 0; w < W; w++) {
                                    // Handle channel tail: when C is not multiple of block size
                                    size_t c_in_blk = (cb == CB - 1 && C % blk_size != 0) ? (C % blk_size) : blk_size;
                                    for (size_t c = 0; c < c_in_blk; c++) {
                                        const size_t idx = b * CB * D * H * W * blk_size + cb * D * H * W * blk_size +
                                                           d * H * W * blk_size + h * W * blk_size + w * blk_size + c;
                                        dst_data_ptr[idx] = static_cast<float>((src_data_ptr[idx] - mean) * inv_sigma);
                                    }
                                }
                            }
                        }
                    }
                } else {
                    const size_t b_offset = b * spatial_size * C;
                    size_t idx = b_offset;
                    const auto mean_f = static_cast<float>(mean);
                    const auto inv_sigma_f = static_cast<float>(inv_sigma);
                    for (size_t d = 0; d < D; d++) {
                        for (size_t h = 0; h < H; h++) {
                            for (size_t w = 0; w < W; w++) {
                                float* dst_ptr = &dst_data_ptr[idx];
                                const float* src_ptr = &src_data_ptr[idx];
                                for (size_t c = 0; c < C; c++) {
                                    dst_ptr[c] = (src_ptr[c] - mean_f) * inv_sigma_f;
                                }
                                idx += C;
                            }
                        }
                    }
                }
            } else {
                // Just subtract mean
                if (attrs.layout == mvn_planar) {
                    size_t idx = b * batch_stride;
                    const auto mean_f = static_cast<float>(mean);
                    for (size_t c = 0; c < C; c++) {
                        float* dst_ptr = &dst_data_ptr[idx];
                        const float* src_ptr = &src_data_ptr[idx];
                        // Vectorizable loop
                        for (size_t i = 0; i < spatial_size; i++) {
                            dst_ptr[i] = src_ptr[i] - mean_f;
                        }
                        idx += spatial_size;
                    }
                } else if (attrs.layout == mvn_block) {
                    // Blocked layout - just subtract mean
                    for (size_t cb = 0; cb < CB; cb++) {
                        for (size_t d = 0; d < D; d++) {
                            for (size_t h = 0; h < H; h++) {
                                for (size_t w = 0; w < W; w++) {
                                    // Handle channel tail: when C is not multiple of block size
                                    size_t c_in_blk = (cb == CB - 1 && C % blk_size != 0) ? (C % blk_size) : blk_size;
                                    for (size_t c = 0; c < c_in_blk; c++) {
                                        const size_t idx = b * CB * D * H * W * blk_size + cb * D * H * W * blk_size +
                                                           d * H * W * blk_size + h * W * blk_size + w * blk_size + c;
                                        dst_data_ptr[idx] = static_cast<float>(src_data_ptr[idx] - mean);
                                    }
                                }
                            }
                        }
                    }
                } else {
                    const size_t b_offset = b * spatial_size * C;
                    size_t idx = b_offset;
                    const auto mean_f = static_cast<float>(mean);
                    for (size_t d = 0; d < D; d++) {
                        for (size_t h = 0; h < H; h++) {
                            for (size_t w = 0; w < W; w++) {
                                float* dst_ptr = &dst_data_ptr[idx];
                                const float* src_ptr = &src_data_ptr[idx];
                                // Vectorizable loop
                                for (size_t c = 0; c < C; c++) {
                                    dst_ptr[c] = src_ptr[c] - mean_f;
                                }
                                idx += C;
                            }
                        }
                    }
                }
            }
        });
    } else {
        // Per-channel normalization
        const size_t CB = div_up(C, blk_size);

        if (attrs.layout == mvn_block) {
            // For blocked layout, process by channel blocks
            parallel_for2d(N, CB, [&](int b, int cb) {
                for (size_t c_in_blk = 0; c_in_blk < blk_size && cb * blk_size + c_in_blk < C; c_in_blk++) {
                    const size_t data_size = D * H * W;

                    // Calculate mean for this channel
                    double mean = 0;
                    for (size_t d = 0; d < D; d++) {
                        for (size_t h = 0; h < H; h++) {
                            for (size_t w = 0; w < W; w++) {
                                const size_t idx = b * CB * D * H * W * blk_size + cb * D * H * W * blk_size +
                                                   d * H * W * blk_size + h * W * blk_size + w * blk_size + c_in_blk;
                                mean += src_data_ptr[idx];
                            }
                        }
                    }
                    mean /= data_size;

                    // Calculate variance (if needed) and normalize
                    if (attrs.normalizeVariance_) {
                        double variance = 0;
                        for (size_t d = 0; d < D; d++) {
                            for (size_t h = 0; h < H; h++) {
                                for (size_t w = 0; w < W; w++) {
                                    const size_t idx = b * CB * D * H * W * blk_size + cb * D * H * W * blk_size +
                                                       d * H * W * blk_size + h * W * blk_size + w * blk_size +
                                                       c_in_blk;
                                    double diff = src_data_ptr[idx] - mean;
                                    variance += diff * diff;
                                }
                            }
                        }
                        variance /= data_size;

                        double sigma = 0.0;
                        sigma = attrs.epsMode_ == INSIDE_SQRT ? std::sqrt(variance + attrs.epsValue_)
                                                              : std::sqrt(variance) + attrs.epsValue_;
                        const double inv_sigma = 1.0 / sigma;

                        // Normalize
                        for (size_t d = 0; d < D; d++) {
                            for (size_t h = 0; h < H; h++) {
                                for (size_t w = 0; w < W; w++) {
                                    const size_t idx = b * CB * D * H * W * blk_size + cb * D * H * W * blk_size +
                                                       d * H * W * blk_size + h * W * blk_size + w * blk_size +
                                                       c_in_blk;
                                    dst_data_ptr[idx] = static_cast<float>((src_data_ptr[idx] - mean) * inv_sigma);
                                }
                            }
                        }
                    } else {
                        // Just subtract mean
                        for (size_t d = 0; d < D; d++) {
                            for (size_t h = 0; h < H; h++) {
                                for (size_t w = 0; w < W; w++) {
                                    const size_t idx = b * CB * D * H * W * blk_size + cb * D * H * W * blk_size +
                                                       d * H * W * blk_size + h * W * blk_size + w * blk_size +
                                                       c_in_blk;
                                    dst_data_ptr[idx] = static_cast<float>(src_data_ptr[idx] - mean);
                                }
                            }
                        }
                    }
                }
            });
        } else {
            // Planar and channel-last layouts
            parallel_for2d(N, C, [&](int b, int c) {
                const size_t data_size = D * H * W;

                // Calculate mean
                double mean = 0;
                if (attrs.layout == mvn_planar) {
                    // NCDHW/NCHW layout - planar format
                    const size_t c_offset = (b * C * data_size) + (c * data_size);
                    const float* src_ptr = &src_data_ptr[c_offset];
                    // Sequential access for better cache usage
                    for (size_t i = 0; i < data_size; i++) {
                        mean += src_ptr[i];
                    }
                } else {
                    // NDHWC/NHWC layout - channel last format
                    // Optimize memory access pattern
                    const size_t b_offset = b * data_size * C;
                    size_t base_idx = b_offset + c;
                    for (size_t d = 0; d < D; d++) {
                        for (size_t h = 0; h < H; h++) {
                            for (size_t w = 0; w < W; w++) {
                                mean += src_data_ptr[base_idx];
                                base_idx += C;
                            }
                        }
                    }
                }
                mean /= data_size;

                // Calculate variance (if needed) and normalize
                if (attrs.normalizeVariance_) {
                    double variance = 0;
                    if (attrs.layout == mvn_planar) {
                        const size_t c_offset = (b * C * data_size) + (c * data_size);
                        const float* src_ptr = &src_data_ptr[c_offset];
                        for (size_t i = 0; i < data_size; i++) {
                            double diff = src_ptr[i] - mean;
                            variance += diff * diff;
                        }
                    } else {
                        const size_t b_offset = b * data_size * C;
                        size_t base_idx = b_offset + c;
                        for (size_t d = 0; d < D; d++) {
                            for (size_t h = 0; h < H; h++) {
                                for (size_t w = 0; w < W; w++) {
                                    double diff = src_data_ptr[base_idx] - mean;
                                    variance += diff * diff;
                                    base_idx += C;
                                }
                            }
                        }
                    }
                    variance /= data_size;

                    double sigma = 0.0;
                    sigma = attrs.epsMode_ == INSIDE_SQRT ? std::sqrt(variance + attrs.epsValue_)
                                                          : std::sqrt(variance) + attrs.epsValue_;
                    const double inv_sigma = 1.0 / sigma;

                    if (attrs.layout == mvn_planar) {
                        const size_t c_offset = (b * C * data_size) + (c * data_size);
                        float* dst_ptr = &dst_data_ptr[c_offset];
                        const float* src_ptr = &src_data_ptr[c_offset];
                        for (size_t i = 0; i < data_size; i++) {
                            dst_ptr[i] = static_cast<float>((src_ptr[i] - mean) * inv_sigma);
                        }
                    } else {
                        const size_t b_offset = b * data_size * C;
                        size_t base_idx = b_offset + c;
                        for (size_t d = 0; d < D; d++) {
                            for (size_t h = 0; h < H; h++) {
                                for (size_t w = 0; w < W; w++) {
                                    dst_data_ptr[base_idx] =
                                        static_cast<float>((src_data_ptr[base_idx] - mean) * inv_sigma);
                                    base_idx += C;
                                }
                            }
                        }
                    }
                } else {
                    if (attrs.layout == mvn_planar) {
                        const size_t c_offset = (b * C * data_size) + (c * data_size);
                        float* dst_ptr = &dst_data_ptr[c_offset];
                        const float* src_ptr = &src_data_ptr[c_offset];
                        for (size_t i = 0; i < data_size; i++) {
                            dst_ptr[i] = static_cast<float>(src_ptr[i] - mean);
                        }
                    } else {
                        const size_t b_offset = b * data_size * C;
                        size_t base_idx = b_offset + c;  // write only current channel c
                        for (size_t d = 0; d < D; d++) {
                            for (size_t h = 0; h < H; h++) {
                                for (size_t w = 0; w < W; w++) {
                                    dst_data_ptr[base_idx] = static_cast<float>(src_data_ptr[base_idx] - mean);
                                    base_idx += C;
                                }
                            }
                        }
                    }
                }
            });
        }
    }

    // Apply post-ops (ScaleShift) in FP32 domain
    if (!attrs.postOps.empty()) {
        // Extract first ScaleShift if present
        for (const auto& postOpAny : attrs.postOps) {
            try {
                const auto& ss = std::any_cast<const ScaleShiftPostOp&>(postOpAny);
                const auto& scales = ss.scales();
                const auto& shifts = ss.shifts();

                const size_t N = shape5d[0];
                const size_t C = shape5d[1];
                const size_t D = shape5d[2];
                const size_t H = shape5d[3];
                const size_t W = shape5d[4];
                const size_t spatial = D * H * W;

                const bool scalar = scales.size() == 1 && shifts.size() == 1;

                if (attrs.layout == mvn_planar) {
                    for (size_t b = 0; b < N; ++b) {
                        size_t base = b * C * spatial;
                        for (size_t c = 0; c < C; ++c) {
                            const float sc = scalar ? scales[0] : scales[c % scales.size()];
                            const float sh = scalar ? shifts[0] : shifts[c % shifts.size()];
                            float* dst_ptr = &dst_data_ptr[base + c * spatial];
                            for (size_t i = 0; i < spatial; ++i) {
                                dst_ptr[i] = dst_ptr[i] * sc + sh;
                            }
                        }
                    }
                } else if (attrs.layout == mvn_by_channel) {
                    // NDHWC/NHWC layout
                    for (size_t b = 0; b < N; ++b) {
                        size_t idx = b * spatial * C;
                        for (size_t d = 0; d < D; ++d) {
                            for (size_t h = 0; h < H; ++h) {
                                for (size_t w = 0; w < W; ++w) {
                                    float* dst_ptr = &dst_data_ptr[idx];
                                    for (size_t c = 0; c < C; ++c) {
                                        const float sc = scalar ? scales[0] : scales[c % scales.size()];
                                        const float sh = scalar ? shifts[0] : shifts[c % shifts.size()];
                                        dst_ptr[c] = dst_ptr[c] * sc + sh;
                                    }
                                    idx += C;
                                }
                            }
                        }
                    }
                } else {
                    // Blocked layout: apply per-element with proper block size
                    const size_t CB = div_up(C, blk_size);
                    for (size_t b = 0; b < N; ++b) {
                        for (size_t cb = 0; cb < CB; ++cb) {
                            const size_t c_in_blk = (cb == CB - 1 && C % blk_size != 0) ? (C % blk_size) : blk_size;
                            for (size_t d = 0; d < D; ++d) {
                                for (size_t h = 0; h < H; ++h) {
                                    for (size_t w = 0; w < W; ++w) {
                                        for (size_t c = 0; c < c_in_blk; ++c) {
                                            const size_t c_idx = cb * blk_size + c;
                                            const float sc = scalar ? scales[0] : scales[c_idx % scales.size()];
                                            const float sh = scalar ? shifts[0] : shifts[c_idx % shifts.size()];
                                            const size_t off = b * CB * D * H * W * blk_size +
                                                               cb * D * H * W * blk_size + d * H * W * blk_size +
                                                               h * W * blk_size + w * blk_size + c;
                                            dst_data_ptr[off] = dst_data_ptr[off] * sc + sh;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } catch (const std::bad_any_cast&) {
                // skip non-ScaleShift ops in ref path
                continue;
            }
        }
    }

    // Convert back if needed
    if (dst_prc != ov::element::f32) {
        cpu_convert(dst_data_ptr, dst_data, ov::element::f32, dst_prc, total_size);
    }
}

void MVNRefExecutor::executeImpl(const MemoryArgs& memory) {
    const auto src = memory.at(ARG_SRC_0);
    const auto dst = memory.at(ARG_DST);

    const auto* src_data = src->getDataAs<const uint8_t>();
    auto* dst_data = dst->getDataAs<uint8_t>();

    // Derive 5D shape and effective execAcrossChannels from input dims.
    // Map to a canonical [N, C, D, H, W] regardless of memory layout.
    const auto& dims = src->getStaticDims();
    VectorDims local5D;

    auto map_ncsp = [&](const VectorDims& d) -> VectorDims {
        switch (d.size()) {
        case 0:
            return {1, 1, 1, 1, 1};
        case 1:
            return attrs.initAcrossChannels_ ? VectorDims{1, 1, 1, 1, d[0]} : VectorDims{1, d[0], 1, 1, 1};
        case 2:
            return attrs.initAcrossChannels_ ? VectorDims{1, d[0], 1, d[1], 1} : VectorDims{d[0], d[1], 1, 1, 1};
        case 3:
            return {d[0], d[1], 1, d[2], 1};
        case 4:
            return {d[0], d[1], 1, d[2], d[3]};
        default:
            return {d[0], d[1], d[2], d[3], d[4]};
        }
    };

    auto map_nspc = [&](const VectorDims& d) -> VectorDims {
        // Channel-last: 3D assumed [N, W, C], 4D [N, H, W, C], 5D [N, D, H, W, C]
        switch (d.size()) {
        case 0:
            return {1, 1, 1, 1, 1};
        case 1:
            return attrs.initAcrossChannels_ ? VectorDims{1, 1, 1, 1, d[0]} : VectorDims{1, d[0], 1, 1, 1};
        case 2:
            return attrs.initAcrossChannels_ ? VectorDims{1, d[1], 1, d[0], 1} : VectorDims{d[0], d[1], 1, 1, 1};
        case 3:
            return {d[0], d[2], 1, d[1], 1};
        case 4:
            return {d[0], d[3], 1, d[1], d[2]};
        default:
            return {d[0], d[4], d[1], d[2], d[3]};
        }
    };

    // Map according to layout
    local5D = (attrs.layout == mvn_by_channel) ? map_nspc(dims) : map_ncsp(dims);

    // Effective execAcrossChannels matches node::transformTo5DCase behavior
    MVNAttrs effectiveAttrs = attrs;
    if (dims.size() <= 2 && attrs.initAcrossChannels_) {
        effectiveAttrs.execAcrossChannels_ = false;
    }

    mvn_ref_impl(effectiveAttrs, memoryArgs, src_data, dst_data, local5D);
}

}  // namespace ov::intel_cpu
