// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ref_normalize.hpp"
#include <selective_build.h>
#include "utils/bfloat16.hpp"
#include "ie_parallel.hpp"

namespace ov {
namespace intel_cpu {
// *=============* Reference case *===============*
RefNormalizeL2Executor::RefNormalizeL2Executor(const ExecutorContext::CPtr context) : NormalizeL2Executor(context) {}

bool RefNormalizeL2Executor::init(const NormalizeL2Attrs& normalizeL2Attrs,
                                  const std::vector<MemoryDescPtr>& srcDescs,
                                  const std::vector<MemoryDescPtr>& dstDescs,
                                  const dnnl::primitive_attr &attr) {
    execCtx.normalizeL2Attrs = normalizeL2Attrs;
    execCtx.workAmount = std::accumulate(normalizeL2Attrs.vectorDims.begin(),
                                         normalizeL2Attrs.vectorDims.end(), 1, std::multiplies<size_t>());
    execCtx.kernel_attrs = attr;

    if (execCtx.normalizeL2Attrs.layout != LayoutType::ncsp) {
        IE_THROW() << "Reference Executor of 'NormalizeL2' supports only ncsp layout!";
    }

    const auto &p = (*execCtx.kernel_attrs.get()).post_ops_;
    for (int i = 0; i < p.len(); i++) {
        auto &post_op = p.entry_[i];
        if (post_op.is_eltwise()) {
            execCtx.eltwise_injectors_ref.push_back(std::make_shared<dnnl::impl::cpu::ref_eltwise_scalar_fwd_t>(
                    post_op.eltwise.alg, post_op.eltwise.alpha, post_op.eltwise.beta, post_op.eltwise.scale));
        } else if (post_op.is_depthwise()) {
            execCtx.depthwise_injectors_ref.push_back(std::make_shared<dnnl::impl::cpu::ref_depthwise_scalar_fwd_t>(post_op.depthwise.alg));
        }
    }
    return true;
}

void RefNormalizeL2Executor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const void **post_ops_data_) {
    execCtx.src_ptr = reinterpret_cast<const uint8_t *>(src[0]->GetPtr());
    execCtx.dst_ptr = reinterpret_cast<uint8_t *>(dst[0]->GetPtr());
    execCtx.post_ops_data = post_ops_data_;
    OV_SWITCH(intel_cpu, FunctionExecutorCreation, execCtx, std::tie(execCtx.normalizeL2Attrs.input_prec, execCtx.normalizeL2Attrs.output_prec),
              OV_CASE2(InferenceEngine::Precision::U8, InferenceEngine::Precision::U8, uint8_t, uint8_t),
              OV_CASE2(InferenceEngine::Precision::I8, InferenceEngine::Precision::U8, int8_t, uint8_t),
              OV_CASE2(InferenceEngine::Precision::FP32, InferenceEngine::Precision::U8, float, uint8_t),
              OV_CASE2(InferenceEngine::Precision::U8, InferenceEngine::Precision::I8, uint8_t, int8_t),
              OV_CASE2(InferenceEngine::Precision::I8, InferenceEngine::Precision::I8, int8_t, int8_t),
              OV_CASE2(InferenceEngine::Precision::FP32, InferenceEngine::Precision::I8, float, int8_t),
              OV_CASE2(InferenceEngine::Precision::U8, InferenceEngine::Precision::FP32, uint8_t, float),
              OV_CASE2(InferenceEngine::Precision::I8, InferenceEngine::Precision::FP32, int8_t, float),
              OV_CASE2(InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP32, float, float),
              OV_CASE2(InferenceEngine::Precision::BF16, InferenceEngine::Precision::BF16, bfloat16_t, bfloat16_t));
}

template <typename in_data_t, typename out_data_t>
void RefNormalizeL2Executor::normalize(const in_data_t* src_data, out_data_t* dst_data, size_t workAmount) {
    parallel_for(workAmount, [&](size_t i) {
        dst_data[i] = src_data[i] == 0 ? 0 : 1;
    });
}

template <typename in_data_t, typename out_data_t>
void RefNormalizeL2Executor::normalize_nchw_ref(const in_data_t* src_data, out_data_t* dst_data,
                                                const void **post_ops_data, NormalizeL2Attrs attrs,
                                                dnnl::primitive_attr kernel_attrs,
                                                ref_eltwise_scalar_fwd_t eltwise_injectors_ref,
                                                ref_depthwise_scalar_fwd_t depthwise_injectors_ref) {
    auto dims = attrs.vectorDims;
    size_t dims_size = dims.size();
    const size_t N = dims[0];
    const size_t C = dims[1];
    const size_t H = (dims_size > 2) ? dims[2] : 1lu;
    const size_t W = (dims_size > 3) ? dims[3] : 1lu;
    const size_t spatial_dims = H * W;
    for (size_t b = 0lu; b < N; b++) {
        const in_data_t *src_data_b = src_data + b * C * spatial_dims;
        out_data_t *dst_data_b = dst_data + b * C * spatial_dims;
        if (attrs.across_spatial) {
            // modulo
            float addition_identity = 0.0f;
            float modulo = 0.0f;
            modulo = parallel_sum(C, addition_identity, [&](int ic) -> float {
                const in_data_t *src_data_bc = src_data_b + ic * spatial_dims;
                float modulo_c = 0.0f;
                for (size_t m = 0; m < spatial_dims; m++) {
                    modulo_c += src_data_bc[m] * src_data_bc[m];
                }
                return modulo_c;
            });

            float modulo_inv = 1.0f / (std::sqrt(epsApply(modulo, attrs.epsMode, attrs.eps)));

            // normalize
            parallel_for(C, [&](size_t ic) {
                const in_data_t *src_data_bc = src_data_b + ic * spatial_dims;
                out_data_t *dst_data_bc = dst_data_b + ic * spatial_dims;
                for (size_t m = 0; m < spatial_dims; m++) {
                    float dst_value = src_data_bc[m] * modulo_inv;
                    apply_post_ops_scalar(dst_value, ic, post_ops_data, kernel_attrs, eltwise_injectors_ref, depthwise_injectors_ref, attrs);
                    if (attrs.output_prec == InferenceEngine::Precision::U8) {
                        dst_data_bc[m] = (dst_value >= 0) ? dst_value : 0;
                    } else {
                        dst_data_bc[m] = dst_value;
                    }
                }
            });
        } else {  // across_spatial: false
            // moduloM
            std::vector<float> moduloM(spatial_dims, 0.f);
            parallel_for(H, [&](size_t ih) {
                size_t offset_h = ih * W;
                const in_data_t *src_data_b_ih = src_data_b + offset_h;
                for (size_t c = 0; c < C; c++) {
                    const in_data_t *src_data_b_ih_c = src_data_b_ih + spatial_dims * c;
                    for (size_t w = 0; w < W; w++) {
                        moduloM[offset_h + w] += src_data_b_ih_c[w] * src_data_b_ih_c[w];
                    }
                }
            });

            for (size_t m = 0; m < spatial_dims; m++) {
                moduloM[m] = 1.0f / (std::sqrt(epsApply(moduloM[m], attrs.epsMode, attrs.eps)));
            }

            // normalize
            parallel_for(C, [&](size_t ic) {
                const in_data_t *src_data_bc = src_data_b + ic * spatial_dims;
                out_data_t *dst_data_bc = dst_data_b + ic * spatial_dims;
                for (size_t m = 0; m < spatial_dims; m++) {
                    float dst_value = src_data_bc[m] * moduloM[m];
                    apply_post_ops_scalar(dst_value, ic, post_ops_data, kernel_attrs, eltwise_injectors_ref, depthwise_injectors_ref, attrs);
                    if (attrs.output_prec == InferenceEngine::Precision::U8) {
                        dst_data_bc[m] = (dst_value >= 0) ? dst_value : 0;
                    } else {
                        dst_data_bc[m] = dst_value;
                    }
                }
            });
        }
    }
}

void RefNormalizeL2Executor::apply_post_ops_scalar(float &dst_value, int index_c, const void **post_ops_data_,
                                                   dnnl::primitive_attr kernel_attrs,
                                                   ref_eltwise_scalar_fwd_t eltwise_injectors_ref,
                                                   ref_depthwise_scalar_fwd_t depthwise_injectors_ref,
                                                   NormalizeL2Attrs normalizeL2Attrs) {
    const auto &p = (*kernel_attrs.get()).post_ops_;
    int eltwise_inj_idx = 0;
    int depthwise_inj_idx = 0;
    // reinterpret cast from (pointer to const void) to (pointer to const pointer to const float)
    const float** post_ops_data = reinterpret_cast<const float**>(post_ops_data_);
    for (int i = 0; i < p.len(); i++) {
        auto &post_op = p.entry_[i];
        if (post_op.is_eltwise()) {
            dst_value = eltwise_injectors_ref[eltwise_inj_idx]->compute_scalar(dst_value);
            eltwise_inj_idx++;
        } else if (post_op.is_depthwise()) {
            auto depthwise_base = *post_ops_data;
            auto depthwise_weights = depthwise_base + post_op.depthwise.offset[post_op.depthwise.scales] + index_c;
            auto depthwise_bias = depthwise_base + post_op.depthwise.offset[post_op.depthwise.shifts] + index_c;

            dst_value = depthwise_injectors_ref[depthwise_inj_idx]->compute_scalar(dst_value, depthwise_weights, depthwise_bias);

            depthwise_inj_idx++;
            post_ops_data++;
        } else if (post_op.is_quantization()) {
            bool do_dequantization = post_op.quantization.alg == dnnl::impl::alg_kind::quantization_quantize_dequantize;
            bool do_rounding = do_dequantization || normalizeL2Attrs.output_prec == InferenceEngine::Precision::FP32 || i != p.len() - 1;

            auto quant = post_op.quantization;

            using quantization_fields = dnnl::impl::post_ops_t::entry_t::quantization_t::quantization_fields;
            auto dataVal = [&](const quantization_fields& field) -> float {
                auto dataPtr = *post_ops_data + post_op.quantization.offset[field];
                const int channelIdx = quant.per_channel[field] ? index_c : 0;
                return dataPtr[channelIdx];
            };

            float crop_low = dataVal(quant.crop_low);
            float crop_high = dataVal(quant.crop_high);
            float input_scale = dataVal(quant.inp_scale);
            float input_shift = dataVal(quant.inp_shift);

            dst_value = dnnl::impl::nstl::min(crop_high, dnnl::impl::nstl::max(crop_low, dst_value));
            dst_value = dst_value * input_scale + input_shift;

            if (do_rounding) {
                dst_value = roundf(dst_value);
            }

            if (do_dequantization) {
                float output_scale = dataVal(quant.output_scale);
                float output_shift = dataVal(quant.output_shift);
                dst_value = dst_value * output_scale + output_shift;
            }

            post_ops_data++;
        }
    }
}
}   // namespace intel_cpu
}   // namespace ov