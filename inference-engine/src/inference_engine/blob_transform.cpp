// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_detector.hpp"
#include "blob_transform.hpp"
#ifdef HAVE_SSE
#include "blob_transform_sse42.hpp"
#endif

#include <cstdint>
#include <cstdlib>

//----------------------------------------------------------------------

namespace InferenceEngine {

template <InferenceEngine::Precision::ePrecision PRC>
static void blob_copy_4d_t(Blob::Ptr src, Blob::Ptr dst) {
    using data_t = typename InferenceEngine::PrecisionTrait<PRC>::value_type;

    auto *src_ptr = src->buffer().as<data_t*>();
    auto *dst_ptr = dst->buffer().as<data_t*>();

    SizeVector dims = src->getTensorDesc().getDims();

    size_t N = dims[0];
    size_t C = dims[1];
    size_t H = dims[2];
    size_t W = dims[3];

    const Layout src_l = src->layout();
    const auto &src_blk_dsc = src->getTensorDesc().getBlockingDesc();
    const auto &src_strides = src_blk_dsc.getStrides();
    const auto N_src_stride = src_strides[0];
    const auto C_src_stride = src_l == NHWC ? src_strides[3] : src_strides[1];
    const auto H_src_stride = src_l == NHWC ? src_strides[1] : src_strides[2];
    const auto W_src_stride = src_l == NHWC ? src_strides[2] : src_strides[3];
    src_ptr += src_blk_dsc.getOffsetPadding();

    const Layout dst_l = dst->layout();
    const auto &dst_blk_desc = dst->getTensorDesc().getBlockingDesc();
    const auto &dst_strides = dst_blk_desc.getStrides();
    const auto N_dst_stride = dst_strides[0];
    const auto C_dst_stride = dst_l == NHWC ? dst_strides[3] : dst_strides[1];
    const auto H_dst_stride = dst_l == NHWC ? dst_strides[1] : dst_strides[2];
    const auto W_dst_stride = dst_l == NHWC ? dst_strides[2] : dst_strides[3];

    src_ptr += dst_blk_desc.getOffsetPadding();

#ifdef HAVE_SSE
    if (src->layout() == NHWC && dst->layout() == NCHW && C == 3
        && C_src_stride == 1 && W_src_stride == 3 && W_dst_stride == 1 &&
        with_cpu_x86_sse42()) {
        if (PRC == Precision::U8) {
            blob_copy_4d_split_u8c3(reinterpret_cast<const uint8_t*>(src_ptr),
                                    reinterpret_cast<      uint8_t*>(dst_ptr),
                                    N_src_stride, H_src_stride,
                                    N_dst_stride, H_dst_stride, C_dst_stride,
                                    static_cast<int>(N), static_cast<int>(H),
                                    static_cast<int>(W));
            return;
        }

        if (PRC == Precision::FP32) {
            blob_copy_4d_split_f32c3(reinterpret_cast<const float*>(src_ptr),
                                     reinterpret_cast<      float*>(dst_ptr),
                                     N_src_stride, H_src_stride,
                                     N_dst_stride, H_dst_stride, C_dst_stride,
                                     static_cast<int>(N), static_cast<int>(H),
                                     static_cast<int>(W));
            return;
        }
    }

    if (src->layout() == NCHW && dst->layout() == NHWC && C == 3 &&
        C_dst_stride == 1 && W_dst_stride == 3 && W_src_stride == 1 &&
        with_cpu_x86_sse42()) {
        if (PRC == Precision::U8) {
            blob_copy_4d_merge_u8c3(reinterpret_cast<const uint8_t*>(src_ptr),
                                    reinterpret_cast<      uint8_t*>(dst_ptr),
                                    N_src_stride, H_src_stride, C_src_stride,
                                    N_dst_stride, H_dst_stride,
                                    static_cast<int>(N), static_cast<int>(H),
                                    static_cast<int>(W));
            return;
        }

        if (PRC == Precision::FP32) {
            blob_copy_4d_merge_f32c3(reinterpret_cast<const float*>(src_ptr),
                                     reinterpret_cast<      float*>(dst_ptr),
                                     N_src_stride, H_src_stride, C_src_stride,
                                     N_dst_stride, H_dst_stride,
                                     static_cast<int>(N), static_cast<int>(H),
                                     static_cast<int>(W));
            return;
        }
    }
#endif  // HAVE_SSE

    if (src->layout() == NHWC && dst->layout() == NCHW) {
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                data_t *dst_ptr_l = dst_ptr + n * N_dst_stride + c * C_dst_stride;
                data_t *src_ptr_l = src_ptr + n * N_src_stride + c * C_src_stride;
                for (int h = 0; h < H; h++) {
                    data_t *src_ptr_l_l = src_ptr_l + h*H_src_stride;
                    for (int w = 0; w < W; w++) {
                        *dst_ptr_l = *src_ptr_l_l;
                        src_ptr_l_l += W_src_stride;
                        dst_ptr_l++;
                    }
                }
            }
        }
    } else if (src->layout() == NCHW && dst->layout() == NHWC) {
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                data_t *src_ptr_l = src_ptr + n * N_src_stride + c * C_src_stride;
                data_t *dst_ptr_l = dst_ptr + n * N_dst_stride + c;
                for (int h = 0; h < H; h++) {
                    data_t *src_ptr_l_l = src_ptr_l + h*H_src_stride;
                    for (int w = 0; w < W; w++) {
                        *dst_ptr_l = *src_ptr_l_l;
                        dst_ptr_l += W_dst_stride;
                        src_ptr_l_l++;
                    }
                }
            }
        }
    } else {
        for (int i = 0; i < N*C*H*W; i++) {
            dst_ptr[i] = src_ptr[i];
        }
    }
}

static inline void blob_copy_4d(Blob::Ptr src, Blob::Ptr dst) {
    switch (src->precision()) {
        case Precision::FP32:
        case Precision::I32:
            blob_copy_4d_t<Precision::FP32>(src, dst);
            break;

        case Precision::FP16:
        case Precision::U16:
        case Precision::I16:
            blob_copy_4d_t<Precision::U16>(src, dst);
            break;

        case Precision::U8:
        case Precision::I8:
            blob_copy_4d_t<Precision::U8>(src, dst);
            break;

        default:
            THROW_IE_EXCEPTION << "Unsupported blob transformation for precision " << src->precision();
    }
}

void blob_copy(Blob::Ptr src, Blob::Ptr dst) {
    if (src->buffer() == nullptr)
        THROW_IE_EXCEPTION << "Cannot copy blob data. Source is not allocated.";

    if (dst->buffer() == nullptr)
        THROW_IE_EXCEPTION << "Cannot copy blob data. Destination is not allocated.";

    if (src->precision() != dst->precision())
        THROW_IE_EXCEPTION << "Unimplemented blob transformation from precision "
                           << src->precision() << " to " << src->precision();

    if (src->dims() != dst->dims())
        THROW_IE_EXCEPTION << "Unimplemented blob transformation from different shapes ";

    if (src->dims().size() == 4)
        blob_copy_4d(src, dst);
    else
        THROW_IE_EXCEPTION << "Unimplemented blob transformation. Only 4d supported.";
}

}  // namespace InferenceEngine
