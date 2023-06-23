// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "blob_transform.hpp"

#include "ie_system_conf.h"
#ifdef HAVE_SSE
#    include "cpu_x86_sse42/blob_transform_sse42.hpp"
#endif

#include <cstdint>
#include <cstdlib>

//----------------------------------------------------------------------

IE_SUPPRESS_DEPRECATED_START

namespace InferenceEngine {

template <InferenceEngine::Precision::ePrecision PRC>
static void blob_copy_4d_t(Blob::Ptr src, Blob::Ptr dst) {
    using data_t = typename InferenceEngine::PrecisionTrait<PRC>::value_type;

    auto* src_ptr = src->buffer().as<data_t*>();
    auto* dst_ptr = dst->buffer().as<data_t*>();

    SizeVector dims = src->getTensorDesc().getDims();

    size_t N = dims[0];
    size_t C = dims[1];
    size_t H = dims[2];
    size_t W = dims[3];

    const Layout src_l = src->getTensorDesc().getLayout();
    const auto& src_blk_dsc = src->getTensorDesc().getBlockingDesc();
    const auto& src_strides = src_blk_dsc.getStrides();
    const auto N_src_stride = src_strides[0];
    const auto C_src_stride = src_l == NHWC ? src_strides[3] : src_strides[1];
    const auto H_src_stride = src_l == NHWC ? src_strides[1] : src_strides[2];
    const auto W_src_stride = src_l == NHWC ? src_strides[2] : src_strides[3];
    src_ptr += src_blk_dsc.getOffsetPadding();

    const Layout dst_l = dst->getTensorDesc().getLayout();
    const auto& dst_blk_desc = dst->getTensorDesc().getBlockingDesc();
    const auto& dst_strides = dst_blk_desc.getStrides();
    const auto N_dst_stride = dst_strides[0];
    const auto C_dst_stride = dst_l == NHWC ? dst_strides[3] : dst_strides[1];
    const auto H_dst_stride = dst_l == NHWC ? dst_strides[1] : dst_strides[2];
    const auto W_dst_stride = dst_l == NHWC ? dst_strides[2] : dst_strides[3];

    dst_ptr += dst_blk_desc.getOffsetPadding();

#ifdef HAVE_SSE
    if (src->getTensorDesc().getLayout() == NHWC && dst->getTensorDesc().getLayout() == NCHW && C == 3 &&
        C_src_stride == 1 && W_src_stride == 3 && W_dst_stride == 1 && with_cpu_x86_sse42()) {
        if (PRC == Precision::U8) {
            blob_copy_4d_split_u8c3(reinterpret_cast<const uint8_t*>(src_ptr),
                                    reinterpret_cast<uint8_t*>(dst_ptr),
                                    N_src_stride,
                                    H_src_stride,
                                    N_dst_stride,
                                    H_dst_stride,
                                    C_dst_stride,
                                    static_cast<int>(N),
                                    static_cast<int>(H),
                                    static_cast<int>(W));
            return;
        }

        if (PRC == Precision::FP32) {
            blob_copy_4d_split_f32c3(reinterpret_cast<const float*>(src_ptr),
                                     reinterpret_cast<float*>(dst_ptr),
                                     N_src_stride,
                                     H_src_stride,
                                     N_dst_stride,
                                     H_dst_stride,
                                     C_dst_stride,
                                     static_cast<int>(N),
                                     static_cast<int>(H),
                                     static_cast<int>(W));
            return;
        }
    }

    if (src->getTensorDesc().getLayout() == NCHW && dst->getTensorDesc().getLayout() == NHWC && C == 3 &&
        C_dst_stride == 1 && W_dst_stride == 3 && W_src_stride == 1 && with_cpu_x86_sse42()) {
        if (PRC == Precision::U8) {
            blob_copy_4d_merge_u8c3(reinterpret_cast<const uint8_t*>(src_ptr),
                                    reinterpret_cast<uint8_t*>(dst_ptr),
                                    N_src_stride,
                                    H_src_stride,
                                    C_src_stride,
                                    N_dst_stride,
                                    H_dst_stride,
                                    static_cast<int>(N),
                                    static_cast<int>(H),
                                    static_cast<int>(W));
            return;
        }

        if (PRC == Precision::FP32) {
            blob_copy_4d_merge_f32c3(reinterpret_cast<const float*>(src_ptr),
                                     reinterpret_cast<float*>(dst_ptr),
                                     N_src_stride,
                                     H_src_stride,
                                     C_src_stride,
                                     N_dst_stride,
                                     H_dst_stride,
                                     static_cast<int>(N),
                                     static_cast<int>(H),
                                     static_cast<int>(W));
            return;
        }
    }
#else
    (void)H_dst_stride;
#endif  // HAVE_SSE

    if (src->getTensorDesc().getLayout() == NHWC && dst->getTensorDesc().getLayout() == NCHW) {
        for (size_t n = 0; n < N; n++) {
            for (size_t c = 0; c < C; c++) {
                data_t* dst_ptr_l = dst_ptr + n * N_dst_stride + c * C_dst_stride;
                data_t* src_ptr_l = src_ptr + n * N_src_stride + c * C_src_stride;
                for (size_t h = 0; h < H; h++) {
                    data_t* src_ptr_l_l = src_ptr_l + h * H_src_stride;
                    for (size_t w = 0; w < W; w++) {
                        *dst_ptr_l = *src_ptr_l_l;
                        src_ptr_l_l += W_src_stride;
                        dst_ptr_l++;
                    }
                }
            }
        }
    } else if (src->getTensorDesc().getLayout() == NCHW && dst->getTensorDesc().getLayout() == NHWC) {
        for (size_t n = 0; n < N; n++) {
            for (size_t c = 0; c < C; c++) {
                data_t* src_ptr_l = src_ptr + n * N_src_stride + c * C_src_stride;
                data_t* dst_ptr_l = dst_ptr + n * N_dst_stride + c;
                for (size_t h = 0; h < H; h++) {
                    data_t* src_ptr_l_l = src_ptr_l + h * H_src_stride;
                    for (size_t w = 0; w < W; w++) {
                        *dst_ptr_l = *src_ptr_l_l;
                        dst_ptr_l += W_dst_stride;
                        src_ptr_l_l++;
                    }
                }
            }
        }
    } else {
        for (size_t i = 0; i < N * C * H * W; i++) {
            dst_ptr[i] = src_ptr[i];
        }
    }
}

static inline void blob_copy_4d(Blob::Ptr src, Blob::Ptr dst) {
    switch (src->getTensorDesc().getPrecision()) {
    case Precision::FP32:
    case Precision::I32:
    case Precision::U32:
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
        IE_THROW() << "Unsupported blob transformation for precision " << src->getTensorDesc().getPrecision();
    }
}

template <InferenceEngine::Precision::ePrecision PRC>
static void blob_copy_5d_t(Blob::Ptr src, Blob::Ptr dst) {
    using data_t = typename InferenceEngine::PrecisionTrait<PRC>::value_type;

    const auto& src_blk_desc = src->getTensorDesc().getBlockingDesc();
    const auto& dst_blk_desc = dst->getTensorDesc().getBlockingDesc();

    data_t* src_ptr = src->buffer().as<data_t*>() + src_blk_desc.getOffsetPadding();
    data_t* dst_ptr = dst->buffer().as<data_t*>() + dst_blk_desc.getOffsetPadding();

    SizeVector dims = src->getTensorDesc().getDims();  // == dst's dims

    const size_t N = dims[0];
    const size_t C = dims[1];
    const size_t D = dims[2];
    const size_t H = dims[3];
    const size_t W = dims[4];

    const Layout src_l = src->getTensorDesc().getLayout();
    const auto& src_strides = src_blk_desc.getStrides();
    const auto N_src_stride = src_strides[0];
    const auto C_src_stride = src_l == NDHWC ? src_strides[4] : src_strides[1];
    const auto D_src_stride = src_l == NDHWC ? src_strides[1] : src_strides[2];
    const auto H_src_stride = src_l == NDHWC ? src_strides[2] : src_strides[3];
    const auto W_src_stride = src_l == NDHWC ? src_strides[3] : src_strides[4];

    const Layout dst_l = dst->getTensorDesc().getLayout();
    const auto& dst_strides = dst_blk_desc.getStrides();
    const auto N_dst_stride = dst_strides[0];
    const auto C_dst_stride = dst_l == NDHWC ? dst_strides[4] : dst_strides[1];
    const auto D_dst_stride = dst_l == NDHWC ? dst_strides[1] : dst_strides[2];
    const auto H_dst_stride = dst_l == NDHWC ? dst_strides[2] : dst_strides[3];
    const auto W_dst_stride = dst_l == NDHWC ? dst_strides[3] : dst_strides[4];

#ifdef HAVE_SSE
    if (src->getTensorDesc().getLayout() == NDHWC && dst->getTensorDesc().getLayout() == NCDHW && C == 3 &&
        C_src_stride == 1 && W_src_stride == 3 && W_dst_stride == 1 && with_cpu_x86_sse42()) {
        if (PRC == Precision::U8) {
            blob_copy_5d_split_u8c3(reinterpret_cast<const uint8_t*>(src_ptr),
                                    reinterpret_cast<uint8_t*>(dst_ptr),
                                    N_src_stride,
                                    D_src_stride,
                                    H_src_stride,
                                    N_dst_stride,
                                    D_dst_stride,
                                    H_dst_stride,
                                    C_dst_stride,
                                    static_cast<int>(N),
                                    static_cast<int>(D),
                                    static_cast<int>(H),
                                    static_cast<int>(W));
            return;
        }

        if (PRC == Precision::FP32) {
            blob_copy_5d_split_f32c3(reinterpret_cast<const float*>(src_ptr),
                                     reinterpret_cast<float*>(dst_ptr),
                                     N_src_stride,
                                     D_src_stride,
                                     H_src_stride,
                                     N_dst_stride,
                                     D_dst_stride,
                                     H_dst_stride,
                                     C_dst_stride,
                                     static_cast<int>(N),
                                     static_cast<int>(D),
                                     static_cast<int>(H),
                                     static_cast<int>(W));
            return;
        }
    }

    if (src->getTensorDesc().getLayout() == NCDHW && dst->getTensorDesc().getLayout() == NDHWC && C == 3 &&
        C_dst_stride == 1 && W_dst_stride == 3 && W_src_stride == 1 && with_cpu_x86_sse42()) {
        if (PRC == Precision::U8) {
            blob_copy_5d_merge_u8c3(reinterpret_cast<const uint8_t*>(src_ptr),
                                    reinterpret_cast<uint8_t*>(dst_ptr),
                                    N_src_stride,
                                    D_src_stride,
                                    H_src_stride,
                                    C_src_stride,
                                    N_dst_stride,
                                    D_dst_stride,
                                    H_dst_stride,
                                    static_cast<int>(N),
                                    static_cast<int>(D),
                                    static_cast<int>(H),
                                    static_cast<int>(W));
            return;
        }

        if (PRC == Precision::FP32) {
            blob_copy_5d_merge_f32c3(reinterpret_cast<const float*>(src_ptr),
                                     reinterpret_cast<float*>(dst_ptr),
                                     N_src_stride,
                                     D_src_stride,
                                     H_src_stride,
                                     C_src_stride,
                                     N_dst_stride,
                                     D_dst_stride,
                                     H_dst_stride,
                                     static_cast<int>(N),
                                     static_cast<int>(D),
                                     static_cast<int>(H),
                                     static_cast<int>(W));
            return;
        }
    }
#else
    (void)H_dst_stride;
#endif  // HAVE_SSE
    if (src->getTensorDesc().getLayout() == NDHWC && dst->getTensorDesc().getLayout() == NCDHW) {
        for (size_t n = 0; n < N; n++) {
            for (size_t c = 0; c < C; c++) {
                for (size_t d = 0; d < D; d++) {
                    data_t* dst_ptr_l = dst_ptr + n * N_dst_stride + c * C_dst_stride + d * D_dst_stride;
                    data_t* src_ptr_l = src_ptr + n * N_src_stride + c * C_src_stride + d * D_src_stride;
                    for (size_t h = 0; h < H; h++) {
                        data_t* src_ptr_l_l = src_ptr_l + h * H_src_stride;
                        for (size_t w = 0; w < W; w++) {
                            *dst_ptr_l = *src_ptr_l_l;
                            src_ptr_l_l += W_src_stride;
                            dst_ptr_l++;
                        }
                    }
                }
            }
        }
    } else if (src->getTensorDesc().getLayout() == NCDHW && dst->getTensorDesc().getLayout() == NDHWC) {
        for (size_t n = 0; n < N; n++) {
            for (size_t c = 0; c < C; c++) {
                for (size_t d = 0; d < D; d++) {
                    data_t* src_ptr_l = src_ptr + n * N_src_stride + c * C_src_stride + d * D_src_stride;
                    data_t* dst_ptr_l = dst_ptr + n * N_dst_stride + c + d * D_dst_stride;
                    for (size_t h = 0; h < H; h++) {
                        data_t* src_ptr_l_l = src_ptr_l + h * H_src_stride;
                        for (size_t w = 0; w < W; w++) {
                            *dst_ptr_l = *src_ptr_l_l;
                            dst_ptr_l += W_dst_stride;
                            src_ptr_l_l++;
                        }
                    }
                }
            }
        }
    } else {
        for (size_t i = 0; i < N * C * D * H * W; i++) {
            dst_ptr[i] = src_ptr[i];
        }
    }
}

static inline void blob_copy_5d(Blob::Ptr src, Blob::Ptr dst) {
    switch (src->getTensorDesc().getPrecision()) {
    case Precision::FP32:
    case Precision::I32:
    case Precision::U32:
        blob_copy_5d_t<Precision::FP32>(src, dst);
        break;

    case Precision::FP16:
    case Precision::U16:
    case Precision::I16:
        blob_copy_5d_t<Precision::U16>(src, dst);
        break;

    case Precision::U8:
    case Precision::I8:
        blob_copy_5d_t<Precision::U8>(src, dst);
        break;

    default:
        IE_THROW() << "Unsupported blob transformation for precision " << src->getTensorDesc().getPrecision();
    }
}

void blob_copy(Blob::Ptr src, Blob::Ptr dst) {
    if (src->buffer() == nullptr)
        IE_THROW() << "Cannot copy blob data. Source is not allocated.";

    if (dst->buffer() == nullptr)
        IE_THROW() << "Cannot copy blob data. Destination is not allocated.";

    if (src->getTensorDesc().getPrecision() != dst->getTensorDesc().getPrecision())
        IE_THROW() << "Unimplemented blob transformation from precision " << src->getTensorDesc().getPrecision()
                   << " to " << src->getTensorDesc().getPrecision();

    if (src->getTensorDesc().getDims() != dst->getTensorDesc().getDims())
        IE_THROW() << "Unimplemented blob transformation from different shapes ";

    if (src->getTensorDesc().getDims().size() == 4)
        blob_copy_4d(src, dst);
    else if (src->getTensorDesc().getDims().size() == 5)
        blob_copy_5d(src, dst);
    else
        IE_THROW() << "Unimplemented blob transformation. Only 4d or 5d supported.";
}

}  // namespace InferenceEngine
