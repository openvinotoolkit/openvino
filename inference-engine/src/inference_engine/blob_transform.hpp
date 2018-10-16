// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_blob.h"

namespace InferenceEngine {

/**
 * Perform data copy with taking into account
 * layout and precision params
 */
static void blob_copy(Blob::Ptr src, Blob::Ptr dst);

template <InferenceEngine::Precision::ePrecision PRC>
static void blob_copy_4d_t(Blob::Ptr src, Blob::Ptr dst) {
    using data_t = typename InferenceEngine::PrecisionTrait<PRC>::value_type;

    data_t *src_ptr = src->buffer().as<data_t*>();
    data_t *dst_ptr = dst->buffer().as<data_t*>();

    auto dims = src->getTensorDesc().getDims();

    int N = dims[0];
    int C = dims[1];
    int H = dims[2];
    int W = dims[3];

    const auto src_blk_desc = src->getTensorDesc().getBlockingDesc();
    const auto src_strides = src_blk_desc.getStrides();
    const auto src_order = src_blk_desc.getOrder();
    int N_src_stride = src_strides[0];
    int C_src_stride = src->layout() == NHWC ? src_strides[3] : src_strides[1];
    int H_src_stride = src->layout() == NHWC ? src_strides[1] : src_strides[2];
    int W_src_stride = src->layout() == NHWC ? src_strides[2] : src_strides[3];
    int src_off = src_blk_desc.getOffsetPadding();

    src_ptr += src_off;


    const auto dst_blk_desc = dst->getTensorDesc().getBlockingDesc();
    const auto dst_strides = dst_blk_desc.getStrides();
    const auto dst_order = dst_blk_desc.getOrder();
    int N_dst_stride = dst_strides[0];
    int C_dst_stride = dst->layout() == NHWC ? dst_strides[3] : dst_strides[1];
    int H_dst_stride = dst->layout() == NHWC ? dst_strides[1] : dst_strides[2];
    int W_dst_stride = dst->layout() == NHWC ? dst_strides[2] : dst_strides[3];
    int dst_off = dst_blk_desc.getOffsetPadding();

    src_ptr += dst_off;

    // WA. Because of wrong filler
    int _N_dst_stride = C*H*W;
    int _C_dst_stride = H*W;
    int _W_dst_stride = C;

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
                data_t *src_ptr_l = src_ptr + n * N_src_stride + c * H * W;
                data_t *dst_ptr_l = dst_ptr + n * _N_dst_stride + c;
                for (int hw = 0; hw < H*W; hw++) {
                    *dst_ptr_l = *src_ptr_l;
                    dst_ptr_l += _W_dst_stride;
                    src_ptr_l++;
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

static void blob_copy(Blob::Ptr src, Blob::Ptr dst) {
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
