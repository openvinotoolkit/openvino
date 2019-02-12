// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_preprocess_data.hpp"
#include "ie_preprocess_data_sse42.hpp"
#if defined(__ANDROID__)
#include <immintrin.h> // SSE 4.2
#else
#include <nmmintrin.h>  // SSE 4.2
#endif
#include <stdint.h>

namespace InferenceEngine {
namespace Resize {

static inline int ceil(double value) {
    __m128d t = _mm_set_sd(value);
    int i = _mm_cvtsd_si32(t);
    return i + _mm_movemask_pd(_mm_cmplt_sd(_mm_cvtsi32_sd(t, i), t));
}


static inline int floor(double value) {
    __m128d t = _mm_set_sd(value);
    int i = _mm_cvtsd_si32(t);
    return i - _mm_movemask_pd(_mm_cmplt_sd(t, _mm_cvtsi32_sd(t, i)));
}

static inline int16_t mulq15(int16_t a, int16_t b) {
    return static_cast<int16_t>(((1 << 14) + (int32_t)a * (int32_t)b) >> 15);
}

static inline uint16_t mulq16(uint16_t a, uint16_t b) {
    return static_cast<uint16_t>(((uint32_t)a * (uint32_t)b) >> 16);
}

void resize_bilinear_u8(const Blob::Ptr inBlob, Blob::Ptr outBlob, uint8_t* buffer) {
    Border border = {BORDER_REPLICATE, 0};

    auto dstDims = outBlob->getTensorDesc().getDims();
    auto srcDims = inBlob->getTensorDesc().getDims();

    auto dwidth = static_cast<const int>(dstDims[3]);
    auto dheight = static_cast<const int>(dstDims[2]);
    auto swidth = static_cast<const int>(srcDims[3]);
    auto channels = static_cast<const int>(srcDims[1]);

    auto src_strides = inBlob->getTensorDesc().getBlockingDesc().getStrides();
    auto dst_strides = outBlob->getTensorDesc().getBlockingDesc().getStrides();
    auto origSrcW = src_strides[2];
    auto origSrcH = src_strides[1] / src_strides[2];
    auto origDstW = dst_strides[2];
    auto origDstH = dst_strides[1] / dst_strides[2];

    const int src_go_x = 0;
    const int src_go_y = 0;
    const int dst_go_x = 0;
    const int dst_go_y = 0;
    auto src_full_width = static_cast<const int>(srcDims[3]);
    auto src_full_height = static_cast<const int>(srcDims[2]);
    auto dst_full_width = static_cast<const int>(dstDims[3]);
    auto dst_full_height = static_cast<const int>(dstDims[2]);

    const uint8_t *sptr = static_cast<uint8_t *>(inBlob->buffer()) +
                          inBlob->getTensorDesc().getBlockingDesc().getOffsetPadding();
    uint8_t *dptr = static_cast<uint8_t *>(outBlob->buffer()) +
                    outBlob->getTensorDesc().getBlockingDesc().getOffsetPadding();

    auto sstep = static_cast<const int>(inBlob->getTensorDesc().getBlockingDesc().getStrides()[2]);
    auto dstep = static_cast<const int>(outBlob->getTensorDesc().getBlockingDesc().getStrides()[2]);
    auto scale_x = static_cast<float>(src_full_width) / dst_full_width;
    auto scale_y = static_cast<float>(src_full_height) / dst_full_height;

    const int BITS = 15;
    const int SCALE = (1 << BITS);
    const int alpha_clones_num = 4;
    const int cols_block_size = 8;
    const int kRowsBlockSize = 4;

    auto *pxofs1 = reinterpret_cast<int32_t *>(buffer);
    auto *alpha = reinterpret_cast<int16_t *>(pxofs1 + dwidth);
    auto *yofs = reinterpret_cast<int32_t *>(alpha + dwidth * alpha_clones_num);
    auto *beta = reinterpret_cast<int16_t *>(yofs + dheight);
    auto *tptr = reinterpret_cast<uint8_t *>(beta + dheight);

    auto tptr_ = tptr;

    tptr_[0] = (uint8_t) border.value;
    tptr_[1] = (uint8_t) border.value;
    tptr_[2] = (uint8_t) border.value;
    tptr_[3] = (uint8_t) border.value;
    tptr_[swidth + 0 + 4] = (uint8_t) border.value;
    tptr_[swidth + 1 + 4] = (uint8_t) border.value;
    tptr_[swidth + 2 + 4] = (uint8_t) border.value;
    tptr_[swidth + 3 + 4] = (uint8_t) border.value;
    tptr_[swidth * kRowsBlockSize + 0 + 4] = (uint8_t) border.value;
    tptr_[swidth * kRowsBlockSize + 1 + 4] = (uint8_t) border.value;
    tptr_[swidth * kRowsBlockSize + 2 + 4] = (uint8_t) border.value;
    tptr_[swidth * kRowsBlockSize + 3 + 4] = (uint8_t) border.value;

    for (int dx = dst_go_x; dx < dst_go_x + dwidth; dx++) {
        auto fx = static_cast<float>((dx + 0.5) * scale_x - 0.5);
        int32_t sx = floor(fx);
        fx -= sx;

        int32_t sx0 = sx;
        if (sx < 0 && border.type == BORDER_REPLICATE) {
            fx = 0;
            sx0 = 0;
        }

        fx = fx * SCALE;

        if (sx >= src_full_width - 1 && border.type == BORDER_REPLICATE) {
            fx = 1.f * SCALE - 1;
            sx0 = (std::max)(src_full_width - 2, 0);
        }

        pxofs1[dx - dst_go_x] = kRowsBlockSize * (sx0 - src_go_x);
        for (int i = 0; i < alpha_clones_num; i++) {
            alpha[(dx - dst_go_x) * alpha_clones_num + i] = (int16_t) fx;
        }
    }

    for (int dy = dst_go_y; dy < dst_go_y + dheight; dy++) {
        float fy = static_cast<float>((dy + 0.5) * scale_y - 0.5);
        int32_t sy = floor(fy);
        fy -= sy;

        int32_t sy0 = sy;
        if (sy < 0 && border.type == BORDER_REPLICATE) {
            fy = 0;
            sy0 = 0;
        }

        fy = fy * SCALE;

        if (sy >= src_full_height - 1 && border.type == BORDER_REPLICATE) {
            fy = 1.f * SCALE - 1;
            sy0 = (std::max)(src_full_height - 2, 0);
        }

        yofs[dy - dst_go_y] = (sy0 - src_go_y) * sstep;
        beta[dy - dst_go_y] = (int16_t) fy;
    }

    if (swidth < cols_block_size || dwidth < cols_block_size || dheight < kRowsBlockSize) {
        auto full_pass = [&](int c, int y) {
            auto sptr_ = sptr + c * origSrcW * origSrcH;
            auto dptr_ = dptr + c * origDstW * origDstH;
            auto tptr_ = tptr;

            for (int x = 0; x < swidth; x++) {
                int val0 = (yofs[y] < 0) ? border.value : sptr_[yofs[y] + x + 0];
                int val1 = (yofs[y] / sstep + 1 >= src_full_height - src_go_y) ? border.value : sptr_[yofs[y] + x +
                                                                                                      sstep];

                int res = val0 + mulq15(beta[y], (int16_t) (val1 - val0));
                tptr_[x + 4] = (uint8_t) res;
            }

            for (int x = 0; x < dwidth; x++) {
                int val0 = tptr_[pxofs1[x] / kRowsBlockSize + 0 + 4];
                int val1 = tptr_[pxofs1[x] / kRowsBlockSize + 1 + 4];

                int res = val0 + mulq15(alpha[x * alpha_clones_num], (int16_t) (val1 - val0));
                dptr_[y * dstep + x] = (uint8_t) res;
            }
        };

        for (int c = 0; c < channels; c++) {
            for (int y = 0; y < dheight; y++) {
                full_pass(c, y);
            }
        }

        return;
    }

    auto full_pass_vec = [&](const uint8_t* sptr_, uint8_t* dptr_, uint8_t* tptr_, int y) {
        int32_t filtered_rows_id[4];
        for (int i = 0; i < 4; i++) {
            filtered_rows_id[i] = (yofs[y + i] < 0) ? 0 :
                                  (yofs[y + i] / sstep >= src_full_height - src_go_y - 1) ? 0 : yofs[y + i];
        }

        __m128i b0 = _mm_set1_epi16(beta[y + 0]);
        __m128i b1 = _mm_set1_epi16(beta[y + 1]);
        __m128i b2 = _mm_set1_epi16(beta[y + 2]);
        __m128i b3 = _mm_set1_epi16(beta[y + 3]);

        int x = 0;
        vertical_pass:
        for (; x <= swidth - cols_block_size; x += cols_block_size) {
            __m128i val0lo = _mm_insert_epi64(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(sptr_ + x + filtered_rows_id[0])),
                                              *(reinterpret_cast<const int64_t *>(sptr_ + x + filtered_rows_id[1])), 1);
            __m128i val0hi = _mm_insert_epi64(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(sptr_ + x + filtered_rows_id[2])),
                                              *(reinterpret_cast<const int64_t *>(sptr_ + x + filtered_rows_id[3])), 1);
            __m128i val1lo = _mm_insert_epi64(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(sptr_ + x + filtered_rows_id[0] + sstep)),
                                              *(reinterpret_cast<const int64_t *>(sptr_ + x + filtered_rows_id[1] + sstep)), 1);
            __m128i val1hi = _mm_insert_epi64(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(sptr_ + x + filtered_rows_id[2] + sstep)),
                                              *(reinterpret_cast<const int64_t *>(sptr_ + x + filtered_rows_id[3] + sstep)), 1);

            __m128i val0_0 = _mm_unpacklo_epi8(val0lo, _mm_setzero_si128());
            __m128i val0_1 = _mm_unpackhi_epi8(val0lo, _mm_setzero_si128());
            __m128i val0_2 = _mm_unpacklo_epi8(val0hi, _mm_setzero_si128());
            __m128i val0_3 = _mm_unpackhi_epi8(val0hi, _mm_setzero_si128());

            __m128i val1_0 = _mm_unpacklo_epi8(val1lo, _mm_setzero_si128());
            __m128i val1_1 = _mm_unpackhi_epi8(val1lo, _mm_setzero_si128());
            __m128i val1_2 = _mm_unpacklo_epi8(val1hi, _mm_setzero_si128());
            __m128i val1_3 = _mm_unpackhi_epi8(val1hi, _mm_setzero_si128());

            __m128i s0_0 = _mm_sub_epi16(val1_0, val0_0);
            __m128i s0_1 = _mm_sub_epi16(val1_1, val0_1);
            __m128i s0_2 = _mm_sub_epi16(val1_2, val0_2);
            __m128i s0_3 = _mm_sub_epi16(val1_3, val0_3);

            __m128i t0 = _mm_mulhrs_epi16(s0_0, b0);
            __m128i t1 = _mm_mulhrs_epi16(s0_1, b1);
            __m128i t2 = _mm_mulhrs_epi16(s0_2, b2);
            __m128i t3 = _mm_mulhrs_epi16(s0_3, b3);

            __m128i r0 = _mm_add_epi16(val0_0, t0);
            __m128i r1 = _mm_add_epi16(val0_1, t1);
            __m128i r2 = _mm_add_epi16(val0_2, t2);
            __m128i r3 = _mm_add_epi16(val0_3, t3);

            __m128i q0 = _mm_packus_epi16(r0, r1);
            __m128i q1 = _mm_packus_epi16(r2, r3);

            __m128i q2 = _mm_blend_epi16(q0, _mm_slli_si128(q1, 4), 0xCC /*0b11001100*/);
            __m128i q3 = _mm_blend_epi16(_mm_srli_si128(q0, 4), q1, 0xCC /*0b11001100*/);

            __m128i q4 = _mm_shuffle_epi8(q2, _mm_setr_epi8(0, 8, 4, 12, 1, 9, 5, 13, 2, 10, 6, 14, 3, 11, 7, 15));
            __m128i q5 = _mm_shuffle_epi8(q3, _mm_setr_epi8(0, 8, 4, 12, 1, 9, 5, 13, 2, 10, 6, 14, 3, 11, 7, 15));

            _mm_storeu_si128(reinterpret_cast<__m128i *>(tptr_ + (x + 0) * kRowsBlockSize + 4), q4);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(tptr_ + (x + 4) * kRowsBlockSize + 4), q5);
        }

        if (x < swidth) {
            x = swidth - cols_block_size;
            goto vertical_pass;
        }

        if (border.type == BORDER_CONSTANT) {
            for (int i = 0; i < kRowsBlockSize; i++) {
                if (yofs[y + i] < 0) {
                    for (x = 0; x < swidth; x++) {
                        int val0 = border.value;
                        int val1 = sptr_[yofs[y + i] + x + sstep];

                        int res = val0 + mulq15(beta[y + i], (int16_t) (val1 - val0));
                        tptr_[x * 4 + i + 4] = (uint8_t) res;
                    }
                }

                if (yofs[y + i] / sstep >= src_full_height - src_go_y - 1) {
                    for (x = 0; x < swidth; x++) {
                        int val0 = sptr_[yofs[y + i] + x];
                        int val1 = border.value;

                        int res = val0 + mulq15(beta[y + i], (int16_t) (val1 - val0));
                        tptr_[x * 4 + i + 4] = (uint8_t) res;
                    }
                }
            }
        }

        x = 0;
        horizontal_pass:
        for (; x <= dwidth - cols_block_size; x += cols_block_size) {
            __m128i a10 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(alpha + (x + 0) * alpha_clones_num));
            __m128i a32 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(alpha + (x + 2) * alpha_clones_num));
            __m128i a54 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(alpha + (x + 4) * alpha_clones_num));
            __m128i a76 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(alpha + (x + 6) * alpha_clones_num));

            __m128i val_0 = _mm_insert_epi64(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(tptr_ + pxofs1[x + 0] + 4)),
                                             *(reinterpret_cast<const int64_t *>(tptr_ + pxofs1[x + 1] + 4)), 1);
            __m128i val_1 = _mm_insert_epi64(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(tptr_ + pxofs1[x + 2] + 4)),
                                             *(reinterpret_cast<const int64_t *>(tptr_ + pxofs1[x + 3] + 4)), 1);
            __m128i val_2 = _mm_insert_epi64(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(tptr_ + pxofs1[x + 4] + 4)),
                                             *(reinterpret_cast<const int64_t *>(tptr_ + pxofs1[x + 5] + 4)), 1);
            __m128i val_3 = _mm_insert_epi64(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(tptr_ + pxofs1[x + 6] + 4)),
                                             *(reinterpret_cast<const int64_t *>(tptr_ + pxofs1[x + 7] + 4)), 1);

            val_0 = _mm_shuffle_epi32(val_0, _MM_SHUFFLE(3, 1, 2, 0));
            val_1 = _mm_shuffle_epi32(val_1, _MM_SHUFFLE(3, 1, 2, 0));
            val_2 = _mm_shuffle_epi32(val_2, _MM_SHUFFLE(3, 1, 2, 0));
            val_3 = _mm_shuffle_epi32(val_3, _MM_SHUFFLE(3, 1, 2, 0));

            __m128i val0_0 = _mm_unpacklo_epi8(val_0, _mm_setzero_si128());
            __m128i val0_1 = _mm_unpacklo_epi8(val_1, _mm_setzero_si128());
            __m128i val0_2 = _mm_unpacklo_epi8(val_2, _mm_setzero_si128());
            __m128i val0_3 = _mm_unpacklo_epi8(val_3, _mm_setzero_si128());

            __m128i val1_0 = _mm_unpackhi_epi8(val_0, _mm_setzero_si128());
            __m128i val1_1 = _mm_unpackhi_epi8(val_1, _mm_setzero_si128());
            __m128i val1_2 = _mm_unpackhi_epi8(val_2, _mm_setzero_si128());
            __m128i val1_3 = _mm_unpackhi_epi8(val_3, _mm_setzero_si128());

            val1_0 = _mm_sub_epi16(val1_0, val0_0);
            val1_1 = _mm_sub_epi16(val1_1, val0_1);
            val1_2 = _mm_sub_epi16(val1_2, val0_2);
            val1_3 = _mm_sub_epi16(val1_3, val0_3);

            __m128i t0 = _mm_mulhrs_epi16(val1_0, a10);
            __m128i t1 = _mm_mulhrs_epi16(val1_1, a32);
            __m128i t2 = _mm_mulhrs_epi16(val1_2, a54);
            __m128i t3 = _mm_mulhrs_epi16(val1_3, a76);

            __m128i r0 = _mm_add_epi16(val0_0, t0);
            __m128i r1 = _mm_add_epi16(val0_1, t1);
            __m128i r2 = _mm_add_epi16(val0_2, t2);
            __m128i r3 = _mm_add_epi16(val0_3, t3);

            __m128i q0 = _mm_packus_epi16(r0, r1);
            __m128i q1 = _mm_packus_epi16(r2, r3);

            __m128i q2 = _mm_shuffle_epi8(q0, _mm_setr_epi8(0, 4, 8, 12, 2, 6, 10, 14, 1, 5, 9, 13, 3, 7, 11, 15));
            __m128i q3 = _mm_shuffle_epi8(q1, _mm_setr_epi8(0, 4, 8, 12, 2, 6, 10, 14, 1, 5, 9, 13, 3, 7, 11, 15));

            __m128i q4 = _mm_blend_epi16(q2, _mm_slli_si128(q3, 4), 0xCC /*0b11001100*/);
            __m128i q5 = _mm_blend_epi16(_mm_srli_si128(q2, 4), q3, 0xCC /*0b11001100*/);

            _mm_storel_epi64(reinterpret_cast<__m128i *>(dptr_ + (y + 0) * dstep + x), q4);
            _mm_storel_epi64(reinterpret_cast<__m128i *>(dptr_ + (y + 1) * dstep + x), _mm_srli_si128(q4, 8));
            _mm_storel_epi64(reinterpret_cast<__m128i *>(dptr_ + (y + 2) * dstep + x), q5);
            _mm_storel_epi64(reinterpret_cast<__m128i *>(dptr_ + (y + 3) * dstep + x), _mm_srli_si128(q5, 8));
        }

        if (x < dwidth) {
            x = dwidth - cols_block_size;
            goto horizontal_pass;
        }
    };

    for (int c = 0; c < channels; c++) {
        for (int y = 0; y <= dheight - kRowsBlockSize; y += kRowsBlockSize) {
            auto sptr_ = sptr + c * origSrcW * origSrcH;
            auto dptr_ = dptr + c * origDstW * origDstH;
            auto tptr_ = tptr;

            full_pass_vec(sptr_, dptr_, tptr_, y);

            if (y + kRowsBlockSize > dheight - kRowsBlockSize)
                full_pass_vec(sptr_, dptr_, tptr_, dheight - kRowsBlockSize);
        }
    }
}

void resize_area_u8_downscale(const Blob::Ptr inBlob, Blob::Ptr outBlob, uint8_t* buffer) {
    auto dstDims = outBlob->getTensorDesc().getDims();
    auto srcDims = inBlob->getTensorDesc().getDims();

    auto dwidth = static_cast<const int>(dstDims[3]);
    auto dheight = static_cast<const int>(dstDims[2]);
    auto swidth = static_cast<const int>(srcDims[3]);
    auto sheight = static_cast<const int>(srcDims[2]);
    auto channels = static_cast<const int>(srcDims[1]);

    auto src_strides = inBlob->getTensorDesc().getBlockingDesc().getStrides();
    auto dst_strides = outBlob->getTensorDesc().getBlockingDesc().getStrides();
    auto origSrcW = src_strides[2];
    auto origSrcH = src_strides[1] / src_strides[2];
    auto origDstW = dst_strides[2];
    auto origDstH = dst_strides[1] / dst_strides[2];

    const int src_go_x = 0;
    const int src_go_y = 0;
    const int dst_go_x = 0;
    const int dst_go_y = 0;

    auto src_full_width = static_cast<const int>(srcDims[3]);
    auto src_full_height = static_cast<const int>(srcDims[2]);
    auto dst_full_width = static_cast<const int>(dstDims[3]);
    auto dst_full_height = static_cast<const int>(dstDims[2]);

    auto sptr = static_cast<uint8_t*>(inBlob->buffer()) + inBlob->getTensorDesc().getBlockingDesc().getOffsetPadding();
    auto dptr = static_cast<uint8_t*>(outBlob->buffer()) + outBlob->getTensorDesc().getBlockingDesc().getOffsetPadding();
    auto sstep = static_cast<const int>(inBlob->getTensorDesc().getBlockingDesc().getStrides()[2]);
    auto dstep = static_cast<const int>(outBlob->getTensorDesc().getBlockingDesc().getStrides()[2]);

    float scale_x = static_cast<float>(src_full_width) / dst_full_width;
    float scale_y = static_cast<float>(src_full_height) / dst_full_height;

    int x_max_count = getResizeAreaTabSize(dst_go_x, src_full_width,  dwidth,  scale_x);
    int y_max_count = getResizeAreaTabSize(dst_go_y, src_full_height, dheight, scale_y);

    auto* xsi = reinterpret_cast<uint16_t*>(buffer);
    auto* ysi = xsi + dwidth;
    auto* xalpha = ysi + dheight;
    auto* yalpha = xalpha + dwidth*x_max_count + 8*16;

    computeResizeAreaTab(src_go_x, dst_go_x, src_full_width,   dwidth, scale_x, xsi, xalpha, x_max_count);
    computeResizeAreaTab(src_go_y, dst_go_y, src_full_height, dheight, scale_y, ysi, yalpha, y_max_count);

    int vest_sum_size = 2*swidth;
    uint16_t* vert_sum = yalpha + dheight*y_max_count;
    uint16_t* alpha0 = vert_sum + vest_sum_size;
    uint16_t* alpha1 = alpha0 + dwidth;
    uint16_t* alpha2 = alpha1 + dwidth;
    uint16_t* alpha3 = alpha2 + dwidth;
    uint16_t* sxid0 = alpha3 + dwidth;
    uint16_t* sxid1 = sxid0 + 4*dwidth;
    uint16_t* sxid2 = sxid1 + 4*dwidth;
    uint16_t* sxid3 = sxid2 + 4*dwidth;

    uint16_t* alpha[] = {alpha0, alpha1, alpha2, alpha3};
    uint16_t* sxid[] = {sxid0, sxid1, sxid2, sxid3};
    generate_alpha_and_id_arrays(x_max_count, dwidth, xalpha, xsi, alpha, sxid);

    auto full_pass = [&](int c, int y) {
        uint8_t* pdst_row = dptr + (y * dstep) + c * origDstW * origDstH;
        uint16_t* vert_sum_ = vert_sum;

        int ysi_row = ysi[y];

        memset(vert_sum_, 0, swidth * sizeof(uint16_t));

        for (int dy = 0; dy < y_max_count; dy++) {
            uint16_t yalpha_dy = yalpha[y * y_max_count + dy];
            const uint8_t *sptr_dy = sptr + ((ysi_row + dy) * sstep) + c * origSrcW * origSrcH;
            if (ysi_row + dy >= sheight) break;

            int x = 0;

            __m128i yalpha_dy_sse = _mm_set1_epi16(yalpha_dy);
            for (; x <= swidth - 16; x += 16) {
                __m128i sval = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sptr_dy + x));

                // sptr_dy[x] << 8
                __m128i sval_Q16_lo = _mm_unpacklo_epi8(_mm_setzero_si128(), sval);
                __m128i sval_Q16_hi = _mm_unpackhi_epi8(_mm_setzero_si128(), sval);

                __m128i vert_sum_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vert_sum_ + x + 0));
                __m128i vert_sum_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vert_sum_ + x + 8));

                vert_sum_lo = _mm_add_epi16(vert_sum_lo, _mm_mulhi_epu16(yalpha_dy_sse, sval_Q16_lo));
                vert_sum_hi = _mm_add_epi16(vert_sum_hi, _mm_mulhi_epu16(yalpha_dy_sse, sval_Q16_hi));

                _mm_storeu_si128(reinterpret_cast<__m128i*>(vert_sum_ + x + 0), vert_sum_lo);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(vert_sum_ + x + 8), vert_sum_hi);
            }

            for (; x < swidth; x++) {
                vert_sum_[x] += mulq16(yalpha_dy, static_cast<uint16_t>(sptr_dy[x] << 8));
            }
        }

        if (x_max_count == 2) {
            int x = 0;
            for (; x <= dwidth - 8; x += 8) {
                __m128i res = _mm_set1_epi16(1 << (8 - 1));

                int id0 = xsi[x];

                __m128i chunk0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vert_sum_ + id0));
                __m128i chunk1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vert_sum_ + id0 + 8));

                __m128i sx0_id0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid0 + x * 2));
                __m128i sx0_id1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid0 + x * 2 + 8));

                __m128i sx1_id0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid1 + x * 2));
                __m128i sx1_id1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid1 + x * 2 + 8));

                __m128i vert_sum0 = _mm_or_si128(_mm_shuffle_epi8(chunk0, sx0_id0),
                                                 _mm_shuffle_epi8(chunk1, sx0_id1));
                __m128i vert_sum1 = _mm_or_si128(_mm_shuffle_epi8(chunk0, sx1_id0),
                                                 _mm_shuffle_epi8(chunk1, sx1_id1));

                res = _mm_add_epi16(res, _mm_mulhi_epu16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(alpha0 + x)), vert_sum0));
                res = _mm_add_epi16(res, _mm_mulhi_epu16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(alpha1 + x)), vert_sum1));

                res = _mm_srli_epi16(res, 8);
                res = _mm_packus_epi16(res, res);
                _mm_storel_epi64(reinterpret_cast<__m128i*>(pdst_row + x), res);
            }

            for (; x < dwidth; x++) {
                uint16_t res = 1 << (8 - 1);
                int id = xsi[x];
                res += mulq16(alpha0[x], vert_sum_[id + 0]);
                res += mulq16(alpha1[x], vert_sum_[id + 1]);
                pdst_row[x] = saturateU32toU8(res >> 8);
            }
        } else if (x_max_count == 3) {
            int x = 0;
            for (; x <= dwidth - 8; x += 8) {
                __m128i res = _mm_set1_epi16(1 << (8 - 1));

                int id0 = xsi[x];

                __m128i chunk0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vert_sum_ + id0));
                __m128i chunk1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vert_sum_ + id0 + 8));
                __m128i chunk2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vert_sum_ + id0 + 16));

                __m128i sx0_id0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid0 + x * 3));
                __m128i sx0_id1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid0 + x * 3 + 8));
                __m128i sx0_id2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid0 + x * 3 + 16));

                __m128i sx1_id0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid1 + x * 3));
                __m128i sx1_id1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid1 + x * 3 + 8));
                __m128i sx1_id2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid1 + x * 3 + 16));

                __m128i sx2_id0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid2 + x * 3));
                __m128i sx2_id1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid2 + x * 3 + 8));
                __m128i sx2_id2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid2 + x * 3 + 16));

                __m128i vert_sum0 = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(chunk0, sx0_id0),
                                                              _mm_shuffle_epi8(chunk1, sx0_id1)),
                                                 _mm_shuffle_epi8(chunk2, sx0_id2));
                __m128i vert_sum1 = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(chunk0, sx1_id0),
                                                              _mm_shuffle_epi8(chunk1, sx1_id1)),
                                                 _mm_shuffle_epi8(chunk2, sx1_id2));
                __m128i vert_sum2 = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(chunk0, sx2_id0),
                                                              _mm_shuffle_epi8(chunk1, sx2_id1)),
                                                 _mm_shuffle_epi8(chunk2, sx2_id2));

                res = _mm_add_epi16(res, _mm_mulhi_epu16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(alpha0 + x)), vert_sum0));
                res = _mm_add_epi16(res, _mm_mulhi_epu16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(alpha1 + x)), vert_sum1));
                res = _mm_add_epi16(res, _mm_mulhi_epu16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(alpha2 + x)), vert_sum2));

                res = _mm_srli_epi16(res, 8);
                res = _mm_packus_epi16(res, res);
                _mm_storel_epi64(reinterpret_cast<__m128i*>(pdst_row + x), res);
            }

            for (; x < dwidth; x++) {
                uint16_t res = 1 << (8 - 1);
                int id = xsi[x];
                res += mulq16(alpha0[x], vert_sum_[id + 0]);
                res += mulq16(alpha1[x], vert_sum_[id + 1]);
                res += mulq16(alpha2[x], vert_sum_[id + 2]);
                pdst_row[x] = saturateU32toU8(res >> 8);
            }
        } else if (x_max_count == 4) {
            int x = 0;
            for (; x <= dwidth - 8; x += 8) {
                __m128i res = _mm_set1_epi16(1 << (8 - 1));

                int id0 = xsi[x];

                __m128i chunk0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vert_sum_ + id0));
                __m128i chunk1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vert_sum_ + id0 + 8));
                __m128i chunk2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vert_sum_ + id0 + 16));
                __m128i chunk3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vert_sum_ + id0 + 24));

                __m128i sx0_id0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid0 + x * 4));
                __m128i sx0_id1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid0 + x * 4 + 8));
                __m128i sx0_id2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid0 + x * 4 + 16));
                __m128i sx0_id3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid0 + x * 4 + 24));

                __m128i sx1_id0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid1 + x * 4));
                __m128i sx1_id1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid1 + x * 4 + 8));
                __m128i sx1_id2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid1 + x * 4 + 16));
                __m128i sx1_id3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid1 + x * 4 + 24));

                __m128i sx2_id0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid2 + x * 4));
                __m128i sx2_id1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid2 + x * 4 + 8));
                __m128i sx2_id2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid2 + x * 4 + 16));
                __m128i sx2_id3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid2 + x * 4 + 24));

                __m128i sx3_id0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid3 + x * 4));
                __m128i sx3_id1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid3 + x * 4 + 8));
                __m128i sx3_id2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid3 + x * 4 + 16));
                __m128i sx3_id3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid3 + x * 4 + 24));

                __m128i vert_sum0 = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(chunk0, sx0_id0),
                                                              _mm_shuffle_epi8(chunk1, sx0_id1)),
                                                 _mm_or_si128(_mm_shuffle_epi8(chunk2, sx0_id2),
                                                              _mm_shuffle_epi8(chunk3, sx0_id3)));
                __m128i vert_sum1 = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(chunk0, sx1_id0),
                                                              _mm_shuffle_epi8(chunk1, sx1_id1)),
                                                 _mm_or_si128(_mm_shuffle_epi8(chunk2, sx1_id2),
                                                              _mm_shuffle_epi8(chunk3, sx1_id3)));
                __m128i vert_sum2 = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(chunk0, sx2_id0),
                                                              _mm_shuffle_epi8(chunk1, sx2_id1)),
                                                 _mm_or_si128(_mm_shuffle_epi8(chunk2, sx2_id2),
                                                              _mm_shuffle_epi8(chunk3, sx2_id3)));
                __m128i vert_sum3 = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(chunk0, sx3_id0),
                                                              _mm_shuffle_epi8(chunk1, sx3_id1)),
                                                 _mm_or_si128(_mm_shuffle_epi8(chunk2, sx3_id2),
                                                              _mm_shuffle_epi8(chunk3, sx3_id3)));

                res = _mm_add_epi16(res, _mm_mulhi_epu16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(alpha0 + x)), vert_sum0));
                res = _mm_add_epi16(res, _mm_mulhi_epu16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(alpha1 + x)), vert_sum1));
                res = _mm_add_epi16(res, _mm_mulhi_epu16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(alpha2 + x)), vert_sum2));
                res = _mm_add_epi16(res, _mm_mulhi_epu16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(alpha3 + x)), vert_sum3));

                res = _mm_srli_epi16(res, 8);
                res = _mm_packus_epi16(res, res);
                _mm_storel_epi64(reinterpret_cast<__m128i*>(pdst_row + x), res);
            }

            for (; x < dwidth; x++) {
                uint16_t res = 1 << (8 - 1);
                int id = xsi[x];
                res += mulq16(alpha0[x], vert_sum_[id + 0]);
                res += mulq16(alpha1[x], vert_sum_[id + 1]);
                res += mulq16(alpha2[x], vert_sum_[id + 2]);
                res += mulq16(alpha3[x], vert_sum_[id + 3]);
                pdst_row[x] = saturateU32toU8(res >> 8);
            }
        } else if (x_max_count <= 7) {
            int x = 0;
            for (; x <= dwidth - 8; x += 8) {
                __m128i res = _mm_set1_epi16(1 << (16 - 8 - 1));
                for (int i = 0; i < x_max_count; i++) {
                    __m128i valpha = _mm_setr_epi16(xalpha[x * x_max_count + x_max_count * 0 + i],
                                                    xalpha[x * x_max_count + x_max_count * 1 + i],
                                                    xalpha[x * x_max_count + x_max_count * 2 + i],
                                                    xalpha[x * x_max_count + x_max_count * 3 + i],
                                                    xalpha[x * x_max_count + x_max_count * 4 + i],
                                                    xalpha[x * x_max_count + x_max_count * 5 + i],
                                                    xalpha[x * x_max_count + x_max_count * 6 + i],
                                                    xalpha[x * x_max_count + x_max_count * 7 + i]);
                    __m128i vvert_sum = _mm_setr_epi16(vert_sum_[xsi[x + 0] + i],
                                                       vert_sum_[xsi[x + 1] + i],
                                                       vert_sum_[xsi[x + 2] + i],
                                                       vert_sum_[xsi[x + 3] + i],
                                                       vert_sum_[xsi[x + 4] + i],
                                                       vert_sum_[xsi[x + 5] + i],
                                                       vert_sum_[xsi[x + 6] + i],
                                                       vert_sum_[xsi[x + 7] + i]);

                    res = _mm_add_epi16(res, _mm_mulhi_epu16(valpha, vvert_sum));
                }
                res = _mm_srli_epi16(res, 8);
                res = _mm_packus_epi16(res, res);
                _mm_storel_epi64(reinterpret_cast<__m128i*>(pdst_row + x), res);
            }

            for (; x < dwidth; x++) {
                uint16_t res = 1 << (8 - 1);
                for (int i = 0; i < x_max_count; i++) {
                    uint16_t a = xalpha[x * x_max_count + i];
                    int sx = xsi[x] + i;

                    res += mulq16(a, vert_sum_[sx]);
                }
                pdst_row[x] = saturateU32toU8(res >> 8);
            }
        } else {
            for (int x = 0; x < dwidth; x++) {
                uint16_t res = 1 << (8 - 1);
                __m128i vres = _mm_setzero_si128();
                int id = xsi[x];

                int i = 0;
                for (; i <= x_max_count - 8; i += 8) {
                    __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(xalpha + x * x_max_count + i));
                    __m128i s = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vert_sum_ + id + i));

                    vres = _mm_add_epi16(vres, _mm_mulhi_epu16(a, s));
                }
                vres = _mm_add_epi16(vres, _mm_slli_si128(vres, 2));
                vres = _mm_add_epi16(vres, _mm_slli_si128(vres, 4));
                vres = _mm_add_epi16(vres, _mm_slli_si128(vres, 8));
                res += static_cast<uint16_t>(_mm_extract_epi16(vres, 7));

                for (; i < x_max_count; i++) {
                    uint16_t a = xalpha[x * x_max_count + i];
                    uint16_t s = vert_sum_[id + i];

                    res += mulq16(a, s);
                }

                pdst_row[x] = saturateU32toU8(res >> 8);
            }
        }
    };

    for (int c = 0; c < channels; c++) {
        for (int y = 0; y < dheight; y++) {
            full_pass(c, y);
        }
    }
}

}  // namespace Resize
}  // namespace InferenceEngine
