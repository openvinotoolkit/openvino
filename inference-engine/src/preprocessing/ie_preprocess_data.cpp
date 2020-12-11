// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_preprocess_gapi.hpp"
#include "ie_system_conf.h"
#include "blob_transform.hpp"
#include "ie_preprocess_data.hpp"
#include "ie_preprocess_itt.hpp"

#ifdef HAVE_SSE
# include "cpu_x86_sse42/ie_preprocess_data_sse42.hpp"
#endif

#include "debug.h"
#include "ie_compound_blob.h"
#include <ie_input_info.hpp>

#include <memory>
#include <algorithm>

namespace InferenceEngine {


namespace Resize {

template<typename data_t> static inline data_t saturate_cast(float res);

template<> inline float saturate_cast(float res) {
    return res;
}

template<> inline uint8_t saturate_cast(float res) {
    int ires = static_cast<int>((std::round)(res));
    return static_cast<uint8_t>((std::max)(0, (std::min)(255, ires)));
}

template<typename data_t = float>
void resize_bilinear(const Blob::Ptr inBlob, Blob::Ptr outBlob, uint8_t* buffer) {
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

    auto *sptr = static_cast<data_t*>(inBlob->buffer()) + inBlob->getTensorDesc().getBlockingDesc().getOffsetPadding();
    auto *dptr = static_cast<data_t*>(outBlob->buffer()) + outBlob->getTensorDesc().getBlockingDesc().getOffsetPadding();
    auto sstep = static_cast<const int>(inBlob->getTensorDesc().getBlockingDesc().getStrides()[2]);
    auto dstep = static_cast<const int>(outBlob->getTensorDesc().getBlockingDesc().getStrides()[2]);
    auto scale_x = static_cast<float>(src_full_width) / dst_full_width;
    auto scale_y = static_cast<float>(src_full_height) / dst_full_height;

    auto* xofs = reinterpret_cast<int32_t*>(buffer);
    auto* yofs = xofs + dwidth;
    auto* alpha = reinterpret_cast<float*>(yofs + dheight);
    auto* beta = alpha + dwidth;
    auto* tptr = beta + dheight;

    for (int dx = dst_go_x; dx < dst_go_x + dwidth; dx++) {
        auto fx = static_cast<float>((dx + 0.5) * scale_x - 0.5);
        int32_t sx = static_cast<int32_t>(floor(fx));
        fx -= sx;

        int32_t sx0 = sx;
        if (sx < 0 && border.type == BORDER_REPLICATE) {
            fx = 0;
            sx0 = 0;
        }

        if (sx >= src_full_width - 1 && border.type == BORDER_REPLICATE) {
            fx = 1.f;
            sx0 = (std::max)(src_full_width - 2, 0);
        }

        xofs[dx - dst_go_x] = sx0 - src_go_x;
        alpha[dx - dst_go_x] = fx;
    }

    for (int dy = dst_go_y; dy < dst_go_y + dheight; dy++) {
        auto fy = static_cast<float>((dy + 0.5) * scale_y - 0.5);
        int32_t sy = static_cast<int32_t>(floor(fy));
        fy -= sy;

        int32_t sy0 = sy;
        if (sy < 0 && border.type == BORDER_REPLICATE) {
            fy = 0;
            sy0 = 0;
        }

        if (sy >= src_full_height - 1 && border.type == BORDER_REPLICATE) {
            fy = 1.f;
            sy0 = (std::max)(src_full_height - 2, 0);
        }

        yofs[dy - dst_go_y] = sy0 - src_go_y;
        beta[dy - dst_go_y] = fy;
    }

    auto full_pass = [&](int c, int y) {
        auto sptr_ = sptr + c * origSrcW * origSrcH;
        auto dptr_ = dptr + c * origDstW * origDstH;
        auto tptr_ = tptr;

        for (int x = 0; x < swidth; x++) {
            bool use_constant0 = yofs[y] + 0 < 0 || yofs[y] + 0 >= src_full_height;
            bool use_constant1 = yofs[y] + 1 < 0 || yofs[y] + 1 >= src_full_height;
            float val0 = static_cast<float>(use_constant0 ? border.value : sptr_[(yofs[y] + 0) * sstep + x]);
            float val1 = static_cast<float>(use_constant1 ? border.value : sptr_[(yofs[y] + 1) * sstep + x]);

            float res = val0 + beta[y] * (val1 - val0);
            tptr_[x] = res;
        }

        for (int x = 0; x < dwidth; x++) {
            bool use_constant0 = xofs[x] + 0 < 0 || xofs[x] + 0 >= src_full_width;
            bool use_constant1 = xofs[x] + 1 < 0 || xofs[x] + 1 >= src_full_width;
            float val0 = use_constant0 ? border.value : tptr_[xofs[x] + 0];
            float val1 = use_constant1 ? border.value : tptr_[xofs[x] + 1];

            float res = val0 + alpha[x] * (val1 - val0);
            dptr_[y * dstep + x] = saturate_cast<data_t>(res);
        }
    };

    for (int c = 0; c < channels; c++) {
        for (int y = 0; y < dheight; y++) {
            full_pass(c, y);
        }
    }
}

int getResizeAreaTabSize(int dst_go, int ssize, int dsize, float scale) {
    static const float threshold = 1e-3f;
    int max_count = 0;

    for (int col = dst_go; col < dst_go + dsize; col++) {
        int count = 0;

        float fsx1 = col * scale;
        float fsx2 = fsx1 + scale;

        int sx1 = static_cast<int>(ceil(fsx1));
        int sx2 = static_cast<int>(floor(fsx2));

        sx2 = (std::min)(sx2, ssize - 1);
        sx1 = (std::min)(sx1, sx2);

        if (sx1 - fsx1 > threshold) {
            count++;
        }

        for (int sx = sx1; sx < sx2; sx++) {
            count++;
        }

        if (fsx2 - sx2 > threshold) {
            count++;
        }
        max_count = (std::max)(max_count, count);
    }

    return max_count;
}

void computeResizeAreaTab(int src_go, int dst_go, int ssize, int dsize, float scale,
                          uint16_t* si, uint16_t* alpha, int max_count) {
    static const float threshold = 1e-3f;
    int k = 0;

    for (int col = dst_go; col < dst_go + dsize; col++) {
        int count = 0;

        float fsx1 = col * scale;
        float fsx2 = fsx1 + scale;
        float cellWidth = (std::min)(scale, ssize - fsx1);

        int sx1 = static_cast<int>(ceil(fsx1));
        int sx2 = static_cast<int>(floor(fsx2));

        sx2 = (std::min)(sx2, ssize - 1);
        sx1 = (std::min)(sx1, sx2);

        si[col - dst_go] = (uint16_t)(sx1 - src_go);

        if (sx1 - fsx1 > threshold) {
            si[col - dst_go] = (uint16_t)(sx1 - src_go - 1);
            alpha[k++] = (uint16_t)((1 << 16) * ((sx1 - fsx1) / cellWidth));
            count++;
        }

        for (int sx = sx1; sx < sx2; sx++) {
            alpha[k++] = (uint16_t)((1 << 16) * (1.0f / cellWidth));
            count++;
        }

        if (fsx2 - sx2 > threshold) {
            alpha[k++] = (uint16_t)((1 << 16) * ((std::min)((std::min)(fsx2 - sx2, 1.f), cellWidth) / cellWidth));
            count++;
        }

        if (count != max_count) {
            alpha[k++] = 0;
        }
    }
}

void generate_alpha_and_id_arrays(int x_max_count, int dcols, const uint16_t* xalpha, uint16_t* xsi,
                                  uint16_t** alpha, uint16_t** sxid) {
    if (x_max_count <= 4) {
        for (int col = 0; col < dcols; col++) {
            for (int x = 0; x < x_max_count; x++) {
                alpha[x][col] = xalpha[col*x_max_count + x];
            }
        }
    }
    if (x_max_count <= 4) {
        for (int col = 0; col <= dcols - 8; col += 8) {
            for (int chunk_num_h = 0; chunk_num_h < x_max_count; chunk_num_h++) {
                for (int i = 0; i < 128 / 16; i++) {
                    int id_diff = xsi[col + i] - xsi[col];

                    for (int chunk_num_v = 0; chunk_num_v < x_max_count; chunk_num_v++) {
                        uint16_t* sxidp = sxid[chunk_num_v] + col * x_max_count + chunk_num_h * 8;

                        int id0 = (id_diff + chunk_num_v) * 2 + 0;
                        int id1 = (id_diff + chunk_num_v) * 2 + 1;

                        (reinterpret_cast<int8_t*>(sxidp + i))[0] = static_cast<int8_t>(id0 >= (chunk_num_h * 16) && id0 < (chunk_num_h + 1) * 16 ? id0 : -1);
                        (reinterpret_cast<int8_t*>(sxidp + i))[1] = static_cast<int8_t>(id1 >= (chunk_num_h * 16) && id1 < (chunk_num_h + 1) * 16 ? id1 : -1);
                    }
                }
            }
        }
    }
}

int computeResizeAreaTabFP32(int src_go, int dst_go, int ssize, int dsize, float scale, uint16_t* si, uint16_t* di, float* alpha) {
    static const float threshold = 1e-3f;
    int k = 0;

    for (int col = dst_go; col < dst_go + dsize; col++) {
        float fsx1 = col * scale;
        float fsx2 = fsx1 + scale;
        float cellWidth = (std::min)(scale, ssize - fsx1);

        int sx1 = static_cast<int>(ceil(fsx1));
        int sx2 = static_cast<int>(floor(fsx2));

        sx2 = (std::min)(sx2, ssize - 1);
        sx1 = (std::min)(sx1, sx2);

        if (sx1 - fsx1 > threshold) {
            di[k] = (uint16_t)(col - dst_go);
            si[k] = (uint16_t)(sx1 - src_go - 1);
            alpha[k++] = (sx1 - fsx1) / cellWidth;
        }

        for (int sx = sx1; sx < sx2; sx++) {
            di[k] = (uint16_t)(col - dst_go);
            si[k] = (uint16_t)(sx  - src_go);
            alpha[k++] = 1.0f / cellWidth;
        }

        if (fsx2 - sx2 > threshold) {
            di[k] = (uint16_t)(col - dst_go);
            si[k] = (uint16_t)(sx2 - src_go);
            alpha[k++] = (std::min)((std::min)(fsx2 - sx2, 1.f), cellWidth) / cellWidth;
        }
    }
    return k;
}

template<typename data_t = float>
void resize_area_downscale(const Blob::Ptr inBlob, Blob::Ptr outBlob, uint8_t* buffer) {
    auto dstDims = outBlob->getTensorDesc().getDims();
    auto srcDims = inBlob->getTensorDesc().getDims();

    auto src_strides = inBlob->getTensorDesc().getBlockingDesc().getStrides();
    auto dst_strides = outBlob->getTensorDesc().getBlockingDesc().getStrides();
    auto origSrcW = src_strides[2];
    auto origSrcH = src_strides[1] / src_strides[2];
    auto origDstW = dst_strides[2];
    auto origDstH = dst_strides[1] / dst_strides[2];

    auto dwidth = static_cast<const int>(dstDims[3]);
    auto dheight = static_cast<const int>(dstDims[2]);
    auto swidth = static_cast<const int>(srcDims[3]);
    auto sheight = static_cast<const int>(srcDims[2]);
    auto channels = static_cast<const int>(srcDims[1]);

    const int src_go_x = 0;
    const int src_go_y = 0;
    const int dst_go_x = 0;
    const int dst_go_y = 0;

    auto src_full_width = static_cast<const int>(srcDims[3]);
    auto src_full_height = static_cast<const int>(srcDims[2]);
    auto dst_full_width = static_cast<const int>(dstDims[3]);
    auto dst_full_height = static_cast<const int>(dstDims[2]);

    auto* sptr = static_cast<const data_t*>(inBlob->buffer()) + inBlob->getTensorDesc().getBlockingDesc().getOffsetPadding();
    auto* dptr = static_cast<data_t*>(outBlob->buffer()) + outBlob->getTensorDesc().getBlockingDesc().getOffsetPadding();

    auto sstep = static_cast<const int>(src_strides[2]);
    auto dstep = static_cast<const int>(dst_strides[2]);

    float scale_x = static_cast<float>(src_full_width) / dst_full_width;
    float scale_y = static_cast<float>(src_full_height) / dst_full_height;

    int vert_sum_size = swidth;
    int tabofs_size = (std::max)(2*swidth, 2*dwidth);
    int xsi_size = (std::max)(2*swidth, 2*dwidth);
    int xdi_size = (std::max)(2*swidth, 2*dwidth);
    int ysi_size = (std::max)(2*sheight, 2*dheight);
    int ydi_size = (std::max)(2*sheight, 2*dheight);
    int xalpha_size = (std::max)(2*swidth, 2*dwidth);

    auto vert_sum = reinterpret_cast<float*>(buffer);
    auto tabofs = reinterpret_cast<int*>(vert_sum + vert_sum_size);
    auto xsi = reinterpret_cast<uint16_t*>(tabofs + tabofs_size + 1);
    auto xdi = xsi + xsi_size;
    auto ysi = xdi + xdi_size;
    auto ydi = ysi + ysi_size;
    auto xalpha = reinterpret_cast<float*>(ydi + ydi_size);
    auto yalpha = xalpha + xalpha_size;

    int ytab_size = computeResizeAreaTabFP32(src_go_y, dst_go_y, src_full_height, dheight, scale_y, ysi, ydi, yalpha);
    int xtab_size = computeResizeAreaTabFP32(src_go_x, dst_go_x, src_full_width,  dwidth,  scale_x, xsi, xdi, xalpha);

    int dy_ = 0;
    for (int i = 0; i < ytab_size && dy_ < dwidth*2; i++) {
        if (i == 0 || ydi[i] != ydi[i-1]) {
            tabofs[dy_++] = i;
        }
    }
    tabofs[dy_] = ytab_size;

    auto full_pass = [&](const data_t* sptr_, data_t* dptr_, int y) {
        auto vert_sum_ = vert_sum;

        memset(vert_sum_, 0, swidth * sizeof(float));

        data_t *pdst = dptr_ + y * dstep;

        for (int dy = tabofs[y]; dy < tabofs[y + 1] && dy < ytab_size; dy++) {
            float beta = yalpha[dy];
            int sy = ysi[dy];

            const data_t *psrc = sptr_ + sy * sstep;
            for (int x = 0; x < swidth; x++) {
                vert_sum_[x] += beta * psrc[x];
            }
        }

        int xtab_ind = 0;
        for (int x = 0; x < dwidth; x++) {
            float res = 0.f;
            int dx = 0;
            for (; x == xdi[xtab_ind + dx] && xtab_ind + dx < xtab_size; dx++) {
                float alpha = xalpha[xtab_ind + dx];
                int sx = xsi[xtab_ind + dx];

                res += alpha * vert_sum_[sx];
            }

            pdst[x] = saturate_cast<data_t>(res);
            xtab_ind += dx;
        }
    };

    for (int ch = 0; ch < channels; ch++) {
        for (int y = 0; y < dheight; y++) {
            auto sptr_ = sptr + ch * origSrcH * origSrcW;
            auto dptr_ = dptr + ch * origDstH * origDstW;

            full_pass(sptr_, dptr_, y);
        }
    }
}

inline int clip(int x, int a, int b) {
    return x >= a ? (x < b ? x : b-1) : a;
}

const int MAX_ESIZE = 16;

template<typename data_t>
void HResizeLinear(const data_t** src, float** dst, int count, const int* xofs, const float* alpha,
                 int swidth, int dwidth, int cn, int xmin, int xmax ) {
    int dx, k;
    int dx0 = 0;

    for (k = 0; k <= count - 2; k++) {
        const data_t *S0 = src[k], *S1 = src[k+1];
        float *D0 = dst[k], *D1 = dst[k+1];
        for (dx = dx0; dx < xmax; dx++) {
            int sx = xofs[dx];
            float a0 = alpha[dx*2], a1 = alpha[dx*2+1];
            float t0 = static_cast<float>(S0[sx])*a0 + static_cast<float>(S0[sx + cn])*a1;
            float t1 = static_cast<float>(S1[sx])*a0 + static_cast<float>(S1[sx + cn])*a1;
            D0[dx] = t0; D1[dx] = t1;
        }

        for (; dx < dwidth; dx++) {
            int sx = xofs[dx];
            D0[dx] = static_cast<float>(S0[sx]); D1[dx] = static_cast<float>(S1[sx]);
        }
    }

    for (; k < count; k++) {
        const data_t *S = src[k];
        float *D = dst[k];
        for (dx = 0; dx < xmax; dx++) {
            int sx = xofs[dx];
            D[dx] = static_cast<float>(S[sx])*alpha[dx*2] + static_cast<float>(S[sx+cn])*alpha[dx*2+1];
        }

        for (; dx < dwidth; dx++)
            D[dx] = static_cast<float>(S[xofs[dx]]);
    }
}

template<typename data_t>
void VResizeLinear(float** src, data_t* dst, const float* beta, int width) {
    float b0 = beta[0], b1 = beta[1];
    const float *S0 = src[0], *S1 = src[1];

    if (sizeof(data_t) == 4) {
        for (int x = 0; x < width; x++)
            dst[x] = static_cast<data_t>(S0[x] * b0 + S1[x] * b1);
    } else {
        for (int x = 0; x < width; x++)
            dst[x] = saturateU32toU8(static_cast<uint32_t>(S0[x] * b0 + S1[x] * b1));
    }
}

template<typename data_t>
static void resize_area_upscale(const Blob::Ptr inBlob, Blob::Ptr outBlob, uint8_t* buffer) {
    auto dstDims = outBlob->getTensorDesc().getDims();
    auto srcDims = inBlob->getTensorDesc().getDims();

    auto src_strides = inBlob->getTensorDesc().getBlockingDesc().getStrides();
    auto dst_strides = outBlob->getTensorDesc().getBlockingDesc().getStrides();
    auto origSrcW = src_strides[2];
    auto origSrcH = src_strides[1] / src_strides[2];
    auto origDstW = dst_strides[2];
    auto origDstH = dst_strides[1] / dst_strides[2];

    auto dwidth = static_cast<const int>(dstDims[3]);
    auto dheight = static_cast<const int>(dstDims[2]);
    auto swidth = static_cast<const int>(srcDims[3]);
    auto sheight = static_cast<const int>(srcDims[2]);
    auto channels = static_cast<const int>(srcDims[1]);

    auto src_full_width = static_cast<const int>(srcDims[3]);
    auto src_full_height = static_cast<const int>(srcDims[2]);
    auto dst_full_width = static_cast<const int>(dstDims[3]);
    auto dst_full_height = static_cast<const int>(dstDims[2]);

    auto sptr = static_cast<const data_t*>(inBlob->buffer()) + inBlob->getTensorDesc().getBlockingDesc().getOffsetPadding();
    auto dptr = static_cast<data_t*>(outBlob->buffer()) + outBlob->getTensorDesc().getBlockingDesc().getOffsetPadding();

    auto sstep = static_cast<const int>(src_strides[2]);
    auto dstep = static_cast<const int>(dst_strides[2]);

    float scale_x = static_cast<float>(src_full_width)  / dst_full_width;
    float scale_y = static_cast<float>(src_full_height) / dst_full_height;
    float inv_scale_x = static_cast<float>(dst_full_width) / src_full_width;
    float inv_scale_y = static_cast<float>(dst_full_height) / src_full_height;

    int xmin = 0, xmax = dwidth, width = dwidth;
    int ksize = 2;
    int ksize2 = ksize/2;

    auto xofs = reinterpret_cast<int*>(buffer);
    auto yofs = xofs + width;
    auto alpha = reinterpret_cast<float*>(yofs + dheight);
    auto beta = alpha + width*ksize;
    float cbuf[2] = {0};

    for (int dx = 0; dx < dwidth; dx++) {
        int sx = static_cast<int>(floor(dx*scale_x));
        float fx = (dx+1) - (sx+1)*inv_scale_x;
        fx = fx <= 0 ? 0.f : fx - floor(fx);

        if (sx < ksize2-1) {
            xmin = dx+1;
            if (sx < 0)
                fx = 0, sx = 0;
        }

        if (sx + ksize2 >= swidth) {
            xmax = (std::min)(xmax, dx);
            if (sx >= swidth-1)
                fx = 0, sx = swidth-1;
        }

        xofs[dx] = sx;

        cbuf[0] = 1.f - fx;
        cbuf[1] = fx;

        for (int k = 0; k < ksize; k++)
            alpha[dx*ksize + k] = cbuf[k];
    }

    for (int dy = 0; dy < dheight; dy++) {
        int sy = static_cast<int>(floor(dy*scale_y));
        float fy = (dy+1) - (sy+1)*inv_scale_y;
        fy = fy <= 0 ? 0.f : fy - floor(fy);

        yofs[dy] = sy;
        cbuf[0] = 1.f - fy;
        cbuf[1] = fy;

        for (int k = 0; k < ksize; k++)
            beta[dy*ksize + k] = cbuf[k];
    }

    auto full_pass = [&](const data_t* sptr_, data_t* dptr_, int dy) {
        int bufstep = dwidth;
        const data_t* srows[MAX_ESIZE]={0};
        float* rows[MAX_ESIZE]={0};
        int prev_sy[MAX_ESIZE];

        for (int k = 0; k < ksize; k++) {
            prev_sy[k] = -1;
            rows[k] = reinterpret_cast<float*>(buffer + (width + dheight)*(sizeof(int) + sizeof(float)*ksize))
                      + k*bufstep;
        }

        int sy0 = yofs[dy], k0 = ksize, k1 = 0;

        for (int k = 0; k < ksize; k++) {
            int sy = clip(sy0 - ksize2 + 1 + k, 0, sheight);
            for (k1 = (std::max)(k1, k); k1 < ksize; k1++) {
                if (k1 < MAX_ESIZE && sy == prev_sy[k1]) {
                    if (k1 > k)
                        memcpy(rows[k], rows[k1], bufstep*sizeof(rows[0][0]));
                    break;
                }
            }

            if (k1 == ksize)
                k0 = (std::min)(k0, k);
            srows[k] = sptr_ + sy * sstep;
            prev_sy[k] = sy;
        }

        if (k0 < ksize)
            HResizeLinear<data_t>(srows + k0, reinterpret_cast<float**>(rows + k0), ksize - k0, xofs,
                                  reinterpret_cast<const float*>(alpha), swidth, dwidth, 1, xmin, xmax);

        VResizeLinear<data_t>(reinterpret_cast<float**>(rows), dptr_ + dstep*dy, beta + dy*ksize, dwidth);
    };

    for (int ch = 0; ch < channels; ch++) {
        for (int dy = 0; dy < dheight; dy++) {
            auto sptr_ = sptr + ch * origSrcH * origSrcW;
            auto dptr_ = dptr + ch * origDstH * origDstW;

            full_pass(sptr_, dptr_, dy);
        }
    }
}

size_t resize_get_buffer_size(Blob::Ptr inBlob, Blob::Ptr outBlob, const ResizeAlgorithm &algorithm) {
    auto dstDims = outBlob->getTensorDesc().getDims();
    auto srcDims = inBlob->getTensorDesc().getDims();

    SizeVector strides = inBlob->getTensorDesc().getBlockingDesc().getStrides();
    size_t origW = strides[2];
    size_t origH = strides[1] / strides[2];

    const int src_full_width = static_cast<int>(origW);
    const int src_full_height = static_cast<int>(origH);
    const int dst_full_width = static_cast<int>(dstDims[3]);
    const int dst_full_height = static_cast<int>(dstDims[2]);

    float scale_x = static_cast<float>(dstDims[3]) / srcDims[3];
    float scale_y = static_cast<float>(dstDims[2]) / srcDims[2];

    auto resize_bilinear_u8_buffer_size = [&]() {
        size_t buffer_size = (sizeof(int16_t) * 4 + sizeof(uint8_t *)) * dstDims[3] +
                             (sizeof(int32_t) + sizeof(int16_t)) * dstDims[2] +
                             sizeof(uint32_t) * dstDims[3] +
                             (((srcDims[3] + 7) / 8) * 8 * 8) +
                             sizeof(uint8_t) * 12;

        return buffer_size;
    };

    auto resize_bilinear_fp32_buffer_size = [&]() {
        size_t buffer_size = (sizeof(float) + sizeof(float *)) * dstDims[3] +
                             (sizeof(int32_t) + sizeof(float)) * dstDims[2] +
                             (((srcDims[3] + 1) / 2) * 2 * 2) * sizeof(float);

        return buffer_size;
    };

    auto resize_area_u8_downscale_sse_buffer_size = [&]() {
        const int dwidth = static_cast<int>(dstDims[3]);
        const int dheight = static_cast<int>(dstDims[2]);
        const int swidth = static_cast<int>(srcDims[3]);

        const int dst_go_x = 0;
        const int dst_go_y = 0;

        int x_max_count = getResizeAreaTabSize(dst_go_x, src_full_width, dwidth, static_cast<float>(src_full_width) / dst_full_width) + 1;
        int y_max_count = getResizeAreaTabSize(dst_go_y, src_full_height, dheight, static_cast<float>(src_full_height) / dst_full_height) + 1;

        size_t si_buf_size = sizeof(uint16_t) * dwidth + sizeof(uint16_t) * dheight;
        size_t alpha_buf_size =
                sizeof(uint16_t) * (dwidth * x_max_count + 8 * 16) + sizeof(uint16_t) * dheight * y_max_count;
        size_t vert_sum_buf_size = sizeof(uint16_t) * (swidth * 2);
        size_t alpha_array_buf_size = sizeof(uint16_t) * 4 * dwidth;
        size_t sxid_array_buf_size = sizeof(uint16_t) * 4 * 4 * dwidth;

        size_t buffer_size = si_buf_size +
                             alpha_buf_size +
                             vert_sum_buf_size +
                             alpha_array_buf_size +
                             sxid_array_buf_size;

        return buffer_size;
    };

    auto resize_area_downscale_buffer_size = [&]() {
        size_t buffer_size = sizeof(float) * (srcDims[3]) +
                             sizeof(uint32_t) * (dstDims[3] * 2 + 1) +
                             sizeof(float) * ((srcDims[3] + srcDims[2]) * 4) +
                             sizeof(float) * ((srcDims[3] + srcDims[2]) * 2);

        return buffer_size;
    };

    auto resize_area_upscale_buffer_size = [&]() {
        size_t buffer_size = (dstDims[3] + dstDims[2])*(sizeof(int) + sizeof(float)*2) + 2*dstDims[3] * sizeof(float);

        return buffer_size;
    };

    if (algorithm == RESIZE_BILINEAR) {
        if (inBlob->getTensorDesc().getPrecision() == Precision::U8) {
            return resize_bilinear_u8_buffer_size();
        } else {
            return resize_bilinear_fp32_buffer_size();
        }
    } else if (algorithm == RESIZE_AREA) {
        if (inBlob->getTensorDesc().getPrecision() == Precision::U8) {
            if (scale_x <= 1 && scale_y <= 1) {
#ifdef HAVE_SSE
                if (with_cpu_x86_sse42() && scale_x < 1 && scale_y < 1)
                    return resize_area_u8_downscale_sse_buffer_size();
                else
#endif
                    return resize_area_downscale_buffer_size();
            } else {
                return resize_area_upscale_buffer_size();
            }
        } else {
            if (scale_x <= 1 && scale_y <= 1)
                return resize_area_downscale_buffer_size();
            else
                return resize_area_upscale_buffer_size();
        }
    }

    return 0;
}

void resize(Blob::Ptr inBlob, Blob::Ptr outBlob, const ResizeAlgorithm &algorithm) {
    if (inBlob->getTensorDesc().getLayout() != NCHW || outBlob->getTensorDesc().getLayout() != NCHW)
        THROW_IE_EXCEPTION << "Resize supports only NCHW layout";

    if (!((inBlob->getTensorDesc().getPrecision() == Precision::U8 && outBlob->getTensorDesc().getPrecision() == Precision::U8) ||
          (inBlob->getTensorDesc().getPrecision() == Precision::FP32 && outBlob->getTensorDesc().getPrecision() == Precision::FP32)))
        THROW_IE_EXCEPTION << "Resize supports only U8 and FP32 precisions";

    if (algorithm != RESIZE_BILINEAR && algorithm != RESIZE_AREA)
        THROW_IE_EXCEPTION << "Unsupported resize algorithm type";

    size_t buffer_size = resize_get_buffer_size(inBlob, outBlob, algorithm);
    auto* buffer = static_cast<uint8_t *>(malloc(buffer_size));
    if (buffer == nullptr) {
        THROW_IE_EXCEPTION << "Could not allocate memory for blob";
    }

    auto dstDims = outBlob->getTensorDesc().getDims();
    auto srcDims = inBlob->getTensorDesc().getDims();
    float scale_x = static_cast<float>(dstDims[3]) / srcDims[3];
    float scale_y = static_cast<float>(dstDims[2]) / srcDims[2];

    if (algorithm == RESIZE_BILINEAR) {
        if (inBlob->getTensorDesc().getPrecision() == Precision::U8) {
#ifdef HAVE_SSE
            if (with_cpu_x86_sse42())
                Resize::resize_bilinear_u8(inBlob, outBlob, buffer);
            else
#endif
                resize_bilinear<uint8_t>(inBlob, outBlob, buffer);
        } else {
            resize_bilinear<float>(inBlob, outBlob, buffer);
        }
    } else if (algorithm == RESIZE_AREA) {
        if (inBlob->getTensorDesc().getPrecision() == Precision::U8) {
            if (scale_x <= 1 && scale_y <= 1) {
#ifdef HAVE_SSE
                if (with_cpu_x86_sse42() && scale_x < 1 && scale_y < 1)
                    Resize::resize_area_u8_downscale(inBlob, outBlob, buffer);
                else
#endif
                    resize_area_downscale<uint8_t>(inBlob, outBlob, buffer);
            } else {
                resize_area_upscale<uint8_t>(inBlob, outBlob, buffer);
            }
        } else {
            if (scale_x <= 1 && scale_y <= 1)
                resize_area_downscale<float>(inBlob, outBlob, buffer);
            else
                resize_area_upscale<float>(inBlob, outBlob, buffer);
        }
    }

    free(buffer);
}

}  // namespace Resize

//----------------------------------------------------------------------

using namespace Resize;

/**
 * @brief This class stores pre-process information for exact input
 */
class PreProcessData : public IPreProcessData {
    /**
     * @brief ROI blob.
     */
    Blob::Ptr _userBlob = nullptr;
    Blob::Ptr _tmp1 = nullptr;
    Blob::Ptr _tmp2 = nullptr;

    /**
     * @brief Pointer-to-implementation (PIMPL) hiding preprocessing implementation details.
     * BEWARE! Will be shared among copies!
     */
    std::shared_ptr<PreprocEngine> _preproc;

public:
    void setRoiBlob(const Blob::Ptr &blob) override;

    Blob::Ptr getRoiBlob() const override;

    void execute(Blob::Ptr &preprocessedBlob, const PreProcessInfo &info, bool serial, int batchSize = -1) override;

    void Release() noexcept override;

    void isApplicable(const Blob::Ptr &src, const Blob::Ptr &dst) override;
};

StatusCode CreatePreProcessData(IPreProcessData *& data, ResponseDesc * /*resp*/) noexcept {
    data = new PreProcessData();
    return StatusCode::OK;
}

void PreProcessData::Release() noexcept {
    delete this;
}

void PreProcessData::setRoiBlob(const Blob::Ptr &blob) {
    _userBlob = blob;
}

Blob::Ptr PreProcessData::getRoiBlob() const {
    return _userBlob;
}

void PreProcessData::execute(Blob::Ptr &preprocessedBlob, const PreProcessInfo &info, bool serial,
        int batchSize) {
    OV_ITT_SCOPED_TASK(itt::domains::IEPreproc, "Preprocessing");

    auto algorithm = info.getResizeAlgorithm();
    auto fmt = info.getColorFormat();

    if (_userBlob == nullptr || preprocessedBlob == nullptr) {
        THROW_IE_EXCEPTION << "Input pre-processing is called with null " << (_userBlob == nullptr ? "_userBlob" : "preprocessedBlob");
    }

    batchSize = PreprocEngine::getCorrectBatchSize(batchSize, _userBlob);

    if (!_preproc) {
        _preproc.reset(new PreprocEngine);
    }

    if (_preproc->preprocessWithGAPI(_userBlob, preprocessedBlob, algorithm, fmt, serial, batchSize)) {
        return;
    }

    if (algorithm == NO_RESIZE) {
       THROW_IE_EXCEPTION << "Input pre-processing is called without the pre-processing info set: "
                             "there's nothing to be done";
    }

    if (batchSize > 1) {
        THROW_IE_EXCEPTION << "Batch pre-processing is unsupported in this mode. "
                              "Use default pre-processing instead to process batches.";
    }

    if (fmt != ColorFormat::RAW) {
        THROW_IE_EXCEPTION << "Non-default (not ColorFormat::RAW) color formats are unsupported "
                              "in this mode. Use default pre-processing instead to process color "
                              "formats.";
    }

    Blob::Ptr res_in, res_out;
    if (_userBlob->getTensorDesc().getLayout() == NHWC) {
        if (!_tmp1 || _tmp1->size() != _userBlob->size()) {
            if (_userBlob->getTensorDesc().getPrecision() == Precision::FP32) {
                _tmp1 = make_shared_blob<float>({Precision::FP32, _userBlob->getTensorDesc().getDims(), Layout::NCHW});
            } else {
                _tmp1 = make_shared_blob<uint8_t>({Precision::U8, _userBlob->getTensorDesc().getDims(), Layout::NCHW});
            }
            _tmp1->allocate();
        }

        {
            OV_ITT_SCOPED_TASK(itt::domains::IEPreproc, "Reorder before");
            blob_copy(_userBlob, _tmp1);
        }
        res_in = _tmp1;
    } else {
        res_in = _userBlob;
    }

    if (preprocessedBlob->getTensorDesc().getLayout() == NHWC) {
        if (!_tmp2 || _tmp2->size() != preprocessedBlob->size()) {
            if (preprocessedBlob->getTensorDesc().getPrecision() == Precision::FP32) {
                _tmp2 = make_shared_blob<float>({Precision::FP32, preprocessedBlob->getTensorDesc().getDims(), Layout::NCHW});
            } else {
                _tmp2 = make_shared_blob<uint8_t>({Precision::U8, preprocessedBlob->getTensorDesc().getDims(), Layout::NCHW});
            }
            _tmp2->allocate();
        }
        res_out = _tmp2;
    } else {
        res_out = preprocessedBlob;
    }

    {
        OV_ITT_SCOPED_TASK(itt::domains::IEPreproc, "Resize");
        resize(res_in, res_out, algorithm);
    }

    if (res_out == _tmp2) {
        OV_ITT_SCOPED_TASK(itt::domains::IEPreproc, "Reorder after");
        blob_copy(_tmp2, preprocessedBlob);
    }
}

void PreProcessData::isApplicable(const Blob::Ptr &src, const Blob::Ptr &dst) {
    // if G-API pre-processing is used, let it check that pre-processing is applicable
    if (PreprocEngine::useGAPI()) {
        PreprocEngine::checkApplicabilityGAPI(src, dst);
        return;
    }

    if (!src->is<MemoryBlob>() || !dst->is<MemoryBlob>()) {
        THROW_IE_EXCEPTION << "Preprocessing is not applicable. Source and destination blobs must "
                              "be memory blobs";
    }

    auto &src_dims = src->getTensorDesc().getDims();
    auto &dst_dims = dst->getTensorDesc().getDims();

    if (src_dims.size() != dst_dims.size())
        THROW_IE_EXCEPTION << "Preprocessing is not applicable. Source and destination blobs have different "
                              "number of dimensions";

    if (src_dims.size() != 4)
        THROW_IE_EXCEPTION << "Preprocessing is not applicable. Only 4D tensors are supported.";

    if (src_dims[0] != dst_dims[0] || src_dims[1] != dst_dims[1])
        THROW_IE_EXCEPTION << "Preprocessing is not applicable. Wrong shape. Network expected 4D input tensor with "
                              "shape [" << dst_dims[0] << "," << dst_dims[1] <<",H,W] but provided tensor has "
                              "shape "  << details::dumpVec(src_dims) << ".";
}

}  // namespace InferenceEngine
