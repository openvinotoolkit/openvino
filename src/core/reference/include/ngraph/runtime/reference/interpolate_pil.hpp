// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// The implementation for BILINEAR_PILLOW and BICUBIC_PILLOW is based on the
// Pillow library code from:
// https://github.com/python-pillow/Pillow/blob/main/src/libImaging/Resample.c

// The Python Imaging Library (PIL) is

//     Copyright © 1997-2011 by Secret Labs AB
//     Copyright © 1995-2011 by Fredrik Lundh

// Pillow is the friendly PIL fork. It is

//     Copyright © 2010-2023 by Jeffrey A. Clark (Alex) and contributors.

// Like PIL, Pillow is licensed under the open source HPND License:

// By obtaining, using, and/or copying this software and/or its associated
// documentation, you agree that you have read, understood, and will comply
// with the following terms and conditions:

// Permission to use, copy, modify and distribute this software and its
// documentation for any purpose and without fee is hereby granted,
// provided that the above copyright notice appears in all copies, and that
// both that copyright notice and this permission notice appear in supporting
// documentation, and that the name of Secret Labs AB or the author not be
// used in advertising or publicity pertaining to distribution of the software
// without specific, written prior permission.

// SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
// SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.
// IN NO EVENT SHALL SECRET LABS AB OR THE AUTHOR BE LIABLE FOR ANY SPECIAL,
// INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
// LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
// OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
// PERFORMANCE OF THIS SOFTWARE.

#pragma once

#include <algorithm>
#include <cmath>

#include "ngraph/op/interpolate.hpp"
#include "ngraph/shape_util.hpp"

namespace ngraph {
namespace runtime {
namespace reference {

// #define ROUND_UP(f) ((int)((f) >= 0.0 ? (f) + 0.5F : (f)-0.5F))

template <typename T_out, typename T_in>
T_out round_up(T_in f) {
    return (T_out)(f >= 0.0 ? f + 0.5F : f - 0.5F);
}

struct filter {
    double (*filter)(double x);
    double support;
};

static inline double bilinear_filter(double x) {
    if (x < 0.0) {
        x = -x;
    }
    if (x < 1.0) {
        return 1.0 - x;
    }
    return 0.0;
}

static inline double bicubic_filter(double x) {
#define a -0.5
    if (x < 0.0) {
        x = -x;
    }
    if (x < 1.0) {
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
    }
    if (x < 2.0) {
        return (((x - 5) * x + 8) * x - 4) * a;
    }
    return 0.0;
#undef a
}

// static struct filter BILINEAR = {bilinear_filter, 1.0};
// static struct filter BICUBIC = {bicubic_filter, 2.0};

static int precompute_coeffs(int inSize,
                             float in0,
                             float in1,
                             int outSize,
                             struct filter* filterp,
                             // std::function<double(double)> filterp,
                             int** boundsp,
                             double** kkp) {
    double support, scale, filterscale;
    double center, ww, ss;
    int xx, x, ksize, xmin, xmax;
    int* bounds;
    double *kk, *k;

    /* prepare for horizontal stretch */
    filterscale = scale = (double)(in1 - in0) / outSize;
    if (filterscale < 1.0) {
        filterscale = 1.0;
    }

    /* determine support size (length of resampling filter) */
    support = filterp->support * filterscale;

    /* maximum number of coeffs */
    ksize = (int)ceil(support) * 2 + 1;

    // check for overflow
    // if (outSize > INT_MAX / (ksize * (int)sizeof(double))) {
    //     return 0;
    // }

    /* coefficient buffer */
    /* malloc check ok, overflow checked above */
    kk = (double*)malloc(outSize * ksize * sizeof(double));
    if (!kk) {
        // ImagingError_MemoryError();
        return 0;
    }

    /* malloc check ok, ksize*sizeof(double) > 2*sizeof(int) */
    bounds = (int*)malloc(outSize * 2 * sizeof(int));
    if (!bounds) {
        free(kk);
        // ImagingError_MemoryError();
        return 0;
    }

    for (xx = 0; xx < outSize; xx++) {
        center = in0 + (xx + 0.5) * scale;
        ww = 0.0;
        ss = 1.0 / filterscale;
        // Round the value
        xmin = (int)(center - support + 0.5);
        if (xmin < 0) {
            xmin = 0;
        }
        // Round the value
        xmax = (int)(center + support + 0.5);
        if (xmax > inSize) {
            xmax = inSize;
        }
        xmax -= xmin;
        k = &kk[xx * ksize];
        for (x = 0; x < xmax; x++) {
            double w = filterp->filter((x + xmin - center + 0.5) * ss);
            k[x] = w;
            ww += w;
        }
        for (x = 0; x < xmax; x++) {
            if (ww != 0.0) {
                k[x] /= ww;
            }
        }
        // Remaining values should stay empty if they are used despite of xmax.
        for (; x < ksize; x++) {
            k[x] = 0;
        }
        bounds[xx * 2 + 0] = xmin;
        bounds[xx * 2 + 1] = xmax;
    }
    *boundsp = bounds;
    *kkp = kk;
    return ksize;
}

template <typename T>
void ImagingResampleHorizontal(T* imOut,
                               Shape imOutShape,
                               const T* imIn,
                               Shape imInShape,
                               int offset,
                               int ksize,
                               int* bounds,
                               double* kk) {
    double ss;
    int x, xmin, xmax;
    // int xx, yy, x, xmin, xmax;

    double* k;

    for (size_t yy = 0; yy < imOutShape[0]; yy++) {
        for (size_t xx = 0; xx < imOutShape[1]; xx++) {
            xmin = bounds[xx * 2 + 0];
            xmax = bounds[xx * 2 + 1];
            k = &kk[xx * ksize];
            ss = 0.0;
            for (x = 0; x < xmax; x++) {
                // ss += IMAGING_PIXEL_I(imIn, x + xmin, yy + offset) * k[x];
                size_t in_idx = ((yy + offset)) * imInShape[1] + (x + xmin);
                ss += imIn[in_idx] * k[x];
            }
            // IMAGING_PIXEL_I(imOut, xx, yy) = ROUND_UP(ss);
            size_t out_idx = (yy)*imOutShape[1] + xx;
            imOut[out_idx] = T(round_up<int, double>(ss));

            // TODO: Enable with precision tests
            // if (std::is_integral<T>()) {
            //     imOut[out_idx] = T(round_up<int, double>(ss));
            // }
            // else {
            //     imOut[out_idx] = T(ss);
            // }
        }
    }
}

template <typename T>
void ImagingResampleVertical(T* imOut,
                             Shape imOutShape,
                             const T* imIn,
                             Shape imInShape,
                             int offset,
                             int ksize,
                             int* bounds,
                             double* kk) {
    double ss;
    int y, ymin, ymax;
    double* k;

    for (size_t yy = 0; yy < imOutShape[0]; yy++) {
        ymin = bounds[yy * 2 + 0];
        ymax = bounds[yy * 2 + 1];
        k = &kk[yy * ksize];
        for (size_t xx = 0; xx < imOutShape[1]; xx++) {
            ss = 0.0;
            for (y = 0; y < ymax; y++) {
                // ss += IMAGING_PIXEL_I(imIn, xx, y + ymin) * k[y];
                size_t in_idx = ((y + ymin)) * imInShape[1] + xx;
                ss += imIn[in_idx] * k[y];
            }
            // IMAGING_PIXEL_I(imOut, xx, yy) = ROUND_UP(ss);
            // imOut[(imOutShape[0] + yy) * imOutShape[1] + xx] = T(round_up<int, double>(ss));

            size_t out_idx = (yy)*imOutShape[1] + xx;
            imOut[out_idx] = T(round_up<int, double>(ss));

            // TODO: Enable with precision tests
            // if (std::is_integral<T>()) {
            //     imOut[out_idx] = T(round_up<int, double>(ss));
            // }
            // else {
            //     imOut[out_idx] = T(ss);
            // }
        }
    }
}

template <typename T>
void ImagingResampleInner(const T* imIn,
                          size_t imIn_xsize,
                          size_t imIn_ysize,
                          size_t xsize,
                          size_t ysize,
                          struct filter* filterp,
                          // float box[4],
                          float* box,
                          T* imOut) {
    // ResampleFunction ResampleHorizontal,
    // ResampleFunction ResampleVertical) {

    int need_horizontal, need_vertical;
    int ybox_first, ybox_last;
    int ksize_horiz, ksize_vert;
    int *bounds_horiz, *bounds_vert;
    double *kk_horiz, *kk_vert;

    need_horizontal = xsize != imIn_xsize || box[0] || box[2] != xsize;
    need_vertical = ysize != imIn_ysize || box[1] || box[3] != ysize;

    ksize_horiz = precompute_coeffs(imIn_xsize, box[0], box[2], xsize, filterp, &bounds_horiz, &kk_horiz);
    if (!ksize_horiz) {
        free(bounds_horiz);
        free(kk_horiz);
        return;
    }

    ksize_vert = precompute_coeffs(imIn_ysize, box[1], box[3], ysize, filterp, &bounds_vert, &kk_vert);
    if (!ksize_vert) {
        free(bounds_vert);
        free(kk_vert);
        return;
    }

    // First used row in the source image
    ybox_first = bounds_vert[0];
    // Last used row in the source image
    ybox_last = bounds_vert[ysize * 2 - 2] + bounds_vert[ysize * 2 - 1];

    // auto out_elem_count = (ybox_last - ybox_first) * xsize;
    size_t imTemp_ysize = (ybox_last - ybox_first);
    // size_t imTemp_ysize = imIn_ysize;
    auto imTemp_elem_count = imTemp_ysize * xsize;
    auto imTemp = std::vector<T>(imTemp_elem_count, 0);

    /* two-pass resize, horizontal pass */
    if (need_horizontal) {
        // Shift bounds for vertical pass
        for (size_t i = 0; i < ysize; i++) {
            bounds_vert[i * 2] -= ybox_first;
        }

        // imTemp = ImagingNewDirty(imIn->mode, xsize, ybox_last - ybox_first);
        if (imTemp.size()) {
            ImagingResampleHorizontal(imTemp.data(),
                                      Shape{imTemp_ysize, xsize},
                                      imIn,
                                      Shape{imIn_ysize, imIn_xsize},
                                      ybox_first,
                                      ksize_horiz,
                                      bounds_horiz,
                                      kk_horiz);
            // ImagingResampleHorizontal(
            //     imTemp.data(), imIn, ybox_first, ksize_horiz, bounds_horiz, kk_horiz);
        }
        free(bounds_horiz);
        free(kk_horiz);
        // if (!imTemp.size()) {
        //     free(bounds_vert);
        //     free(kk_vert);
        //     return;
        // }
        // imOut = imIn = imTemp.data();
    } else {
        // Free in any case
        free(bounds_horiz);
        free(kk_horiz);
    }

    // std::copy(imTemp.begin(), imTemp.end(), imOut);

    /* vertical pass */
    if (need_vertical) {
        // imOut = ImagingNewDirty(imIn->mode, imIn->xsize, ysize);
        // imOut = std::vector<std::vector<T>>(ysize, std::vector<T>(xsize, 0));
        if (imOut) {
            /* imIn can be the original image or horizontally resampled one */
            if (need_horizontal) {
                ImagingResampleVertical(imOut,
                                        Shape{ysize, xsize},
                                        imTemp.data(),
                                        Shape{imTemp_ysize, xsize},
                                        0,
                                        ksize_vert,
                                        bounds_vert,
                                        kk_vert);
            } else {
                ImagingResampleVertical(imOut,
                                        Shape{ysize, xsize},
                                        imIn,
                                        Shape{imIn_ysize, imIn_xsize},
                                        0,
                                        ksize_vert,
                                        bounds_vert,
                                        kk_vert);
            }
        }
        /* it's safe to call ImagingDelete with empty value
           if previous step was not performed. */
        // ImagingDelete(imTemp);
        free(bounds_vert);
        free(kk_vert);

        if (!imOut) {
            return;
        }
    } else {
        // Free in any case
        free(bounds_vert);
        free(kk_vert);
    }

    /* none of the previous steps are performed, copying */
    if (!need_horizontal && !need_vertical) {
        std::copy(imIn, imIn + (imIn_xsize * imIn_ysize), imOut);
    } else if (need_horizontal && !need_vertical) {
        std::copy(imTemp.begin(), imTemp.end(), imOut);
    }

    return;
}

// void
// normalize_coeffs_8bpc(int outSize, int ksize, double *prekk) {
//     int x;
//     INT32 *kk;

//     // use the same buffer for normalized coefficients
//     kk = (INT32 *)prekk;

//     for (x = 0; x < outSize * ksize; x++) {
//         if (prekk[x] < 0) {
//             kk[x] = (int)(-0.5 + prekk[x] * (1 << PRECISION_BITS));
//         } else {
//             kk[x] = (int)(0.5 + prekk[x] * (1 << PRECISION_BITS));
//         }
//     }
// }

}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
