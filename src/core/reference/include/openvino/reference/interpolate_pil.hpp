// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// The implementation for BILINEAR_PILLOW and BICUBIC_PILLOW is based on the
// Pillow library code from:
// https://github.com/python-pillow/Pillow/blob/9.4.0/src/libImaging/Resample.c

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
#include <limits>

#include "openvino/core/shape.hpp"

namespace ov {
namespace reference {
namespace interpolate_pil {

struct filter {
    double (*filter)(double x, double coeff_a);
    double support;
    double coeff_a;
};

template <typename T_out, typename T_in>
T_out round_up(T_in x) {
    return (T_out)(x >= 0.0 ? x + 0.5F : x - 0.5F);
}

template <typename T_out, typename T_in>
T_out clip(const T_in& x,
           const T_out& min = std::numeric_limits<T_out>::min(),
           const T_out& max = std::numeric_limits<T_out>::max()) {
    return T_out(std::max(T_in(min), std::min(x, T_in(max))));
}

static inline double bilinear_filter(double x, double) {
    if (x < 0.0) {
        x = -x;
    }
    if (x < 1.0) {
        return 1.0 - x;
    }
    return 0.0;
}

static inline double bicubic_filter(double x, double a) {
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
}

static inline int precompute_coeffs(int in_size,
                                    float in0,
                                    float in1,
                                    int out_size,
                                    const filter& filterp,
                                    std::vector<int>& bounds,
                                    std::vector<double>& kk) {
    double support, scale, filterscale;
    double center, ww, ss;
    int xx, x, ksize, xmin, xmax;
    /* prepare for horizontal stretch */
    filterscale = scale = (double)(in1 - in0) / out_size;
    if (filterscale < 1.0) {
        filterscale = 1.0;
    }

    /* determine support size (length of resampling filter) */
    support = filterp.support * filterscale;

    /* maximum number of coeffs */
    ksize = (int)ceil(support) * 2 + 1;

    /* coefficient buffer */
    kk.resize(out_size * ksize);
    bounds.resize(out_size * 2);

    for (xx = 0; xx < out_size; xx++) {
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
        if (xmax > in_size) {
            xmax = in_size;
        }
        xmax -= xmin;
        double* k = &kk[xx * ksize];
        for (x = 0; x < xmax; x++) {
            double w = filterp.filter((x + xmin - center + 0.5) * ss, filterp.coeff_a);
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
    return ksize;
}

template <typename T>
void imaging_resample_horizontal(T* im_out,
                                 Shape im_out_shape,
                                 const T* im_in,
                                 Shape im_in_shape,
                                 int offset,
                                 int ksize,
                                 std::vector<int>& bounds,
                                 std::vector<double>& kk) {
    double ss;
    int x, xmin, xmax;
    double* k;

    for (size_t yy = 0; yy < im_out_shape[0]; yy++) {
        for (size_t xx = 0; xx < im_out_shape[1]; xx++) {
            xmin = bounds[xx * 2 + 0];
            xmax = bounds[xx * 2 + 1];
            k = &kk[xx * ksize];
            ss = 0.0;
            for (x = 0; x < xmax; x++) {
                size_t in_idx = (yy + offset) * im_in_shape[1] + (x + xmin);
                ss += im_in[in_idx] * k[x];
            }
            size_t out_idx = yy * im_out_shape[1] + xx;
            if (std::is_integral<T>()) {
                im_out[out_idx] = T(clip<T, int64_t>(round_up<int64_t, double>(ss)));
            } else {
                im_out[out_idx] = T(ss);
            }
        }
    }
}

template <typename T>
void imaging_resample_vertical(T* im_out,
                               Shape im_out_shape,
                               const T* im_in,
                               Shape im_in_shape,
                               int offset,
                               int ksize,
                               std::vector<int>& bounds,
                               std::vector<double>& kk) {
    double ss;
    int y, ymin, ymax;
    double* k;

    for (size_t yy = 0; yy < im_out_shape[0]; yy++) {
        ymin = bounds[yy * 2 + 0];
        ymax = bounds[yy * 2 + 1];
        k = &kk[yy * ksize];
        for (size_t xx = 0; xx < im_out_shape[1]; xx++) {
            ss = 0.0;
            for (y = 0; y < ymax; y++) {
                size_t in_idx = (y + ymin) * im_in_shape[1] + xx;
                ss += im_in[in_idx] * k[y];
            }
            size_t out_idx = yy * im_out_shape[1] + xx;
            if (std::is_integral<T>()) {
                im_out[out_idx] = T(clip<T, int64_t>(round_up<int64_t, double>(ss)));
            } else {
                im_out[out_idx] = T(ss);
            }
        }
    }
}

template <typename T>
void imaging_resample_inner(const T* im_in,
                            size_t im_in_xsize,
                            size_t im_in_ysize,
                            size_t xsize,
                            size_t ysize,
                            const filter& filterp,
                            float* box,
                            T* im_out) {
    int ybox_first, ybox_last;
    int ksize_horiz, ksize_vert;

    std::vector<int> bounds_horiz;
    std::vector<int> bounds_vert;
    std::vector<double> kk_horiz;
    std::vector<double> kk_vert;

    auto need_horizontal = xsize != im_in_xsize || bool(box[0]) || box[2] != xsize;
    auto need_vertical = ysize != im_in_ysize || bool(box[1]) || box[3] != ysize;

    ksize_horiz = precompute_coeffs(static_cast<int>(im_in_xsize),
                                    box[0],
                                    box[2],
                                    static_cast<int>(xsize),
                                    filterp,
                                    bounds_horiz,
                                    kk_horiz);
    ksize_vert = precompute_coeffs(static_cast<int>(im_in_ysize),
                                   box[1],
                                   box[3],
                                   static_cast<int>(ysize),
                                   filterp,
                                   bounds_vert,
                                   kk_vert);

    // First used row in the source image
    ybox_first = bounds_vert[0];
    // Last used row in the source image
    ybox_last = bounds_vert[ysize * 2 - 2] + bounds_vert[ysize * 2 - 1];

    size_t im_temp_ysize = (ybox_last - ybox_first);
    auto im_temp_elem_count = im_temp_ysize * xsize;
    auto im_temp = std::vector<T>(im_temp_elem_count, 0);

    /* two-pass resize, horizontal pass */
    if (need_horizontal) {
        // Shift bounds for vertical pass
        for (size_t i = 0; i < ysize; i++) {
            bounds_vert[i * 2] -= ybox_first;
        }

        if (im_temp.size() > 0) {
            imaging_resample_horizontal(im_temp.data(),
                                        Shape{im_temp_ysize, xsize},
                                        im_in,
                                        Shape{im_in_ysize, im_in_xsize},
                                        ybox_first,
                                        ksize_horiz,
                                        bounds_horiz,
                                        kk_horiz);
        }
    }

    /* vertical pass */
    if (need_vertical) {
        /* im_in can be the original image or horizontally resampled one */
        if (need_horizontal) {
            imaging_resample_vertical(im_out,
                                      Shape{ysize, xsize},
                                      im_temp.data(),
                                      Shape{im_temp_ysize, xsize},
                                      0,
                                      ksize_vert,
                                      bounds_vert,
                                      kk_vert);
        } else {
            imaging_resample_vertical(im_out,
                                      Shape{ysize, xsize},
                                      im_in,
                                      Shape{im_in_ysize, im_in_xsize},
                                      0,
                                      ksize_vert,
                                      bounds_vert,
                                      kk_vert);
        }
    }
    /* none of the previous steps are performed, copying */
    if (!need_horizontal && !need_vertical) {
        std::copy(im_in, im_in + (im_in_xsize * im_in_ysize), im_out);
    } else if (need_horizontal && !need_vertical) {
        std::copy(im_temp.begin(), im_temp.end(), im_out);
    }

    return;
}

}  // namespace interpolate_pil
}  // namespace reference
}  // namespace ov
