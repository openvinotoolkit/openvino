/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifdef __INTEL_COMPILER
#include <immintrin.h>
#endif

#include "mkldnn_types.h"

#include "c_types_map.hpp"
#include "mkldnn_thread.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "jit_avx512_common_convolution_winograd.hpp"

#ifndef _MSC_VER
#define pragma_unroll _Pragma("unroll")
#else
#define pragma_unroll
#endif

namespace mkldnn {
namespace impl {
namespace cpu {

namespace {

unsigned int LLC_cache_size = get_cache_size(3, false);

void inline load_ps(float *dest, const float *src_mem) {
#ifdef __INTEL_COMPILER
    __m512 *Iv512 = (__m512 *)dest;
    Iv512[0] = _mm512_load_ps(src_mem);
#else
    PRAGMA_OMP_SIMD()
    for (int v = 0; v < simd_w; v++) dest[v] = src_mem[v];
#endif
}

void inline store_output(float *dest, const float *data, bool streamout) {
#ifdef __INTEL_COMPILER
    if (streamout)
        _mm512_stream_ps(dest, *((__m512 *)data));
    else
        _mm512_store_ps(dest, *((__m512 *)data));
#else
    PRAGMA_OMP_SIMD()
    for (int v = 0; v < simd_w; v++)
        dest[v] = data[v];
#endif
}

void inline accum_output(
        float *dest, float *data, bool streamout, bool with_relu_postsum) {
#ifdef __INTEL_COMPILER
    __m512 _data = _mm512_loadu_ps(data);
    __m512 _dest = _mm512_loadu_ps(dest);
    _data = _mm512_add_ps(_data, _dest);
    if (with_relu_postsum)
        _data = _mm512_max_ps(_data, _mm512_setzero_ps());
    if (streamout)
        _mm512_stream_ps(dest, _data);
    else
        _mm512_store_ps(dest, _data);
#else
    PRAGMA_OMP_SIMD()
    for (int v = 0; v < simd_w; v++)
        data[v] += dest[v];

    if (with_relu_postsum) {
        PRAGMA_OMP_SIMD()
        for (int v = 0; v < simd_w; v++)
            if (data[v] < 0.f)
                data[v] = 0.f;
    }

    PRAGMA_OMP_SIMD()
    for (int v = 0; v < simd_w; v++)
        dest[v] = data[v];
#endif
}
}

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;

void trans_W_4x4_3x3(float Fw_[6][6][16][16], float F[3][3][16][16]) {
    float Fw[6][16];
    float T[6][3][16];
    float t0[16];
    float t1[16];
    float t2[16];

    for (int j = 0; j < 16; j++) {
#pragma unroll
        for (int i = 0; i < 3; i++) {
            PRAGMA_OMP_SIMD()
            for (int k = 0; k < 16; k++) {
                t0[k] = 0.26890756302521f * F[2][i][j][k];
                t1[k] = -t0[k] - 0.688403361344538f * F[0][i][j][k];
                t2[k] = t0[k] + 0.119514472455649f * F[0][i][j][k];

                T[0][i][k] = 1.13777777777778f * F[0][i][j][k];
                T[1][i][k] = t1[k] - 0.430252100840336f * F[1][i][j][k];
                T[2][i][k] = t1[k] + 0.430252100840336f * F[1][i][j][k];
                T[3][i][k] = t2[k] + 0.179271708683473f * F[1][i][j][k];
                T[4][i][k] = t2[k] - 0.179271708683473f * F[1][i][j][k];
                T[5][i][k] = F[2][i][j][k];
            }
        }
#pragma unroll
        for (int i = 0; i < 6; i++) {
            PRAGMA_OMP_SIMD()
            for (int k = 0; k < 16; k++) {
                t0[k] = 0.26890756302521f * T[i][2][k];
                t1[k] = -t0[k] - 0.688403361344538f * T[i][0][k];
                t2[k] = t0[k] + 0.119514472455649f * T[i][0][k];

                Fw[0][k] = 1.13777777777778f * T[i][0][k];
                Fw[1][k] = t1[k] - 0.430252100840336f * T[i][1][k];
                Fw[2][k] = t1[k] + 0.430252100840336f * T[i][1][k];
                Fw[3][k] = t2[k] + 0.179271708683473f * T[i][1][k];
                Fw[4][k] = t2[k] - 0.179271708683473f * T[i][1][k];
                Fw[5][k] = T[i][2][k];
#pragma unroll
                for (int l = 0; l < 6; l++) {
                    Fw_[i][l][j][k] = Fw[l][k];
                }
            }
        }
    }
}

void trans_O_4x4_3x3(float Mw[6][6][16], float O[4][4][16]) {
    float T[4][6][16];
    float t0[16];
    float t1[16];
    float t2[16];
    float t3[16];

#pragma unroll
    for (int i = 0; i < 6; i++) {
        PRAGMA_OMP_SIMD()
        for (int v = 0; v < 16; v++) {
            t0[v] = Mw[1][i][v] + Mw[2][i][v];
            t1[v] = Mw[3][i][v] + Mw[4][i][v];
            t2[v] = Mw[1][i][v] - Mw[2][i][v];
            t3[v] = Mw[3][i][v] - Mw[4][i][v];

            T[0][i][v] = t0[v] + t1[v] + Mw[0][i][v];
            T[1][i][v] = t2[v] * 0.625f + t3[v] * 1.5f;
            T[2][i][v] = t0[v] * 0.390625f + t1[v] * 2.25f;
            T[3][i][v] = t2[v] * 0.244140625f + t3[v] * 3.375f + Mw[5][i][v];
        }
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        PRAGMA_OMP_SIMD()
        for (int v = 0; v < 16; v++) {
            t0[v] = T[i][1][v] + T[i][2][v];
            t1[v] = T[i][3][v] + T[i][4][v];
            t2[v] = T[i][1][v] - T[i][2][v];
            t3[v] = T[i][3][v] - T[i][4][v];

            O[i][0][v] = t0[v] + t1[v] + T[i][0][v];
            O[i][1][v] = t2[v] * 0.625f + t3[v] * 1.5f;
            O[i][2][v] = t0[v] * 0.390625f + t1[v] * 2.25f;
            O[i][3][v] = t2[v] * 0.244140625f + t3[v] * 3.375f + T[i][5][v];
        }
    }
}


void trans_W_3x3_4x4(float Fw[6][6][16], float F[4][6][16])
{
    const float rcp3 = 1.0f / 3.0f;
    const float rcp4 = 1.0f / 4.0f;
    const float rcp6 = 1.0f / 6.0f;
    const float rcp12 = 1.0f / 12.0f;
    const float rcp24 = 1.0f / 24.0f;
    float t0[16];
    float t1[16];
    float t2[16];
    float t3[16];
    float t4[16];
    float T[6][4][16];

pragma_unroll
    for (int i = 0; i < 4; i++) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < 16; j++) {
            t0[j] = F[2][i][j] * rcp6;
            t1[j] = F[0][i][j] * -rcp6 - t0[j];
            t2[j] = F[0][i][j] * rcp24 + t0[j];
            t3[j] = (F[1][i][j] + F[3][i][j]) * rcp6;
            t4[j] = F[1][i][j] * rcp12 + F[3][i][j] * rcp3;

            T[0][i][j] = F[0][i][j] * rcp4;
            T[1][i][j] = t1[j] - t3[j];
            T[2][i][j] = t1[j] + t3[j];
            T[3][i][j] = t2[j] + t4[j];
            T[4][i][j] = t2[j] - t4[j];
            T[5][i][j] = F[3][i][j];
        }
    }
pragma_unroll
    for (int i = 0; i < 6; i++) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < 16; j++) {
            t0[j] = T[i][2][j] * rcp6;
            t1[j] = T[i][0][j] * -rcp6 - t0[j];
            t2[j] = T[i][0][j] * rcp24 + t0[j];
            t3[j] = (T[i][1][j] + T[i][3][j]) * rcp6;
            t4[j] = T[i][1][j] * rcp12 + T[i][3][j] * rcp3;

            Fw[i][0][j] = T[i][0][j] * rcp4;
            Fw[i][1][j] = t1[j] - t3[j];
            Fw[i][2][j] = t1[j] + t3[j];
            Fw[i][3][j] = t2[j] + t4[j];
            Fw[i][4][j] = t2[j] - t4[j];
            Fw[i][5][j] = T[i][3][j];
        }
    }
}

void trans_O_3x3_4x4(float Mw[6][6][16][16], float M[3][3][16][16])
{
    float T[4][6][16];
    float M_[3][16];
    float t0[16];
    float t1[16];
    float t2[16];

    for (int j = 0; j < 16; j++) {
pragma_unroll
        for (int i = 0; i < 6; i++) {
            PRAGMA_OMP_SIMD()
            for (int l = 0; l < 16; l++) {
                t0[l] = Mw[1][i][j][l] + Mw[2][i][j][l];
                t1[l] = Mw[3][i][j][l] + Mw[4][i][j][l];
                t2[l] = t1[l] * 4.0f + Mw[5][i][j][l];

                T[0][i][l] = Mw[0][i][j][l] + t0[l] + t1[l];
                T[1][i][l] = (Mw[1][i][j][l] - Mw[2][i][j][l]) +
                             2.0f * (Mw[3][i][j][l] - Mw[4][i][j][l]);
                T[2][i][l] = t0[l] + t2[l];
            }
        }
pragma_unroll
        for (int i = 0; i < 3; i++) {
            PRAGMA_OMP_SIMD()
            for (int l = 0; l < 16; l++) {
                t0[l] = T[i][1][l] + T[i][2][l];
                t1[l] = T[i][3][l] + T[i][4][l];
                t2[l] = t1[l] * 4.0f + T[i][5][l];

                M_[0][l] = T[i][0][l] + t0[l] + t1[l];
                M_[1][l] = (T[i][1][l] - T[i][2][l]) +
                           2.0f * (T[i][3][l] - T[i][4][l]);
                M_[2][l] = t0[l] + t2[l];

                for (int k = 0; k < 3; k++) {
                    M[i][k][j][l] = M_[k][l];
                }
            }
        }
    }
}

void trans_I_4x4_3x3(float Iw[6][6][16], float I[6][6][16])
{
    float T[6][6][16];
    float t0[16];
    float t1[16];
    float t2[16];
    float t3[16];
    float t4[16];
    float t5[16];

pragma_unroll
    for (int i = 0; i < 6; i++) {
        PRAGMA_OMP_SIMD()
        for (int v = 0; v < 16; v++) {
            t0[v] = I[2][i][v] * -2.25f + I[4][i][v];
            t1[v] = I[1][i][v] * -2.25f + I[3][i][v];
            t2[v] = I[2][i][v] * -0.390625f + I[4][i][v];
            t3[v] = I[1][i][v] * -0.390625f + I[3][i][v];
            t4[v] = I[0][i][v] * 0.87890625f + I[4][i][v];
            t5[v] = I[1][i][v] * 0.87890625f + I[5][i][v];

            T[0][i][v] = I[2][i][v] * -2.640625f + t4[v];
            T[1][i][v] = t1[v] * 0.625f + t0[v];
            T[2][i][v] = t1[v] * -0.625f + t0[v];
            T[3][i][v] = t3[v] * 1.5f + t2[v];
            T[4][i][v] = t3[v] * -1.5f + t2[v];
            T[5][i][v] = I[3][i][v] * -2.640625f + t5[v];
        }
    }

pragma_unroll
    for (int i = 0; i < 6; i++) {
        PRAGMA_OMP_SIMD()
        for (int v = 0; v < 16; v++) {
            t0[v] = T[i][2][v] * -2.25f + T[i][4][v];
            t1[v] = T[i][1][v] * -2.25f + T[i][3][v];
            t2[v] = T[i][2][v] * -0.390625f + T[i][4][v];
            t3[v] = T[i][1][v] * -0.390625f + T[i][3][v];
            t4[v] = T[i][0][v] * 0.87890625f + T[i][4][v];
            t5[v] = T[i][1][v] * 0.87890625f + T[i][5][v];

            Iw[i][0][v] = T[i][2][v] * -2.640625f + t4[v];
            Iw[i][1][v] = t1[v] * 0.625f + t0[v];
            Iw[i][2][v] = t1[v] * -0.625f + t0[v];
            Iw[i][3][v] = t3[v] * 1.5f + t2[v];
            Iw[i][4][v] = t3[v] * -1.5f + t2[v];
            Iw[i][5][v] = T[i][3][v] * -2.640625f + t5[v];
        }
    }
}

void trans_W_3x3_4x4_wu(float Fw[6][6][16], float F[4][6][16])
{
    float T[6][4][16];
    float t0[16];
    float t1[16];
    float t2[16];
    float t3[16];
    float t4[16];

pragma_unroll
    for (int i = 0; i < 4; i++) {
        PRAGMA_OMP_SIMD()
        for (int v = 0; v < 16; v++) {
            t0[v] = F[2][i][v] * 0.26890756302521f;
            t1[v] = F[0][i][v] * -0.688403361344538f - t0[v];
            t2[v] = F[0][i][v] * 0.119514472455649f + t0[v];
            t3[v] = F[1][i][v] * 0.430252100840336f +
                    F[3][i][v] * 0.168067226890756f;
            t4[v] = F[1][i][v] * 0.179271708683473f +
                    F[3][i][v] * 0.403361344537815f;

            T[0][i][v] = F[0][i][v] * 1.13777777777778f;
            T[1][i][v] = t1[v] - t3[v];
            T[2][i][v] = t1[v] + t3[v];
            T[3][i][v] = t2[v] + t4[v];
            T[4][i][v] = t2[v] - t4[v];
            T[5][i][v] = F[3][i][v];
        }
    }
pragma_unroll
    for (int i = 0; i < 6; i++) {
        for (int v = 0; v < 16; v++) {
            t0[v] = T[i][2][v] * 0.26890756302521f;
            t1[v] = T[i][0][v] * -0.688403361344538f - t0[v];
            t2[v] = T[i][0][v] * 0.119514472455649f + t0[v];
            t3[v] = T[i][1][v] * 0.430252100840336f +
                    T[i][3][v] * 0.168067226890756f;
            t4[v] = T[i][1][v] * 0.179271708683473f +
                    T[i][3][v] * 0.403361344537815f;

            Fw[i][0][v] = T[i][0][v] * 1.13777777777778f;
            Fw[i][1][v] = t1[v] - t3[v];
            Fw[i][2][v] = t1[v] + t3[v];
            Fw[i][3][v] = t2[v] + t4[v];
            Fw[i][4][v] = t2[v] - t4[v];
            Fw[i][5][v] = T[i][3][v];
        }
    }
}

void trans_O_3x3_4x4_wu(float Mw[6][6][16][16], float M[3][3][16][16])
{
    float T[3][6][16];
    float t0[16];
    float t1[16];
    float t2[16];
    float M_[3][16];

    for (int j = 0; j < 16; j++) {
pragma_unroll
        for (int i = 0; i < 6; i++) {
            PRAGMA_OMP_SIMD()
            for (int v = 0; v < 16; v++) {
                t0[v] = Mw[1][i][j][v] + Mw[2][i][j][v];
                t1[v] = Mw[3][i][j][v] + Mw[4][i][j][v];
                t2[v] = t1[v] * 2.25f + Mw[5][i][j][v];

                T[0][i][v] = Mw[0][i][j][v] + t0[v] + t1[v];
                T[1][i][v] = 0.625f * (Mw[1][i][j][v] - Mw[2][i][j][v]) +
                             1.5f * (Mw[3][i][j][v] - Mw[4][i][j][v]);
                T[2][i][v] = t0[v] * 0.390625f + t2[v];
            }
        }
pragma_unroll
        for (int i = 0; i < 3; i++) {
            PRAGMA_OMP_SIMD()
            for (int v = 0; v < 16; v++) {
                t0[v] = T[i][1][v] + T[i][2][v];
                t1[v] = T[i][3][v] + T[i][4][v];
                t2[v] = t1[v] * 2.25f + T[i][5][v];

                M_[0][v] = T[i][0][v] + t0[v] + t1[v];
                M_[1][v] = 0.625f * (T[i][1][v] - T[i][2][v]) +
                           1.5f * (T[i][3][v] - T[i][4][v]);
                M_[2][v] = t0[v] * 0.390625f + t2[v];
            }

pragma_unroll
            for (int k = 0; k < 3; k++) {
                PRAGMA_OMP_SIMD()
                for (int v = 0; v < 16; v++) {
                    M[i][k][j][v] = M_[k][v];
                }
            }
        }
    }
}

template <bool is_fwd>
void input_transform_data(int image, const jit_conv_winograd_conf_t &jcp,
        float *inp, float *tinp, bool streamout = true)
{
    const int inpw = is_fwd ? jcp.iw : jcp.ow;
    const int inph = is_fwd ? jcp.ih : jcp.oh;
    const int l_pad = is_fwd ? jcp.l_pad : jcp.iw + jcp.r_pad - jcp.ow;
    const int t_pad = is_fwd ? jcp.t_pad : jcp.ih + jcp.t_pad - jcp.oh;
    const int wp_max = inpw + l_pad;
    const int hp_max = inph + t_pad;
    float Iw[alpha][alpha][simd_w];
    float I[alpha][alpha][simd_w];

    array_offset_calculator<float, 5> input(inp,
            jcp.mb, jcp.dimK/simd_w, inph, inpw,
            simd_w);
    array_offset_calculator<float, 8> output(tinp,
            jcp.dimN_nb_block, alpha, alpha,
            jcp.dimN_block, jcp.dimK_nb_block, jcp.dimK_block,
            jcp.dimN_reg_block, jcp.dimK_reg_block);

    int tile_base_index = image * jcp.itiles * jcp.jtiles;
    int tile_block_ur = tile_base_index % jcp.tile_block_ur;
    int nb_tile_block_ur =
        (tile_base_index / jcp.tile_block_ur) % jcp.nb_tile_block_ur;
    int tile_block =
        (tile_base_index / jcp.tile_block_ur) / jcp.nb_tile_block_ur;

    for (int tj = 0; tj < jcp.jtiles; tj++) {
        for (int ti = 0; ti < jcp.itiles; ti++) {
            for (int j = 0; j < alpha; j++) {
                int ydim = tj * tile_size + j;
                if ((t_pad <= ydim) && (ydim < hp_max)) {
                    float *pinp_j = inp + (ydim - t_pad) * inpw * 16 ;
                    for (int i = 0; i < alpha; i++) {
                        int xdim = ti * tile_size + i;
                        if ((l_pad <= xdim) && (xdim < wp_max)) {
                            float *pinp_i = pinp_j + (xdim - l_pad) * 16;
                            load_ps(I[j][i], pinp_i);
                        } else {
                            PRAGMA_OMP_SIMD()
                            for (int v = 0; v < simd_w; v++) {
                                I[j][i][v] = 0.0f;
                            }
                        }
                    }
                } else {
                    for (int i = 0; i < alpha; i++) {
                        PRAGMA_OMP_SIMD()
                        for (int v = 0; v < simd_w; v++) {
                            I[j][i][v] = 0.0f;
                        }
                    }
                }
            }

            trans_I_4x4_3x3(Iw, I);

            for (int j = 0; j < alpha; j++) {
                for (int i = 0; i < alpha; i++) {
                    store_output(&(output(tile_block, j, i,
                                    nb_tile_block_ur, 0, 0,
                                    tile_block_ur, 0)),
                                 Iw[j][i], streamout);
                }
            }
            tile_block_ur++;
            if (tile_block_ur >= jcp.tile_block_ur) {
                tile_block_ur = 0;
                nb_tile_block_ur++;
            }
            if (nb_tile_block_ur >= jcp.nb_tile_block_ur) {
                nb_tile_block_ur = 0;
                tile_block++;
            }
        }
    }
}

template <bool is_fwd>
void input_transform_tileblock_data(int tile_block,
        const jit_conv_winograd_conf_t &jcp,
        float *inp, float *tinp)
{
    const int inph = is_fwd ? jcp.ih : jcp.oh;
    const int inpw = is_fwd ? jcp.iw : jcp.ow;
    const int t_pad = is_fwd ? jcp.t_pad : jcp.ih + jcp.t_pad - jcp.oh;
    const int l_pad = is_fwd ? jcp.l_pad : jcp.iw + jcp.r_pad - jcp.ow;
    const int wp_max = inpw + l_pad;
    const int hp_max = inph + t_pad;
    float Iw[alpha][alpha][simd_w];
    float I[alpha][alpha][simd_w];

    array_offset_calculator<float, 5> input(inp,
            jcp.mb, jcp.dimK/simd_w, inph, inpw, simd_w);
    array_offset_calculator<float, 7> output(tinp,
            alpha, alpha,
            jcp.dimN_block, jcp.dimK_nb_block, jcp.dimK_block,
            jcp.dimN_reg_block, jcp.dimK_reg_block);

    int tile_index = tile_block * jcp.nb_tile_block_ur * jcp.tile_block_ur;

    for (int nb_tile_block_ur = 0;
            nb_tile_block_ur < jcp.nb_tile_block_ur;
            nb_tile_block_ur++) {
        for (int tile_block_ur = 0; tile_block_ur < jcp.tile_block_ur;
                tile_block_ur++) {

            int img = tile_index / (jcp.jtiles * jcp.itiles);
            int ti = tile_index % jcp.itiles;
            int tj = (tile_index / jcp.itiles) % jcp.jtiles;
            float *pinp_b = &(input(img, 0, 0, 0, 0));

            for (int j = 0; j < alpha; j++) {
                int ydim = tj * tile_size + j;
                if ((t_pad <= ydim) && (ydim < hp_max)) {
                    float *pinp_j = pinp_b + (ydim - t_pad) * inpw * simd_w;
                    for (int i = 0; i < alpha; i++) {
                        int xdim = ti * tile_size + i;
                        if ((l_pad <= xdim) && (xdim < wp_max)) {
                            float *pinp_i = pinp_j + (xdim - l_pad) * simd_w;
                            load_ps(I[j][i], pinp_i);
                        } else {
                            PRAGMA_OMP_SIMD()
                            for (int v = 0; v < simd_w; v++) {
                                I[j][i][v] = 0.0f;
                            }
                        }
                    }
                } else {
                    for (int i = 0; i < alpha; i++) {
                        PRAGMA_OMP_SIMD()
                        for (int v = 0; v < simd_w; v++) {
                            I[j][i][v] = 0.0f;
                        }
                    }
                }
            }

            trans_I_4x4_3x3(Iw, I);
            for (int j = 0; j < alpha; j++) {
                for (int i = 0; i < alpha; i++) {
                    store_output(&(output(j, i,
                                    nb_tile_block_ur, 0, 0,
                                    tile_block_ur, 0)),
                                 Iw[j][i], false);
                }
            }
            tile_index++;
        }
    }
}

template <bool is_fwd>
void weight_transform_data(const jit_conv_winograd_conf_t &jcp,
        float *wp, float *twp)
{
    const int kh = 3;
    const int kw = 3;
    array_offset_calculator<float, 6> input(wp,
            jcp.oc/jcp.oc_simd_block,
            jcp.ic/jcp.ic_simd_block,
            jcp.kh, jcp.kw,
            simd_w, simd_w);
    array_offset_calculator<float, 8> output(twp,
            jcp.dimM_nb_block,
            alpha, alpha,
            jcp.dimK_nb_block,
            jcp.dimM_block, jcp.dimK_block,
            simd_w, simd_w);
    float Fw[alpha][alpha][simd_w][simd_w];
    float F[kh][kw][simd_w][simd_w];

    for (int j = 0; j < kh; j++) {
        for (int i = 0; i < kw; i++) {
            for (int v1 = 0; v1 < simd_w; v1++) {
                float *base_inp = is_fwd
                                ? &(input(0, 0, j, i, v1, 0))
                                : &(input(0, 0, 2 - j, 2 - i, v1, 0));
                PRAGMA_OMP_SIMD()
                for (int v2 = 0; v2 < simd_w; v2++) {
                    if (is_fwd)
                        F[j][i][v1][v2] = *(base_inp + v2);
                    else
                        F[j][i][v2][v1] = *(base_inp + v2);
                }
            }
        }
    }

    trans_W_4x4_3x3(Fw, F);

    for (int j = 0; j < alpha; j++) {
        for (int i = 0; i < alpha; i++) {
            for (int v1 = 0; v1 < simd_w; v1++) {
                PRAGMA_OMP_SIMD()
                for (int v2 = 0; v2 < simd_w; v2++) {
                    output(0, j, i, 0, 0, 0, v1, v2) = Fw[j][i][v1][v2];
                }
            }
        }
    }
}

template <bool is_fwd, bool with_bias, bool with_relu_presum, bool with_sum>
void output_transform_data(int image, const jit_conv_winograd_conf_t &jcp,
        const post_ops_t &p_ops, float *toutp, float *pout_b, float *bias,
        bool streamout = true) {
    float Ow[alpha][alpha][simd_w];
    float O[tile_size][tile_size][simd_w];
    int outw = is_fwd ? jcp.ow : jcp.iw;
    int outh = is_fwd ? jcp.oh : jcp.ih;

    /* Prepare for PostOps */
    bool with_relu_postsum = p_ops.find(primitive_kind::eltwise, 1) != -1;

    array_offset_calculator<float, 8> input(toutp,
            jcp.dimN_nb_block, jcp.dimM_nb_block,
            alpha, alpha,
            jcp.dimN_block, jcp.dimM_block,
            jcp.dimN_reg_block, jcp.dimM_simd_block);

    int tile_base_index = image * jcp.itiles * jcp.jtiles;
    int tile_block_ur = tile_base_index % jcp.tile_block_ur;
    int nb_tile_block_ur =
        (tile_base_index / jcp.tile_block_ur) % jcp.nb_tile_block_ur;
    int tile_block =
        (tile_base_index / jcp.tile_block_ur) / jcp.nb_tile_block_ur;

    for (int tj = 0; tj < jcp.jtiles; tj++) {
        for (int ti = 0; ti < jcp.itiles; ti++) {
            for (int j = 0; j < alpha; j++) {
                for (int i = 0; i < alpha; i++) {
                    PRAGMA_OMP_SIMD()
                    for (int v = 0; v < simd_w; v++) {
                        Ow[j][i][v] = input(tile_block, 0,
                                j, i,
                                nb_tile_block_ur, 0,
                                tile_block_ur, v);
                    }
                }
            }

            trans_O_4x4_3x3(Ow, O);

            for (int j = 0; j < tile_size; j++) {
                int ydim = tj * tile_size + j;
                if (ydim < outh) {
                    float *pout_j = pout_b + ydim * outw * simd_w;
                    for (int i = 0; i < tile_size; i++) {
                        int xdim = ti * tile_size + i;
                        if (xdim < outw) {
                            float *pout_i = pout_j + xdim * simd_w;
                            if (is_fwd) {
                                PRAGMA_OMP_SIMD()
                                for (int v = 0; v < simd_w; v++) {
                                    O[j][i][v] += with_bias ? bias[v] : 0.f;
                                    O[j][i][v] = true
                                        && with_relu_presum && O[j][i][v] < 0.f
                                                ? O[j][i][v]
                                                * jcp.eltwise_alpha
                                                : O[j][i][v];
                                }
                            }
                            if (with_sum)
                                accum_output(pout_i, O[j][i], streamout,
                                        with_relu_postsum);
                            else
                                store_output(pout_i, O[j][i], streamout);
                        }
                    }
                }
            }
            tile_block_ur++;
            if (tile_block_ur >= jcp.tile_block_ur) {
                tile_block_ur = 0;
                nb_tile_block_ur++;
            }
            if (nb_tile_block_ur >= jcp.nb_tile_block_ur) {
                nb_tile_block_ur = 0;
                tile_block++;
            }
        }
    }
}

template <bool is_fwd, bool with_bias, bool with_relu_presum, bool with_sum>
void output_transform_tileblock_data(int tile_block,
        const jit_conv_winograd_conf_t &jcp, const post_ops_t &p_ops,
        float *toutp, float *outp, float *bias, bool streamout) {
    float Ow[alpha][alpha][simd_w];
    float O[tile_size][tile_size][simd_w];
    int outw = is_fwd ? jcp.ow : jcp.iw;
    int outh = is_fwd ? jcp.oh : jcp.ih;

    /* Prepare for PostOps */
    bool with_relu_postsum = p_ops.find(primitive_kind::eltwise, 1) != -1;

    array_offset_calculator<float, 6> input(toutp,
            alpha, alpha,
            jcp.dimN_block, jcp.dimM_block,
            jcp.dimN_reg_block, jcp.dimM_simd_block);
    array_offset_calculator<float, 5> output(outp,
            jcp.mb, jcp.dimM/jcp.dimM_simd_block, outh, outw,
            jcp.dimM_simd_block);

    int tile_index = tile_block * jcp.nb_tile_block_ur * jcp.tile_block_ur;

    for (int nb_tile_block_ur = 0;
            nb_tile_block_ur < jcp.nb_tile_block_ur;
            nb_tile_block_ur++) {

        for (int tile_block_ur = 0; tile_block_ur < jcp.tile_block_ur;
                tile_block_ur++) {
            int img = tile_index / (jcp.jtiles * jcp.itiles);
            int ti = tile_index % jcp.itiles;
            int tj = (tile_index / jcp.itiles) % jcp.jtiles;

            for (int j = 0; j < alpha; j++) {
                for (int i = 0; i < alpha; i++) {
                    float *pinp_tile = &(input(j, i, nb_tile_block_ur, 0,
                                tile_block_ur, 0));
                    load_ps(Ow[j][i], pinp_tile);
                }
            }

            trans_O_4x4_3x3(Ow, O);

            float *pout_b = &(output(img, 0, 0, 0, 0));
            for (int j = 0; j < tile_size; j++) {
                int ydim = tj * tile_size + j;
                if (ydim < outh) {
                    float *pout_j = pout_b + ydim * outw * simd_w;
                    for (int i = 0; i < tile_size; i++) {
                        int xdim = ti * tile_size + i;
                        if (xdim < outw) {
                            float *pout_i = pout_j + xdim * simd_w;
                            if (is_fwd) {
                                PRAGMA_OMP_SIMD()
                                for (int v = 0; v < simd_w; v++) {
                                    O[j][i][v] += with_bias ? bias[v] : 0.f;
                                    O[j][i][v] = true
                                        && with_relu_presum && O[j][i][v] < 0.f
                                                ? O[j][i][v]
                                                * jcp.eltwise_alpha
                                                : O[j][i][v];

                                }
                            }
                            if (with_sum)
                                accum_output(pout_i, O[j][i], streamout,
                                        with_relu_postsum);
                            else
                                store_output(pout_i, O[j][i], streamout);
                        }
                    }
                }
            }
            tile_index++;
        }
    }
}

template <bool ver_4fma>
void diff_src_transform_bwd_weights(int image, jit_conv_winograd_conf_t conv,
        float *inp, float *tinp, float *Iw_temp,
        void (*transpose_4fma_ker)(float *, float *))
{

    const int ifwp = conv.iw + conv.l_pad;
    const int ifhp = conv.ih + conv.t_pad;
    float I[alpha][alpha][simd_w];
    float Iw[alpha][alpha][simd_w];

    array_offset_calculator<float, 4> Iw_trans_temp(Iw_temp,
            alpha, alpha, conv.tile_4fma, simd_w);
    array_offset_calculator<float, 5> input(inp,
            conv.mb, conv.ic/simd_w, conv.ih, conv.iw, simd_w);
    array_offset_calculator<float, 8> output(tinp,
            conv.nb_ic, alpha, alpha,
            conv.tile_block, conv.ic_block,
            conv.nb_tile_block_ur, conv.tile_block_ur,
            conv.ic_simd_block * conv.tile_4fma);

    int tile_base_index =
        image * (conv.itiles * conv.jtiles + conv.tile_4fma_padding);
    int tile_4fma = 0;
    int tile_block_ur = (tile_base_index / conv.tile_4fma) % conv.tile_block_ur;
    int nb_tile_block_ur =
        (tile_base_index / conv.tile_4fma / conv.tile_block_ur)
        % conv.nb_tile_block_ur;
    int tile_block = (tile_base_index / conv.tile_4fma / conv.tile_block_ur)
            / conv.nb_tile_block_ur;

    for (int tj = 0; tj < conv.jtiles; tj++) {
        for (int ti = 0; ti < conv.itiles; ti++) {
            for (int j = 0; j < alpha; j++) {
                int ydim = tj * tile_size + j;
                if ((conv.t_pad <= ydim) && ydim < ifhp) {
                    for (int i = 0; i < alpha; i++) {
                        int xdim = ti * tile_size + i;
                        if ((conv.l_pad <= xdim) && xdim < ifwp) {
                            PRAGMA_OMP_SIMD()
                            for (int v = 0; v < simd_w; v++) {
                                I[j][i][v] = input(0, 0,
                                        ydim - conv.t_pad,
                                        xdim - conv.l_pad, v);
                            }
                        } else {
                            PRAGMA_OMP_SIMD()
                            for (int v = 0; v < simd_w; v++) {
                                I[j][i][v] = 0.0f;
                            }
                        }
                    }
                } else {
                    for (int i = 0; i < alpha; i++) {
                        PRAGMA_OMP_SIMD()
                        for (int v = 0; v < simd_w; v++) {
                            I[j][i][v] = 0.0f;
                        }
                    }
                }
            }
            trans_I_4x4_3x3(Iw, I);

            if (ver_4fma) {
                for (int j = 0; j < alpha; j++) {
                    for (int i = 0; i < alpha; i++) {
                        float *Iw_temp_base = &(Iw_trans_temp(j, i,
                                                        tile_4fma, 0));
                        PRAGMA_OMP_SIMD()
                        for (int v = 0; v < simd_w; v++) {
                            Iw_temp_base[v] = Iw[j][i][v];
                        }
                    }
                }
                tile_4fma++;
                if (tile_4fma == conv.tile_4fma) {
                    float *outp = &(output(0, 0, 0,
                                tile_block, 0,
                                nb_tile_block_ur, tile_block_ur, 0));
                    transpose_4fma_ker(outp, (float *)Iw_temp);
                    tile_4fma = 0;
                    tile_block_ur++;
                }
            } else {
                for (int j = 0; j < alpha; j++) {
                    for (int i = 0; i < alpha; i++) {
                        store_output(&(output(0, j, i,
                                        tile_block, 0,
                                        nb_tile_block_ur, tile_block_ur, 0)),
                                     Iw[j][i], true);
                    }
                }
                tile_block_ur++;
            }

            if (tile_block_ur == conv.tile_block_ur) {
                tile_block_ur = 0;
                ++nb_tile_block_ur;
            }
            if (nb_tile_block_ur == conv.nb_tile_block_ur) {
                nb_tile_block_ur = 0;
                tile_block++;
            }
        }
    }

    if (ver_4fma && tile_4fma < conv.tile_4fma && conv.tile_4fma_padding != 0) {

        for (int j = 0; j < alpha; j++) {
            for (int i = 0; i < alpha; i++) {
                for (int tb = tile_4fma; tb < conv.tile_4fma; tb++) {
                    float *Iw_temp_base = &(Iw_trans_temp(j, i, tb, 0));
                    PRAGMA_OMP_SIMD()
                    for (int v = 0; v < simd_w; v++) {
                        Iw_temp_base[v] = 0;
                    }
                }
            }
        }
        float *outp = &(output(0, 0, 0,
                    tile_block, 0,
                    nb_tile_block_ur, tile_block_ur, 0));
        transpose_4fma_ker(outp, (float *)Iw_temp);
    }
}

template <bool with_bias>
void diff_dst_transform_bwd_weights(int image, jit_conv_winograd_conf_t conv,
        float *inp, float *tinp, float *dbias)
{

    const int total_tiles = conv.itiles * conv.jtiles + conv.tile_4fma_padding;
    float I[alpha][alpha][simd_w];
    float Iw[alpha][alpha][simd_w];

    array_offset_calculator<float, 5> input(inp,
            conv.mb, conv.oc/simd_w, conv.oh, conv.ow, conv.oc_simd_block);
    array_offset_calculator<float, 8> output(tinp,
            conv.nb_oc, alpha, alpha,
            conv.tile_block, conv.oc_block,
            conv.nb_tile_block_ur,
            conv.tile_block_ur * conv.tile_4fma, conv.oc_simd_block);

    int tile_base_index = image * total_tiles;
    int tile_block_ur = tile_base_index % (conv.tile_block_ur * conv.tile_4fma);
    int nb_tile_block_ur =
        (tile_base_index / conv.tile_block_ur / conv.tile_4fma)
            % conv.nb_tile_block_ur;
    int tile_block = (tile_base_index / conv.tile_block_ur / conv.tile_4fma)
            / conv.nb_tile_block_ur;

    for (int tj = 0; tj < conv.jtiles; tj++) {
        for (int ti = 0; ti < conv.itiles; ti++) {
            for (int j = 0; j < alpha; j++) {
                int ydim = tj * tile_size + j;
                if (ydim < conv.oh) {
                    for (int i = 0; i < alpha; i++) {
                        int xdim = ti * tile_size + i;
                        if (xdim < conv.ow) {
                            float *input_base = &(input(0, 0, ydim, xdim, 0));

                            PRAGMA_OMP_SIMD()
                            for (int v = 0; v < simd_w; v++) {
                                I[j][i][v] = input_base[v];
                            }
                            if (with_bias && j < tile_size && i < tile_size) {
                                PRAGMA_OMP_SIMD()
                                for (int v = 0; v < simd_w; v++) {
                                    dbias[v] += input_base[v];
                                }
                            }
                        } else {
                            PRAGMA_OMP_SIMD()
                            for (int v = 0; v < simd_w; v++) {
                                I[j][i][v] = 0.0f;
                            }
                        }
                    }
                } else {
                    for (int i = 0; i < alpha; i++) {
                        PRAGMA_OMP_SIMD()
                        for (int v = 0; v < simd_w; v++) {
                            I[j][i][v] = 0.0f;
                        }
                    }
                }
            }

            trans_W_3x3_4x4_wu(Iw, I);

            for (int j = 0; j < alpha; j++) {
                for (int i = 0; i < alpha; i++) {
                    store_output(&(output(0, j, i,
                                    tile_block, 0,
                                    nb_tile_block_ur,
                                    tile_block_ur, 0)),
                                 Iw[j][i], true);
                }
            }
            tile_block_ur++;
            if (tile_block_ur >= conv.tile_block_ur * conv.tile_4fma) {
                tile_block_ur = 0;
                nb_tile_block_ur++;
            }
            if (nb_tile_block_ur >= conv.nb_tile_block_ur) {
                nb_tile_block_ur = 0;
                tile_block++;
            }
        }
    }
}

void diff_weights_transform_bwd_weights(jit_conv_winograd_conf_t conv,
        float *wp, float *twp)
{
    const int kh = 3;
    const int kw = 3;
    float Fw[alpha][alpha][simd_w][simd_w];
    float F[kh][kw][simd_w][simd_w];

    array_offset_calculator<float, 8> input(twp,
            conv.nb_ic, conv.nb_oc,
            alpha, alpha,
            conv.oc_block, conv.ic_block,
            conv.ic_simd_block, conv.oc_simd_block);
    array_offset_calculator<float, 6> output(wp,
            conv.oc/simd_w, conv.ic/simd_w,
            conv.kh, conv.kw,
            conv.ic_simd_block, conv.oc_simd_block);

    for (int j = 0; j < alpha; j++) {
        for (int i = 0; i < alpha; i++) {
            for (int v = 0; v < conv.ic_simd_block; v++) {
                PRAGMA_OMP_SIMD()
                for (int k = 0; k < conv.oc_simd_block; k++) {
                    Fw[j][i][v][k] = input(0, 0, j, i, 0, 0, v, k);
                }
            }
        }
    }

    trans_O_3x3_4x4_wu(Fw, F);

    for (int j = 0; j < kh; j++) {
        for (int i = 0; i < kw; i++) {
            for (int v = 0; v < conv.ic_simd_block; v++) {
                store_output(&(output(0, 0, j, i, v, 0)),
                             F[j][i][v], true);
            }
        }
    }
}

template <bool is_fwd>
void _jit_avx512_common_convolution_winograd_t<is_fwd>::_execute_data_W_S_G_D(
        const int MB, float *inp_ptr, float *out_ptr, float *wei_ptr, float *bias_ptr) {
    const auto &jcp = kernel_->jcp;
    const auto &p_ops = attr_->post_ops_;

    const int inph = is_fwd ? jcp.ih : jcp.oh;
    const int inpw = is_fwd ? jcp.iw : jcp.ow;
    const int outh = is_fwd ? jcp.oh : jcp.ih;
    const int outw = is_fwd ? jcp.ow : jcp.iw;

    /* Note that jcp.with_relu is true for both fused conv+relu primitive
     * and conv primitive with PostOps with relu before sum
     * (PostOps relu after sum is handled later) */
    auto output_transform = jcp.with_bias
            ? (jcp.with_eltwise
                ? (jcp.with_sum
                    ? output_transform_data<is_fwd, true, true, true>
                    : output_transform_data<is_fwd, true, true, false>)
                : (jcp.with_sum
                    ? output_transform_data<is_fwd, true, false, true>
                    : output_transform_data<is_fwd, true, false, false>))
            : (jcp.with_eltwise
                ? (jcp.with_sum
                    ? output_transform_data<is_fwd, false, true, true>
                    : output_transform_data<is_fwd, false, true, false>)
                : (jcp.with_sum
                    ? output_transform_data<is_fwd, false, false, true>
                    : output_transform_data<is_fwd, false, false, false>));

    /* Notation:
       FWD: dimM:oc, dimN:ntiles, dimK:ic,
       BWD: dimM:ic, dimN:ntiles, dimK:oc,
       FWD/BWD: V: src/diff_dst transform, U:weight transform,
                M:dst/diff_src transform  */
    array_offset_calculator<float, 5> input(inp_ptr,
            MB, jcp.dimK/jcp.dimK_reg_block, inph, inpw,
            jcp.dimK_reg_block);
    array_offset_calculator<float, 5> output(out_ptr,
            MB, jcp.dimM/jcp.dimM_simd_block, outh, outw,
            jcp.dimM_simd_block);
    array_offset_calculator<float, 6> weights(wei_ptr,
            jcp.oc/jcp.oc_simd_block, jcp.ic/jcp.ic_simd_block, jcp.kh, jcp.kw,
            jcp.ic_simd_block, jcp.oc_simd_block);
    array_offset_calculator<float, 2> bias(bias_ptr,
            jcp.dimM/jcp.dimM_simd_block, jcp.dimM_simd_block);

    array_offset_calculator<float, 8> M(
            (float *)((is_fwd
                    ? (this->scratchpad_)->M_ptr()
                    : (this->scratchpad_)->V_ptr())),
            jcp.dimN_nb_block, jcp.dimM_nb_block,
            alpha, alpha,
            jcp.dimN_block, jcp.dimM_block,
            jcp.dimN_reg_block, jcp.dimM_simd_block);
    array_offset_calculator<float, 8> U((float *)((this->scratchpad_)->U_ptr()),
            jcp.dimM_nb_block,
            alpha, alpha,
            jcp.dimK_nb_block,
            jcp.dimM_block, jcp.dimK_block,
            jcp.dimK_reg_block, jcp.dimM_simd_block);
    array_offset_calculator<float, 8> V(
            (float *)((is_fwd
                    ? (this->scratchpad_)->V_ptr()
                    : (this->scratchpad_)->M_ptr())),
            jcp.dimN_nb_block, alpha, alpha,
            jcp.dimN_block, jcp.dimK_nb_block,
            jcp.dimK_block, jcp.dimN_reg_block, jcp.dimK_reg_block);

    bool V_streamout = jcp.dimN * jcp.dimK * alpha * alpha * sizeof(float)
        > 2 * LLC_cache_size ? true : false;

    const bool output_is_aligned = ((size_t)out_ptr & (64 - 1)) == 0;

    const bool want_padded_bias = jcp.with_bias
        && jcp.oc_without_padding != jcp.oc;
    float last_slice_bias[simd_w] = {0};
    if (want_padded_bias) {
        for (int oc = 0; oc < jcp.oc_without_padding % jcp.oc_simd_block; ++oc)
            last_slice_bias[oc] = bias(jcp.dimM / jcp.dimM_simd_block - 1, oc);
    }

#pragma omp parallel
    {
        parallel_nd_in_omp(MB, jcp.dimK_nb_block, jcp.dimK_block,
            [&](int img, int K_blk1, int K_blk2) {
            input_transform_data<is_fwd>(img, jcp,
                &(input(img, K_blk1 * jcp.dimK_block + K_blk2, 0, 0, 0)),
                &(V(0, 0, 0, 0, K_blk1, K_blk2, 0, 0)), V_streamout);
        });

        parallel_nd_in_omp(jcp.nb_oc, jcp.nb_ic, jcp.oc_block, jcp.ic_block,
            [&](int ofm1, int ifm1, int ofm2, int ifm2) {
            float *U_base_ptr = is_fwd
                ? &(U(ofm1, 0, 0, ifm1, ofm2, ifm2, 0, 0))
                : &(U(ifm1, 0, 0, ofm1, ifm2, ofm2, 0, 0));
            weight_transform_data<is_fwd>(jcp,
                &(weights(ofm1 * jcp.oc_block + ofm2,
                ifm1 * jcp.ic_block + ifm2, 0, 0, 0, 0)), U_base_ptr);
        });

#pragma omp barrier

        parallel_nd_in_omp(jcp.dimN_nb_block, alpha, alpha, jcp.dimM_nb_block, jcp.dimN_block,
            [&](int N_blk1, int oj, int oi, int M_blk1, int N_blk2) {

            kernel_->gemm_loop_ker_first_iter(
                    (float *)&(M(N_blk1, M_blk1, oj, oi,
                            N_blk2, 0, 0, 0)),
                    (const float *)&(U(M_blk1, oj, oi,
                            0, 0, 0, 0, 0)),
                    (const float *)&(V(N_blk1, oj, oi,
                            N_blk2, 0, 0, 0, 0)));
            for (int K_blk1 = 1; K_blk1 < jcp.dimK_nb_block; K_blk1++) {
                kernel_->gemm_loop_ker(
                        (float *)&(M(N_blk1, M_blk1, oj, oi,
                                N_blk2, 0, 0, 0)),
                        (const float *)&(U(M_blk1, oj, oi,
                                K_blk1, 0, 0, 0, 0)),
                        (const float *)&(V(N_blk1, oj, oi,
                                N_blk2, K_blk1,
                                0, 0, 0)));
            }

        });


#pragma omp barrier

        parallel_nd_in_omp(MB, jcp.dimM_nb_block, jcp.dimM_block,
                    [&](int img, int M_blk1, int M_blk2) {

            const int M_blk = M_blk1 * jcp.dimM_block + M_blk2;

            float *bias_ptr = want_padded_bias
                && M_blk == jcp.dimM / jcp.dimM_simd_block - 1
                ? last_slice_bias : &bias(M_blk, 0);

            output_transform(img, jcp, p_ops,
                    &(M(0, M_blk1, 0, 0, 0, M_blk2, 0, 0)),
                    &(output(img, M_blk, 0, 0, 0)),
                    bias_ptr, output_is_aligned);

       });
    }
}

template void
_jit_avx512_common_convolution_winograd_t<true>::_execute_data_W_S_G_D(
        const int, float *, float *, float *, float *);
template void
_jit_avx512_common_convolution_winograd_t<false>::_execute_data_W_S_G_D(
        const int, float *, float *, float *, float *);

template <bool is_fwd>
void _jit_avx512_common_convolution_winograd_t<is_fwd>::_execute_data_W_SGD(
        const int MB, float *inp_ptr, float *out_ptr, float *wei_ptr, float *bias_ptr) {
    const auto &jcp = kernel_->jcp;
    const auto &p_ops = attr_->post_ops_;

    const int inph = is_fwd ? jcp.ih : jcp.oh;
    const int inpw = is_fwd ? jcp.iw : jcp.ow;
    const int outh = is_fwd ? jcp.oh : jcp.ih;
    const int outw = is_fwd ? jcp.ow : jcp.iw;

    /* Note that jcp.with_relu is true for both fused conv+relu primitive
     * and conv primitive with PostOps with relu before sum
     * (PostOps relu after sum is handled later) */
    auto output_transform_tileblock = jcp.with_bias
            ? (jcp.with_eltwise
                ? (jcp.with_sum
                    ? output_transform_tileblock_data<is_fwd, true, true, true>
                    : output_transform_tileblock_data<is_fwd, true, true, false>)
                : (jcp.with_sum
                    ? output_transform_tileblock_data<is_fwd, true, false, true>
                    : output_transform_tileblock_data<is_fwd, true, false, false>))
            : (jcp.with_eltwise
                ? (jcp.with_sum
                    ? output_transform_tileblock_data<is_fwd, false, true, true>
                    : output_transform_tileblock_data<is_fwd, false, true, false>)
                : (jcp.with_sum
                    ? output_transform_tileblock_data<is_fwd, false, false, true>
                    : output_transform_tileblock_data<is_fwd, false, false, false>));

    array_offset_calculator<float, 5> input(inp_ptr,
            MB, jcp.dimK/jcp.dimK_reg_block, inph, inpw, jcp.dimK_reg_block);
    array_offset_calculator<float, 5> output(out_ptr,
            MB, jcp.dimM/jcp.dimM_simd_block, outh, outw, jcp.dimM_simd_block);
    array_offset_calculator<float, 6> weights(wei_ptr,
            jcp.oc/jcp.oc_simd_block, jcp.ic/jcp.ic_simd_block, jcp.kh, jcp.kw,
            jcp.ic_simd_block, jcp.oc_simd_block);
    array_offset_calculator<float, 2> bias(bias_ptr,
            jcp.oc/jcp.oc_simd_block, jcp.oc_simd_block);

    array_offset_calculator<float, 8> U((float *)((this->scratchpad_)->U_ptr()),
            jcp.dimM_nb_block,
            alpha, alpha,
            jcp.dimK_nb_block,
            jcp.dimM_block, jcp.dimK_block,
            jcp.dimK_reg_block, jcp.dimM_simd_block);

    array_offset_calculator<float, 8> M(
            (float *)((is_fwd
                    ? (this->scratchpad_)->M_ptr()
                    : (this->scratchpad_)->V_ptr())),
            0, jcp.dimM_nb_block, alpha, alpha,
            jcp.dimN_block, jcp.dimM_block,
            jcp.dimN_reg_block, jcp.dimM_simd_block);

    array_offset_calculator<float, 8> V(
            (float *)((is_fwd
                    ? (this->scratchpad_)->V_ptr()
                    : (this->scratchpad_)->M_ptr())),
            0, alpha, alpha, jcp.dimN_block,
            jcp.dimK_nb_block, jcp.dimK_block,
            jcp.dimN_reg_block, jcp.dimK_reg_block);

    const bool output_is_aligned = ((size_t)out_ptr & (64 - 1)) == 0;

    const bool want_padded_bias = jcp.with_bias
        && jcp.oc_without_padding != jcp.oc;
    float last_slice_bias[simd_w] = {0};
    if (want_padded_bias) {
        for (int oc = 0; oc < jcp.oc_without_padding % jcp.oc_simd_block; ++oc)
            last_slice_bias[oc] = bias(jcp.dimM / jcp.dimM_simd_block - 1, oc);
    }

#pragma omp parallel
    {
        parallel_nd_in_omp(jcp.nb_oc, jcp.nb_ic, jcp.oc_block, jcp.ic_block,
            [&](int ofm1, int ifm1, int ofm2, int ifm2) {

            float *U_base_ptr = is_fwd
                              ? &(U(ofm1, 0, 0, ifm1, ofm2, ifm2, 0, 0))
                              : &(U(ifm1, 0, 0, ofm1, ifm2, ofm2, 0, 0));
            weight_transform_data<is_fwd>(jcp,
                    &(weights(ofm1 * jcp.oc_block + ofm2,
                            ifm1 * jcp.ic_block + ifm2,
                            0, 0, 0, 0)),
                    U_base_ptr);
        });

#pragma omp barrier

    int ithr = mkldnn_get_thread_num();

#pragma omp for schedule(static)
    for (int tile_block = 0; tile_block < jcp.tile_block; tile_block++) {
        for (int K_blk1 = 0; K_blk1 < jcp.dimK_nb_block; K_blk1++) {
            for (int K_blk2 = 0; K_blk2 < jcp.dimK_block; K_blk2++) {
                input_transform_tileblock_data<is_fwd>(
                        tile_block, jcp,
                        &(input(0, K_blk1 * jcp.dimK_block + K_blk2, 0, 0, 0)),
                        &(V(ithr, 0, 0, 0, K_blk1, K_blk2, 0, 0)));
            }
        }

        for (int oj = 0; oj < alpha; oj++) {
            for (int oi = 0; oi < alpha; oi++) {
                for (int M_blk1 = 0; M_blk1 < jcp.dimM_nb_block; M_blk1++) {
                    for (int N_blk = 0; N_blk < jcp.dimN_block; N_blk++) {
                        kernel_->gemm_loop_ker_first_iter(
                                (float *)&(M(ithr, M_blk1, oj, oi,
                                        N_blk, 0, 0, 0)),
                                (const float *)&(U(M_blk1, oj, oi, 0,
                                        0, 0, 0, 0)),
                                (const float *)&(V(ithr, oj, oi,
                                        N_blk, 0, 0, 0, 0)));
                        for (int K_blk1 = 1; K_blk1 < jcp.dimK_nb_block; K_blk1++) {
                            kernel_->gemm_loop_ker(
                                    (float *)&(M(ithr, M_blk1, oj, oi,
                                            N_blk, 0, 0, 0)),
                                    (const float *)&(U(M_blk1, oj, oi, K_blk1,
                                            0, 0, 0, 0)),
                                    (const float *)&(V(ithr, oj, oi,
                                            N_blk, K_blk1, 0, 0, 0)));
                        }
                    }
                }
            }
        }

        for (int M_blk1 = 0; M_blk1 < jcp.dimM_nb_block; M_blk1++) {
            for (int M_blk2 = 0; M_blk2 < jcp.dimM_block; M_blk2++) {
                const int M_blk = M_blk1 * jcp.dimM_block + M_blk2;

                float *bias_ptr = want_padded_bias
                    && M_blk == jcp.dimM / jcp.dimM_simd_block - 1
                    ? last_slice_bias : &bias(M_blk, 0);

                output_transform_tileblock(tile_block, jcp, p_ops,
                        &(M(ithr, M_blk1, 0, 0, 0, M_blk2, 0, 0)),
                        &(output(0, M_blk, 0, 0, 0)),
                        bias_ptr, output_is_aligned);
            }
        }
    }
    }
}

template void
_jit_avx512_common_convolution_winograd_t<true>::_execute_data_W_SGD(
        const int, float *, float *, float *, float *);
template void
_jit_avx512_common_convolution_winograd_t<false>::_execute_data_W_SGD(
        const int, float *, float *, float *, float *);

void jit_avx512_common_convolution_winograd_bwd_weights_t::
_maybe_execute_diff_bias_copy() {
    if (conf_.want_padded_bias()) {
        float *diff_bias = (float *)this->memory(1);
        for (int oc = 0; oc < conf_.jcp_.oc_without_padding; ++oc)
            diff_bias[oc] = this->padded_bias_[oc];
    }
}

void jit_avx512_common_convolution_winograd_bwd_weights_t::
_execute_backward_weights_S_D_G_W()
{
    const auto &jcp = kernel_->jcp;
    const int nthreads = scratchpad_->num_threads();

    auto diff_src_transform_bwd_weights_ver = jcp.ver == ver_4fma ?
            diff_src_transform_bwd_weights<true> :
            diff_src_transform_bwd_weights<false>;
    auto diff_dst_transform_bwd_weights_ver = jcp.with_bias
                                            ? diff_dst_transform_bwd_weights<true>
                                            : diff_dst_transform_bwd_weights<false>;

    array_offset_calculator<float, 5> diff_src((float *)this->input_memory(0),
            jcp.mb, jcp.ic/simd_w, jcp.ih, jcp.iw, simd_w);
    array_offset_calculator<float, 5> diff_dst((float *)this->input_memory(1),
            jcp.mb, jcp.oc/simd_w, jcp.oh, jcp.ow, simd_w);
    array_offset_calculator<float, 6> diff_weights((float *)this->memory(0),
            jcp.oc/simd_w, jcp.ic/simd_w, jcp.kh, jcp.kw, simd_w, simd_w);
    array_offset_calculator<float, 2> diff_bias(
            conf_.want_padded_bias() ? padded_bias_ : (float *)this->memory(1),
            jcp.oc/simd_w, simd_w);

    array_offset_calculator<float, 8> U(
            (float *)(scratchpad_->U_ptr()),
            jcp.nb_ic, jcp.nb_oc,
            alpha, alpha,
            jcp.oc_block, jcp.ic_block,
            jcp.ic_simd_block, jcp.oc_simd_block);

    array_offset_calculator<float, 8> M(
            (float *)(scratchpad_->M_ptr()),
            jcp.nb_oc, alpha, alpha,
            jcp.tile_block, jcp.oc_block,
            jcp.nb_tile_block_ur, jcp.tile_block_ur * jcp.tile_4fma,
            jcp.oc_simd_block);
    array_offset_calculator<float, 8> V(
            (float *)(scratchpad_->V_ptr()),
            jcp.nb_ic, alpha, alpha,
            jcp.tile_block, jcp.ic_block,
            jcp.nb_tile_block_ur, jcp.tile_block_ur,
            jcp.ic_simd_block * jcp.tile_4fma);

    const int trans_buffer_size = alpha * alpha * jcp.tile_4fma
                                * jcp.ic_simd_block;
    array_offset_calculator<float, 2> trans_buffer(
            (float *)(scratchpad_->src_transpose_ptr()),
            nthreads,
            trans_buffer_size);

    array_offset_calculator<float, 2> diff_bias_prv(
            (float *)(scratchpad_->bias_ptr()),
            mkldnn_get_max_threads(),
            jcp.oc);

#pragma omp parallel num_threads(nthreads)
    {
        if (jcp.with_bias) {
            parallel_nd_in_omp(nthreads, jcp.oc, [&](int ithr, int ofm) {
                diff_bias_prv(ithr, ofm) = 0.0f;
            });

#pragma omp for nowait
            for (int bofm = 0; bofm < jcp.oc / simd_w; bofm++) {
                PRAGMA_OMP_SIMD()
                for (int v = 0; v < simd_w; v++)
                    diff_bias(bofm, v) = 0.0f;
            }
        }

        const int ithread = mkldnn_get_thread_num();

        parallel_nd_in_omp(jcp.mb, jcp.nb_ic, jcp.ic_block,
            [&](int img, int ifm1, int ifm2) {
            float *transb = jcp.ver == ver_4fma
               ? &(trans_buffer(ithread, 0))
               : NULL;
            diff_src_transform_bwd_weights_ver(img, jcp,
               &(diff_src(img, ifm1 * jcp.ic_block + ifm2,
                       0, 0, 0)),
               &(V(ifm1, 0, 0, 0, ifm2, 0, 0, 0)),
               transb,
               kernel_->transpose_4fma_ker);
        });

        parallel_nd_in_omp(jcp.mb, jcp.nb_oc, jcp.oc_block,
            [&](int img, int ofm1, int ofm2) {
            float *dbias = jcp.with_bias
                   ? &(diff_bias_prv(ithread,
                               simd_w * (ofm1 * jcp.oc_block + ofm2)))
                   : NULL;
            diff_dst_transform_bwd_weights_ver(img, jcp,
                    &(diff_dst(img, ofm1 * jcp.oc_block + ofm2,
                            0, 0, 0)),
                    &(M(ofm1, 0, 0, 0, ofm2, 0, 0, 0)),
                    dbias);
        });

#pragma omp barrier

        for (int ifm1 = 0; ifm1 < jcp.nb_ic; ifm1++) {
            parallel_nd_in_omp(alpha, alpha, jcp.nb_oc,
                [&](int oj, int oi, int ofm1) {
                kernel_->gemm_loop_ker_first_iter(
                    (float *)&(U(ifm1, ofm1, oj, oi,
                            0, 0, 0, 0)),
                    (const float *)&(M(ofm1, oj, oi,
                            0, 0, 0, 0, 0)),
                    (const float *)&(V(ifm1, oj, oi,
                            0, 0, 0, 0, 0)));
                for (int tile_block = 1; tile_block < jcp.tile_block;
                     tile_block++) {
                    kernel_->gemm_loop_ker((float *)&(U(ifm1, ofm1,
                                oj, oi,
                                0, 0, 0, 0)),
                        (const float *)&(M(ofm1, oj, oi, tile_block,
                                0, 0, 0, 0)),
                        (const float *)&(V(ifm1, oj, oi, tile_block,
                                0, 0, 0, 0)));
                }
            });
        }

#pragma omp barrier

        parallel_nd_in_omp(jcp.nb_ic, jcp.nb_oc, jcp.oc_block, jcp.ic_block,
            [&](int ifm1, int ofm1, int ofm2, int ifm2) {
            diff_weights_transform_bwd_weights(jcp,
                    &(diff_weights(ofm1 * jcp.oc_block + ofm2,
                            ifm1 * jcp.ic_block + ifm2, 0, 0, 0, 0)),
                    &(U(ifm1, ofm1, 0, 0, ofm2, ifm2, 0, 0)));
        });

        if (jcp.with_bias) {
#pragma omp for
            for (int ofm1 = 0; ofm1 < jcp.oc / simd_w; ofm1++) {
                for (int ithr = 0; ithr < nthreads; ithr++) {
                    float* base_bias_ptr = &(diff_bias(ofm1, 0));
                    float* base_bias_prv_ptr = &(diff_bias_prv(
                                ithr * jcp.oc + ofm1 * simd_w));
                    PRAGMA_OMP_SIMD()
                    for (int ofm2 = 0; ofm2 < simd_w; ofm2++) {
                        base_bias_ptr[ofm2] += base_bias_prv_ptr[ofm2];
                    }
                }
            }
        }
    }

    _maybe_execute_diff_bias_copy();
}

namespace {

const int max_threads_number = 1024;

template <bool ver_4fma>
void diff_src_transform_bwd_weights_tile(int tile_block,
    jit_conv_winograd_conf_t conv, float *inp, float *tinp,
    void(*transpose_4fma_ker)(float *, float *))
{
    const int ifwp = conv.iw + conv.l_pad;
    const int ifhp = conv.ih + conv.t_pad;
    float I[alpha][alpha][simd_w];
    float Iw[alpha][alpha][simd_w];

    float *Iw_buffer = nullptr;
    if (ver_4fma) {
        Iw_buffer = (float *)malloc(alpha * alpha * conv.tile_4fma
            * simd_w * sizeof(float), 64);
    }
    array_offset_calculator<float, 4> Iw_scratchpad(Iw_buffer,
        alpha, alpha, conv.tile_4fma, simd_w);
    array_offset_calculator<float, 5> input(inp,
        conv.mb, conv.ic / simd_w, conv.ih, conv.iw, simd_w);
    array_offset_calculator<float, 7> output(tinp,
        0, alpha, alpha,
        conv.ic_block,
        conv.nb_tile_block_ur, conv.tile_block_ur,
        conv.ic_simd_block * conv.tile_4fma);

    int tile_4fma = 0;

    int n_tiles = tile_block * conv.nb_tile_block_ur * conv.tile_block_ur;
    for (int nb_tile_block_ur = 0; nb_tile_block_ur < conv.nb_tile_block_ur;
        nb_tile_block_ur++) {
        for (int tile_block_ur = 0; tile_block_ur < conv.tile_block_ur;
            tile_block_ur++) {

            int img = n_tiles / (conv.jtiles * conv.itiles);
            int no_tile = n_tiles % (conv.jtiles * conv.itiles);
            int ti = no_tile % conv.itiles;
            int tj = no_tile / conv.itiles;

            for (int j = 0; j < alpha; j++) {
                int ydim = tj * tile_size + j;
                if ((conv.t_pad <= ydim) && ydim < ifhp) {
                    for (int i = 0; i < alpha; i++) {
                        int xdim = ti * tile_size + i;
                        if ((conv.l_pad <= xdim) && xdim < ifwp) {
                            PRAGMA_OMP_SIMD()
                            for (int v = 0; v < simd_w; v++) {
                                I[j][i][v] = input(img, 0,
                                    ydim - conv.t_pad,
                                    xdim - conv.l_pad, v);
                            }
                        }
                        else {
                            PRAGMA_OMP_SIMD()
                            for (int v = 0; v < simd_w; v++) {
                                I[j][i][v] = 0.0f;
                            }
                        }
                    }
                }
                else {
                    for (int i = 0; i < alpha; i++) {
                        PRAGMA_OMP_SIMD()
                        for (int v = 0; v < simd_w; v++) {
                            I[j][i][v] = 0.0f;
                        }
                    }
                }
            }

            trans_I_4x4_3x3(Iw, I);

            if (ver_4fma) {
                for (int j = 0; j < alpha; j++) {
                    for (int i = 0; i < alpha; i++) {
                        PRAGMA_OMP_SIMD()
                        for (int v = 0; v < simd_w; v++) {
                            Iw_scratchpad(j, i, tile_4fma, v) = Iw[j][i][v];
                        }
                    }
                }
                tile_4fma++;
                if (tile_4fma == conv.tile_4fma) {
                    float *outp = &(output(0, 0, 0, 0,
                        nb_tile_block_ur, tile_block_ur, 0));
                    transpose_4fma_ker(outp, (float *)Iw_buffer);
                    tile_4fma = 0;
                }
            }
            else {
                for (int j = 0; j < alpha; j++) {
                    for (int i = 0; i < alpha; i++) {
                        store_output(
                            &(output(0, j, i, 0,
                                nb_tile_block_ur, tile_block_ur, 0)),
                            Iw[j][i], false);

                    }
                }
            }
            n_tiles++;
        }
    }
}

template <bool with_bias>
void diff_dst_transform_bwd_weights_tile(int tile_block,
    jit_conv_winograd_conf_t conv, float *inp, float *tinp, float *dbias)
{
    float I[alpha][alpha][simd_w];
    float Iw[alpha][alpha][simd_w];

    array_offset_calculator<float, 5> input(inp,
        conv.mb, conv.oc / simd_w, conv.oh, conv.ow, conv.oc_simd_block);
    array_offset_calculator<float, 7> output(tinp,
        conv.nb_oc, alpha, alpha,
        conv.oc_block,
        conv.nb_tile_block_ur,
        conv.tile_block_ur * conv.tile_4fma, conv.oc_simd_block);

    int n_tiles = tile_block * conv.nb_tile_block_ur * conv.tile_block_ur;
    for (int nb_tile_block_ur = 0; nb_tile_block_ur < conv.nb_tile_block_ur;
        nb_tile_block_ur++) {
        for (int tile_block_ur = 0; tile_block_ur < conv.tile_block_ur;
            tile_block_ur++) {

            int img = n_tiles / (conv.jtiles * conv.itiles);
            int no_tile = n_tiles % (conv.jtiles * conv.itiles);
            int ti = no_tile % conv.itiles;
            int tj = no_tile / conv.itiles;

            for (int j = 0; j < alpha; j++) {
                int ydim = tj * tile_size + j;
                if (ydim < conv.oh) {
                    for (int i = 0; i < alpha; i++) {
                        int xdim = ti * tile_size + i;
                        if (xdim < conv.ow) {
                            float *input_base = &input(img, 0, ydim, xdim, 0);

                            PRAGMA_OMP_SIMD()
                            for (int v = 0; v < simd_w; v++) {
                                I[j][i][v] = input_base[v];
                            }
                            if (with_bias && j < tile_size && i < tile_size) {
                                PRAGMA_OMP_SIMD()
                                for (int v = 0; v < simd_w; v++) {
                                    dbias[v] += input_base[v];
                                }
                            }
                        }
                        else {
                            PRAGMA_OMP_SIMD()
                            for (int v = 0; v < simd_w; v++) {
                                I[j][i][v] = 0.0f;
                            }
                        }
                    }
                }
                else {
                    for (int i = 0; i < alpha; i++) {
                        PRAGMA_OMP_SIMD()
                        for (int v = 0; v < simd_w; v++) {
                            I[j][i][v] = 0.0f;
                        }
                    }
                }
            }

            trans_W_3x3_4x4_wu(Iw, I);

            for (int j = 0; j < alpha; j++) {
                for (int i = 0; i < alpha; i++) {
                    /*TODO: Try instrinsic for casting into __m512*/
                    store_output(&(output(0, j, i, 0,
                        nb_tile_block_ur, tile_block_ur, 0)),
                        Iw[j][i], false);
                }
            }
            n_tiles++;
        }
    }
}

// Sum to the first buffer array
void array_sum(int num_arrs, float *output,
    size_t nelems, float *input_ptrs[], bool reduce_to_first = true)
{
    const size_t block_size = 16 * 1024 / sizeof(float);
    const size_t blocks_number = nelems / block_size;
    const size_t tail = nelems % block_size;

#pragma omp parallel
    {
        const int ithr = mkldnn_get_thread_num();
        const int nthr = mkldnn_get_num_threads();
        size_t start{ 0 }, end{ 0 };
        balance211(blocks_number, nthr, ithr, start, end);

        for (size_t nb = start; nb < end; ++nb) {
            size_t start_e = nb * block_size;
            size_t end_e = start_e + block_size;
            if (!reduce_to_first) {
                PRAGMA_OMP_SIMD()
                for (size_t e = start_e; e < end_e; e++) {
                    output[e] = input_ptrs[0][e];
                }
            }
            for (int a = 1; a < num_arrs; a++) {
                PRAGMA_OMP_SIMD()
                for (size_t e = start_e; e < end_e; e++) {
                    output[e] += input_ptrs[a][e];
                }
            }
        }

        if (tail != 0 && ithr == nthr - 1) {
            size_t start_e = nelems - tail;
            size_t end_e = nelems;
            if (!reduce_to_first) {
                PRAGMA_OMP_SIMD()
                for (size_t e = start_e; e < end_e; e++) {
                    output[e] = input_ptrs[0][e];
                }
            }
            for (int a = 1; a < num_arrs; a++) {
                PRAGMA_OMP_SIMD()
                for (size_t e = start_e; e < end_e; e++) {
                    output[e] += input_ptrs[a][e];
                }
            }
        }
    }
}

void subarray_sum(int num_arrs, float *output, size_t nelems,
        float *input_ptrs[], size_t input_starts[], size_t input_ends[])
{
    using namespace nstl;
    const size_t block_size = 16 * 1024 / sizeof(float);
    const size_t blocks_number = nelems / block_size;
    const size_t tail = nelems % block_size;

#pragma omp parallel
    {
        const int ithr = mkldnn_get_thread_num();
        const int nthr = mkldnn_get_num_threads();
        size_t start{ 0 }, end{ 0 };
        balance211(blocks_number, nthr, ithr, start, end);

        for (size_t nb = start; nb < end; ++nb) {
            size_t start_e = nb * block_size;
            size_t end_e = start_e + block_size;
            size_t input_start = max(start_e, min(input_starts[0], end_e));
            size_t input_end = max(start_e, min(input_ends[0], end_e));

            PRAGMA_OMP_SIMD()
            for (size_t e = start_e; e < input_start; e++) {
                output[e] = 0.f;
            }

            PRAGMA_OMP_SIMD()
            for (size_t e = input_start; e < input_end; e++) {
                output[e] = input_ptrs[0][e];
            }

            PRAGMA_OMP_SIMD()
            for (size_t e = input_end; e < end_e; e++) {
                output[e] = 0.f;
            }
            for (int a = 1; a < num_arrs; a++) {
                input_start = max(start_e, input_starts[a]);
                input_end = min(input_ends[a], end_e);

                PRAGMA_OMP_SIMD()
                for (size_t e = input_start; e < input_end; e++) {
                    output[e] += input_ptrs[a][e];
                }
            }
        }

        if (tail != 0 && ithr == nthr - 1) {
            size_t start_e = nelems - tail;
            size_t end_e = nelems;
            size_t input_start = max(start_e, min(input_starts[0], end_e));
            size_t input_end = max(start_e, min(input_ends[0], end_e));

            PRAGMA_OMP_SIMD()
            for (size_t e = start_e; e < input_start; e++) {
                output[e] = 0.f;
            }

            PRAGMA_OMP_SIMD()
            for (size_t e = input_start; e < input_end; e++) {
                output[e] = input_ptrs[0][e];
            }

            PRAGMA_OMP_SIMD()
            for (size_t e = input_end; e < end_e; e++) {
                output[e] = 0.f;
            }
            for (int a = 1; a < num_arrs; a++) {
                input_start = max(start_e, input_starts[a]);
                input_end = min(input_ends[a], end_e);

                PRAGMA_OMP_SIMD()
                for (size_t e = start_e; e < end_e; e++) {
                    output[e] += input_ptrs[a][e];
                }
            }
        }
    }
}
} // namespace

void jit_avx512_common_convolution_winograd_bwd_weights_t::
_execute_backward_weights_S_D_Giot_W()
{
    const auto &jcp = kernel_->jcp;
    const int nthreads = scratchpad_->num_threads();
    int U_size = jcp.oc * jcp.ic * alpha * alpha * sizeof(float);

    auto diff_src_transform_bwd_weights_ver = jcp.ver == ver_4fma ?
            diff_src_transform_bwd_weights<true> :
            diff_src_transform_bwd_weights<false>;
    auto diff_dst_transform_bwd_weights_ver = jcp.with_bias
                                        ? diff_dst_transform_bwd_weights<true>
                                        : diff_dst_transform_bwd_weights<false>;

    array_offset_calculator<float, 5> diff_src((float *)this->input_memory(0),
            jcp.mb, jcp.ic / simd_w, jcp.ih, jcp.iw, simd_w);
    array_offset_calculator<float, 5> diff_dst((float *)this->input_memory(1),
            jcp.mb, jcp.oc / simd_w, jcp.oh, jcp.ow, simd_w);
    array_offset_calculator<float, 6> diff_weights((float *)this->memory(0),
            jcp.oc / simd_w, jcp.ic / simd_w, jcp.kh, jcp.kw, simd_w, simd_w);
    array_offset_calculator<float, 2> diff_bias(
            conf_.want_padded_bias() ? padded_bias_ : (float *)this->memory(1),
            jcp.oc / simd_w, simd_w);

    array_offset_calculator<float, 8> U((float *)(scratchpad_->U_ptr()),
            jcp.nb_ic, jcp.nb_oc,
            alpha, alpha,
            jcp.oc_block, jcp.ic_block,
            jcp.ic_simd_block, jcp.oc_simd_block);

    array_offset_calculator<float, 9> Us(
            (float *)(scratchpad_->U_ptr() + U_size),
            0, jcp.nb_ic, jcp.nb_oc,
            alpha, alpha,
            jcp.oc_block, jcp.ic_block,
            jcp.ic_simd_block, jcp.oc_simd_block);

    array_offset_calculator<float, 8> M((float *)(scratchpad_->M_ptr()),
            jcp.nb_oc, alpha, alpha,
            jcp.tile_block, jcp.oc_block,
            jcp.nb_tile_block_ur, jcp.tile_block_ur * jcp.tile_4fma,
            jcp.oc_simd_block);

    array_offset_calculator<float, 8> V((float *)(scratchpad_->V_ptr()),
            jcp.nb_ic, alpha, alpha,
            jcp.tile_block, jcp.ic_block,
            jcp.nb_tile_block_ur, jcp.tile_block_ur,
            jcp.ic_simd_block * jcp.tile_4fma);

    const int trans_buffer_size = alpha * alpha * jcp.tile_4fma
        * jcp.ic_simd_block;
    array_offset_calculator<float, 2> trans_buffer(
        (float *)(scratchpad_->src_transpose_ptr()),
        nthreads,
        trans_buffer_size);

    array_offset_calculator<float, 2> diff_bias_prv(
            (float *)(scratchpad_->bias_ptr()), nthreads, jcp.oc);

#pragma omp parallel
    {
        if (jcp.with_bias) {
            parallel_nd_in_omp(nthreads, jcp.oc, [&](int ithr, int ofm) {
                diff_bias_prv(ithr, ofm) = 0.0f;
            });
#pragma omp for nowait
            for (int bofm = 0; bofm < jcp.oc / simd_w; bofm++) {
                PRAGMA_OMP_SIMD()
                for (int v = 0; v < simd_w; v++)
                    diff_bias(bofm, v) = 0.0f;
            }
        }
    }

#pragma omp parallel
    {
        const int ithread = mkldnn_get_thread_num();
        parallel_nd_in_omp(jcp.mb, jcp.nb_ic, jcp.ic_block,
            [&](int img, int ifm1, int ifm2) {
                float *transb = jcp.ver == ver_4fma
                    ? &(trans_buffer(ithread, 0))
                    : NULL;
                diff_src_transform_bwd_weights_ver(img, jcp,
                    &(diff_src(img, ifm1 * jcp.ic_block + ifm2,
                            0, 0, 0)),
                    &(V(ifm1, 0, 0, 0, ifm2, 0, 0, 0)),
                    transb,
                    kernel_->transpose_4fma_ker);
        });
    }

#pragma omp parallel num_threads(nthreads)
    {
        parallel_nd_in_omp(jcp.mb, jcp.nb_oc, jcp.oc_block,
            [&](int img, int ofm1, int ofm2) {
                const int ithread = mkldnn_get_thread_num();
                float *dbias = jcp.with_bias
                    ? &(diff_bias_prv(ithread,
                        simd_w * (ofm1 * jcp.oc_block + ofm2)))
                    : NULL;
                diff_dst_transform_bwd_weights_ver(img, jcp,
                    &(diff_dst(img, ofm1 * jcp.oc_block + ofm2, 0, 0, 0)),
                    &(M(ofm1, 0, 0, 0, ofm2, 0, 0, 0)), dbias);
        });
    }

    size_t input_starts[max_threads_number];
    size_t input_ends[max_threads_number];
    int th_counter = 0;
#pragma omp parallel firstprivate(th_counter) num_threads(nthreads)
    {
        parallel_nd_in_omp(jcp.nb_ic, jcp.nb_oc, alpha, alpha, jcp.tile_block,
            [&](int ifm1, int ofm1, int oj, int oi, int tile_block) {
                int ithr = mkldnn_get_thread_num();
                if (th_counter == 0) {
                    input_starts[ithr] = (float *)&(Us(ithr, ifm1, ofm1,
                            oj, oi, 0, 0, 0, 0)) - (float *)&(Us(ithr, 0, 0,
                            0, 0, 0, 0, 0, 0));
                    input_ends[ithr] = input_starts[ithr]
                        + jcp.oc_block * jcp.ic_block
                        * jcp.ic_simd_block * jcp.oc_simd_block;
                }
                else if (tile_block == 0) {
                    input_ends[ithr] += jcp.oc_block * jcp.ic_block
                        * jcp.ic_simd_block * jcp.oc_simd_block;
                }

                if (th_counter == 0 || tile_block == 0) {
                    kernel_->gemm_loop_ker_first_iter(
                        &(Us(ithr, ifm1, ofm1, oj, oi, 0, 0, 0, 0)),
                        &(M(ofm1, oj, oi, tile_block, 0, 0, 0, 0)),
                        &(V(ifm1, oj, oi, tile_block, 0, 0, 0, 0)));
                } else {
                    kernel_->gemm_loop_ker(
                        &(Us(ithr, ifm1, ofm1, oj, oi, 0, 0, 0, 0)),
                        &(M(ofm1, oj, oi, tile_block, 0, 0, 0, 0)),
                        &(V(ifm1, oj, oi, tile_block, 0, 0, 0, 0)));
                }
                th_counter++;
        });
    }


    // Reduce diff-weights
    {
        float *output = &(U(0, 0, 0, 0, 0, 0, 0, 0));
        size_t nelems = jcp.ic * jcp.oc * alpha * alpha;
        float *input_ptrs[max_threads_number];
        for (int i = 0; i < nthreads; i++)
            input_ptrs[i] = output + nelems * (i + 1);
        subarray_sum(
                nthreads, output, nelems, input_ptrs, input_starts, input_ends);
    }

    parallel_nd(jcp.nb_ic, jcp.nb_oc, jcp.oc_block, jcp.ic_block,
        [&](int ifm1, int ofm1, int ofm2, int ifm2) {
            diff_weights_transform_bwd_weights(jcp,
                    &(diff_weights(ofm1 * jcp.oc_block + ofm2,
                            ifm1 * jcp.ic_block + ifm2,
                            0, 0, 0, 0)),
                    &(U(ifm1, ofm1, 0, 0, ofm2, ifm2, 0, 0)));
    });

#pragma omp parallel
    if (jcp.with_bias) {
#pragma omp for
        for (int ofm1 = 0; ofm1 < jcp.oc / simd_w; ofm1++) {
            for (int ithr = 0; ithr < nthreads; ithr++) {
                float* base_bias_ptr = &(diff_bias(ofm1, 0));
                float* base_bias_prv_ptr = &(diff_bias_prv(
                            ithr * jcp.oc + ofm1 * simd_w));
                PRAGMA_OMP_SIMD()
                for (int ofm2 = 0; ofm2 < simd_w; ofm2++) {
                    base_bias_ptr[ofm2] += base_bias_prv_ptr[ofm2];
                }
            }
        }
    }

    _maybe_execute_diff_bias_copy();
}

void jit_avx512_common_convolution_winograd_bwd_weights_t::
_execute_backward_weights_SDGtWo()
{
    const auto &jcp = kernel_->jcp;
    const int nthreads = scratchpad_->num_threads();

    auto diff_src_transform_bwd_weights_ver_tile = jcp.ver == ver_4fma ?
            diff_src_transform_bwd_weights_tile<true> :
            diff_src_transform_bwd_weights_tile<false>;
    auto diff_dst_transform_bwd_weights_ver = jcp.with_bias
                                  ? diff_dst_transform_bwd_weights_tile<true>
                                  : diff_dst_transform_bwd_weights_tile<false>;

    array_offset_calculator<float, 5> diff_src((float *)this->input_memory(0),
            jcp.mb, jcp.ic / simd_w, jcp.ih, jcp.iw, simd_w);
    array_offset_calculator<float, 5> diff_dst((float *)this->input_memory(1),
            jcp.mb, jcp.oc / simd_w, jcp.oh, jcp.ow, simd_w);
    array_offset_calculator<float, 6> diff_weights((float *)this->memory(0),
            jcp.oc / simd_w, jcp.ic / simd_w, jcp.kh, jcp.kw, simd_w, simd_w);
    array_offset_calculator<float, 3> diff_bias(
            conf_.want_padded_bias() ? padded_bias_ : (float *)this->memory(1),
            jcp.nb_oc, jcp.oc_block, simd_w);

    array_offset_calculator<float, 8> Us((float *)(scratchpad_->U_ptr()),
            0, jcp.nb_ic, alpha, alpha,
            jcp.oc_block, jcp.ic_block,
            jcp.ic_simd_block, jcp.oc_simd_block);

    array_offset_calculator<float, 7> M((float *)(scratchpad_->M_ptr()),
            0, alpha, alpha,
            jcp.oc_block,
            jcp.nb_tile_block_ur, jcp.tile_block_ur * jcp.tile_4fma,
            jcp.oc_simd_block);

    array_offset_calculator<float, 8> V((float *)(scratchpad_->V_ptr()),
            0, jcp.nb_ic, alpha, alpha,
            jcp.ic_block,
            jcp.nb_tile_block_ur, jcp.tile_block_ur,
            jcp.ic_simd_block * jcp.tile_4fma);

    array_offset_calculator<float, 2> diff_bias_prv(
            (float *)(scratchpad_->bias_ptr()),
            nthreads, jcp.oc / jcp.nb_oc);

    for (int ofm1 = 0; ofm1 < jcp.nb_oc; ++ofm1) {
        int th_counter = 0;

#pragma omp parallel
        {
            if (jcp.with_bias) {
                parallel_nd_in_omp(nthreads, jcp.oc / jcp.nb_oc,
                    [&](int ithr, int ofm) {
                        diff_bias_prv(ithr, ofm) = 0.0f;
                });
#pragma omp for nowait
                for (int bofm = 0; bofm < jcp.oc_block; bofm++) {
                    PRAGMA_OMP_SIMD()
                    for (int v = 0; v < simd_w; v++)
                        diff_bias(ofm1, bofm, v) = 0.0f;
                }
            }
        }

#pragma omp parallel firstprivate(th_counter) num_threads(nthreads)
#pragma omp for nowait
        for (int tile_block = 0; tile_block < jcp.tile_block; tile_block++) {
            int ithr = mkldnn_get_thread_num();
            for (int ifm1 = 0; ifm1 < jcp.nb_ic; ++ifm1) {
                for (int ifm2 = 0; ifm2 < jcp.ic_block; ++ifm2) {
                    diff_src_transform_bwd_weights_ver_tile(tile_block, jcp,
                            &(diff_src(0, ifm1 * jcp.ic_block + ifm2, 0, 0, 0)),
                            &(V(ithr, ifm1, 0, 0, ifm2, 0, 0, 0)),
                            kernel_->transpose_4fma_ker);
                }
            }

            for (int ofm2 = 0; ofm2 < jcp.oc_block; ofm2++) {
                float *dbias = jcp.with_bias
                    ? &(diff_bias_prv(ithr, simd_w * ofm2))
                    : NULL;
                diff_dst_transform_bwd_weights_ver(tile_block, jcp,
                        &(diff_dst(0, ofm1 * jcp.oc_block + ofm2, 0, 0, 0)),
                        &(M(ithr, 0, 0, ofm2, 0, 0, 0)),
                        dbias);
            }

            for (int ifm1 = 0; ifm1 < jcp.nb_ic; ifm1++) {
                for (int oj = 0; oj < alpha; oj++) {
                    for (int oi = 0; oi < alpha; oi++) {
                        if (th_counter == 0)
                            kernel_->gemm_loop_ker_first_iter(
                                    &(Us(ithr, ifm1, oj, oi, 0, 0, 0, 0)),
                                    &(M(ithr, oj, oi, 0, 0, 0, 0)),
                                    &(V(ithr, ifm1, oj, oi, 0, 0, 0, 0)));
                        else
                            kernel_->gemm_loop_ker(
                                    &(Us(ithr, ifm1, oj, oi, 0, 0, 0, 0)),
                                    &(M(ithr, oj, oi, 0, 0, 0, 0)),
                                    &(V(ithr, ifm1, oj, oi, 0, 0, 0, 0)));
                    }
                }
            }
            th_counter++;
        }
        // Reduce diff-weights
        {
            float *output = (float *)(scratchpad_->U_ptr());
            size_t nelems
                    = jcp.ic * (jcp.oc / jcp.nb_oc) * alpha * alpha;
            float *input_ptrs[max_threads_number];
            for (int i = 0; i < nthreads; i++) {
                input_ptrs[i] = output + nelems * i;
            }
            array_sum(nthreads, output, nelems, input_ptrs);
        }

        parallel_nd(jcp.nb_ic, jcp.oc_block, jcp.ic_block,
            [&](int ifm1, int ofm2, int ifm2) {
            diff_weights_transform_bwd_weights(jcp,
                    &(diff_weights(ofm1 * jcp.oc_block + ofm2,
                            ifm1 * jcp.ic_block + ifm2,
                            0, 0, 0, 0)),
                    &(Us(0, ifm1, 0, 0, ofm2, ifm2, 0, 0)));
        });

#pragma omp parallel
        if (jcp.with_bias) {
#pragma omp for
            for (int ofm2 = 0; ofm2 < jcp.oc_block; ofm2++) {
                for (int ithr = 0; ithr < nthreads; ithr++) {
                    float* base_bias_ptr = &(diff_bias(ofm1, ofm2, 0));
                    float* base_bias_prv_ptr = &(diff_bias_prv(
                                ithr * jcp.oc_block * simd_w + ofm2 * simd_w));
                    PRAGMA_OMP_SIMD()
                    for (int ofm3 = 0; ofm3 < simd_w; ofm3++) {
                        base_bias_ptr[ofm3] += base_bias_prv_ptr[ofm3];
                    }
                }
            }
        }
    }

    _maybe_execute_diff_bias_copy();
}

void jit_avx512_common_convolution_winograd_bwd_weights_t::
_execute_backward_weights_SDGt_W()
{
    const auto &jcp = kernel_->jcp;
    const int nthreads = scratchpad_->num_threads();

    auto diff_src_transform_bwd_weights_ver_tile = jcp.ver == ver_4fma ?
            diff_src_transform_bwd_weights_tile<true> :
            diff_src_transform_bwd_weights_tile<false>;
    auto diff_dst_transform_bwd_weights_ver = jcp.with_bias
                                  ? diff_dst_transform_bwd_weights_tile<true>
                                  : diff_dst_transform_bwd_weights_tile<false>;

    array_offset_calculator<float, 5> diff_src((float *)this->input_memory(0),
            jcp.mb, jcp.ic / simd_w, jcp.ih, jcp.iw, simd_w);
    array_offset_calculator<float, 5> diff_dst((float *)this->input_memory(1),
            jcp.mb, jcp.oc / simd_w, jcp.oh, jcp.ow, simd_w);
    array_offset_calculator<float, 6> diff_weights((float *)this->memory(0),
            jcp.oc / simd_w, jcp.ic / simd_w, jcp.kh, jcp.kw, simd_w, simd_w);
    array_offset_calculator<float, 2> diff_bias(
            conf_.want_padded_bias() ? padded_bias_ : (float *)this->memory(1),
            jcp.oc / simd_w, simd_w);

    array_offset_calculator<float, 8> U((float *)(scratchpad_->U_ptr()),
            jcp.nb_oc, jcp.nb_ic,
            alpha, alpha,
            jcp.oc_block, jcp.ic_block,
            jcp.ic_simd_block, jcp.oc_simd_block);

    array_offset_calculator<float, 9> Us((float *)(scratchpad_->U_ptr()),
            0, jcp.nb_oc, jcp.nb_ic,
            alpha, alpha,
            jcp.oc_block, jcp.ic_block,
            jcp.ic_simd_block, jcp.oc_simd_block);

    array_offset_calculator<float, 8> M((float *)(scratchpad_->M_ptr()),
            0, jcp.nb_oc, alpha, alpha, jcp.oc_block,
            jcp.nb_tile_block_ur, jcp.tile_block_ur * jcp.tile_4fma,
            jcp.oc_simd_block);

    array_offset_calculator<float, 8> V((float *)(scratchpad_->V_ptr()),
            0, jcp.nb_ic, alpha, alpha, jcp.ic_block,
            jcp.nb_tile_block_ur, jcp.tile_block_ur,
            jcp.ic_simd_block * jcp.tile_4fma);

    array_offset_calculator<float, 2> diff_bias_prv(
            (float *)(scratchpad_->bias_ptr()),
            nthreads, jcp.oc);

#pragma omp parallel
    {
        if (jcp.with_bias) {
            parallel_nd_in_omp(nthreads, jcp.oc,
                [&](int ithr, int ofm) {
                    diff_bias_prv(ithr, ofm) = 0.0f;
            });
#pragma omp for nowait
            for (int bofm = 0; bofm < jcp.oc / simd_w; bofm++) {
                PRAGMA_OMP_SIMD()
                for (int v = 0; v < simd_w; v++)
                    diff_bias(bofm, v) = 0.0f;
            }
        }
    }

    int th_counter = 0;
#pragma omp parallel firstprivate(th_counter) num_threads(nthreads)
#pragma omp for nowait
    for (int tile_block = 0; tile_block < jcp.tile_block; tile_block++) {
        int ithr = mkldnn_get_thread_num();

        for (int ifm1 = 0; ifm1 < jcp.nb_ic; ++ifm1) {
            for (int ifm2 = 0; ifm2 < jcp.ic_block; ++ifm2) {
                diff_src_transform_bwd_weights_ver_tile(tile_block, jcp,
                        &(diff_src(0, ifm1 * jcp.ic_block + ifm2,
                                0, 0, 0)),
                        &(V(ithr, ifm1, 0, 0, ifm2, 0, 0, 0)),
                        kernel_->transpose_4fma_ker);
            }
        }

        for (int ofm1 = 0; ofm1 < jcp.nb_oc; ofm1++) {
            for (int ofm2 = 0; ofm2 < jcp.oc_block; ofm2++) {
                float *dbias = jcp.with_bias
                    ? &(diff_bias_prv(ithr,
                                simd_w * (ofm1 * jcp.oc_block + ofm2)))
                    : NULL;
                diff_dst_transform_bwd_weights_ver(tile_block, jcp,
                        &(diff_dst(0, ofm1 * jcp.oc_block + ofm2,
                                0, 0, 0)),
                        &(M(ithr, ofm1, 0, 0, ofm2, 0, 0, 0)),
                        dbias);
            }
        }

        for (int ofm1 = 0; ofm1 < jcp.nb_oc; ofm1++) {
            for (int oj = 0; oj < alpha; oj++) {
                for (int oi = 0; oi < alpha; oi++) {
                    for (int ifm1 = 0; ifm1 < jcp.nb_ic; ifm1++) {
                        if (th_counter == 0)
                            kernel_->gemm_loop_ker_first_iter(
                                    &(Us(ithr, ofm1, ifm1, oj, oi, 0, 0, 0, 0)),
                                    &(M(ithr, ofm1, oj, oi, 0, 0, 0, 0)),
                                    &(V(ithr, ifm1, oj, oi, 0, 0, 0, 0)));
                        else
                            kernel_->gemm_loop_ker(
                                    &(Us(ithr, ofm1, ifm1, oj, oi, 0, 0, 0, 0)),
                                    &(M(ithr, ofm1, oj, oi, 0, 0, 0, 0)),
                                    &(V(ithr, ifm1, oj, oi, 0, 0, 0, 0)));
                    }
                }
            }
        }
        th_counter++;
    }

    // Reduce diff-weights
    {
        float *output = (float *)(scratchpad_->U_ptr());
        size_t nelems = jcp.ic * jcp.oc * alpha * alpha;
        float *input_ptrs[max_threads_number];
        for (int i = 0; i < nthreads; i++) {
            input_ptrs[i] = output + nelems * i;
        }
        array_sum(nthreads, output, nelems, input_ptrs);
    }

    parallel_nd(jcp.nb_oc, jcp.nb_ic, jcp.oc_block, jcp.ic_block,
        [&](int ofm1, int ifm1, int ofm2, int ifm2) {
        diff_weights_transform_bwd_weights(jcp,
                &(diff_weights(ofm1 * jcp.oc_block + ofm2,
                        ifm1 * jcp.ic_block + ifm2, 0, 0, 0, 0)),
                &(U(ofm1, ifm1, 0, 0, ofm2, ifm2, 0, 0)));
    });

#pragma omp parallel
    if (jcp.with_bias) {
#pragma omp for
        for (int ofm1 = 0; ofm1 < jcp.oc / simd_w; ofm1++) {
            for (int ithr = 0; ithr < nthreads; ithr++) {
                float* base_bias_ptr = &(diff_bias(ofm1, 0));
                float* base_bias_prv_ptr = &(diff_bias_prv(
                            ithr * jcp.oc + ofm1 * simd_w));
                PRAGMA_OMP_SIMD()
                for (int ofm2 = 0; ofm2 < simd_w; ofm2++) {
                    base_bias_ptr[ofm2] += base_bias_prv_ptr[ofm2];
                }
            }
        }
    }

    _maybe_execute_diff_bias_copy();
}
}
}
}
// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
