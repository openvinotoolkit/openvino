/*******************************************************************************
 * Copyright 2020-2021 Intel Corporation
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

#ifndef GPU_OCL_OFFSETS_H
#define GPU_OCL_OFFSETS_H

int off_ncdhw(int n, int c, int d, int h, int w, int C, int D, int H, int W) {
    int off = 0;
    off += n * C * D * H * W;
    off += c * D * H * W;
    off += d * H * W;
    off += h * W;
    off += w;
    return off;
}
int off_ndhwc(int n, int c, int d, int h, int w, int C, int D, int H, int W) {
    int off = 0;
    off += n * D * H * W * C;
    off += d * H * W * C;
    off += h * W * C;
    off += w * C;
    off += c;
    return off;
}

int off_nCdhw16c(
        int n, int c, int d, int h, int w, int C, int D, int H, int W) {
    int off = 0;
    off += n * (C / 16) * D * H * W * 16;
    off += (c / 16) * D * H * W * 16;
    off += d * H * W * 16;
    off += h * W * 16;
    off += w * 16;
    off += c % 16;
    return off;
}

int off_NCdhw16n16c(
        int n, int c, int d, int h, int w, int C, int D, int H, int W) {
    int off = 0;
    off += (n / 16) * (C / 16) * D * H * W * 16 * 16;
    off += (c / 16) * D * H * W * 16 * 16;
    off += d * H * W * 16 * 16;
    off += h * W * 16 * 16;
    off += w * 16 * 16;
    off += (n % 16) * 16;
    off += (c % 16);
    return off;
}

int off_nCdhw32c(
        int n, int c, int d, int h, int w, int C, int D, int H, int W) {
    int c_32_block = OC % 32 ? (32 + OC - (OC % 32)) : OC;
    int off = 0;
    off += n * (c_32_block / 32) * G * D * H * W * 32;
    off += (c / 32) * D * H * W * 32;
    off += d * H * W * 32;
    off += h * W * 32;
    off += w * 32;
    off += c % 32;
    return off;
}

int off_NCdhw32n16c(
        int n, int c, int d, int h, int w, int C, int D, int H, int W) {
    int off = 0;
    off += (n / 32) * (C / 16) * D * H * W * 32 * 16;
    off += (c / 16) * D * H * W * 32 * 16;
    off += d * H * W * 32 * 16;
    off += h * W * 32 * 16;
    off += w * 32 * 16;
    off += (n % 32) * 16;
    off += (c % 16);
    return off;
}

int off_NCdhw32n32c(
        int n, int c, int d, int h, int w, int C, int D, int H, int W) {
    int c_32_block = OC % 32 ? (32 + OC - (OC % 32)) : OC;
    int off = 0;
    off += (n / 32) * (c_32_block / 32) * D * H * W * 32 * 32;
    off += (c / 32) * D * H * W * 32 * 32;
    off += d * H * W * 32 * 32;
    off += h * W * 32 * 32;
    off += w * 32 * 32;
    off += (n % 32) * 32;
    off += (c % 32);
    return off;
}

int off_gOdhwi16o(int g, int o, int i, int d, int h, int w, int O, int I, int D,
        int H, int W) {
    int off = 0;
    off += g * (O / 16) * D * H * W * I * 16;
    off += (o / 16) * D * H * W * I * 16;
    off += d * H * W * I * 16;
    off += h * W * I * 16;
    off += w * I * 16;
    off += i * 16;
    off += (o % 16);
    return off;
}

int off_gOIdhw16i16o(int g, int o, int i, int d, int h, int w, int O, int I,
        int D, int H, int W) {
    int off = 0;
    off += g * (O / 16) * (I / 16) * D * H * W * 16 * 16;
    off += (o / 16) * (I / 16) * D * H * W * 16 * 16;
    off += (i / 16) * D * H * W * 16 * 16;
    off += d * H * W * 16 * 16;
    off += h * W * 16 * 16;
    off += w * 16 * 16;
    off += (i % 16) * 16;
    off += (o % 16);
    return off;
}

int off_gIOdhw16i16o(int g, int o, int i, int d, int h, int w, int O, int I,
        int D, int H, int W) {
    int off = 0;
    off += g * (I / 16) * (O / 16) * D * H * W * 16 * 16;
    off += (i / 16) * (O / 16) * D * H * W * 16 * 16;
    off += (o / 16) * D * H * W * 16 * 16;
    off += d * H * W * 16 * 16;
    off += h * W * 16 * 16;
    off += w * 16 * 16;
    off += (i % 16) * 16;
    off += (o % 16);
    return off;
}

int src_off(int n, int c, int d, int h, int w) {
#if SRC_NCHW
    return off_ncdhw(n, c, d, h, w, G * IC, ID, IH, IW);
#elif SRC_NHWC
    return off_ndhwc(n, c, d, h, w, G * IC, ID, IH, IW);
#elif SRC_W16C
    return off_nCdhw16c(n, c, d, h, w, G * IC, ID, IH, IW);
#elif SRC_16N16C
    return off_NCdhw16n16c(n, c, d, h, w, G * IC, ID, IH, IW);
#else
#error "Unknown layout"
#endif
}

int wei_off(int g, int o, int i, int d, int h, int w) {
#if WEI_I16O
    return off_gOdhwi16o(g, o, i, d, h, w, OC, IC, KD, KH, KW);
#elif WEI_16I16O
    return off_gOIdhw16i16o(g, o, i, d, h, w, OC, IC, KD, KH, KW);
#elif WEI_16I16O_FLIPPED
    return off_gIOdhw16i16o(g, o, i, d, h, w, OC, IC, KD, KH, KW);
#else
#error "Unknown layout"
#endif
    return 0;
}

int dst_off(int n, int c, int d, int h, int w) {
#if DST_NCHW
    return off_ncdhw(n, c, d, h, w, G * OC_WO_PADDING, OD, OH, OW);
#elif DST_NHWC
    return off_ndhwc(n, c, d, h, w, G * OC, OD, OH, OW);
#elif DST_W16C
    return off_nCdhw16c(n, c, d, h, w, G * OC, OD, OH, OW);
#elif DST_16N16C
    return off_NCdhw16n16c(n, c, d, h, w, G * OC, OD, OH, OW);
#elif DST_W32C
    return off_nCdhw32c(n, c, d, h, w, G * OC, OD, OH, OW);
#elif DST_32N16C
    return off_NCdhw32n16c(n, c, d, h, w, G * OC, OD, OH, OW);
#elif DST_32N32C
    return off_NCdhw32n32c(n, c, d, h, w, G * OC, OD, OH, OW);
#else
#error "Unknown layout"
#endif
}

#endif // GPU_OCL_OFFSETS_H
