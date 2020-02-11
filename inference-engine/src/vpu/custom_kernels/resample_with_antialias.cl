// Copyright (C) 2018-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

static inline float triangleCoeff(float x)
{
    return 1.0f - fabs(x);//fmax(0.0f, 1 - fabs(x));
}

static inline float4 triangleCoeff4(float4 x)
{
    return 1.0f - fabs(x);//fmax(0.0f, 1 - fabs(x));
}

__kernel void resample_with_antialias(const __global half* restrict src,
                                      __global half* restrict dst,
                                      int iw,
                                      int ih,
                                      float factor,
                                      int ow,
                                      int oh,
                                      int channels)
{
    int oy = min((int)get_global_id(0), oh-1);
    int c = get_global_id(1);
    int b = get_global_id(2);

    float fx = 1.f / factor;
    float fy = 1.f / factor;

    float ax = 1.0f / fx;
    float ay = 1.0f / fy;

    int rx = (fx < 1.0f) ? 2 : ceil((1.0f)/ax);
    int ry = (fy < 1.0f) ? 2 : ceil((1.0f)/ay);

    const __global half* restrict start_src = src + b * iw * ih * channels + iw * ih * c;
    __global half* restrict start_dst = dst + b * ow * oh * channels + ow * oh * c;

    float iy_r0 = oy*fy + fy / 2.0f - 0.5f;
    int iy_r1 = (int)(round(iy_r0));

    for (int ox = 0; ox < ow; ox++)
    {
        float ix_r0 = ox*fx + fx / 2.0f - 0.5f;
        int ix_r1 = (int)(round(ix_r0));

        float4 v_sum = 0.f;
        float4 v_wsum = 0.f;

        for (int y = max(iy_r1 - ry, 0);
            y <= min(iy_r1 + ry, (int)ih - 1); y++)
        {
            float dy = iy_r0 - y;
            int x = max(ix_r1 - rx, 0);
            int end_x = min(ix_r1 + rx, (int)iw - 1);

            float4 dx;
            for (int i = 0; i < 4; i++)
                dx[i] = ix_r0 - x - i;

            for (; x <= end_x - 3; x += 4, dx -= 4)
            {
                float4 w = ax*triangleCoeff4(ax*dx) * ay*triangleCoeff(ay*dy);
                float4 src_vec = { start_src[y*iw + x + 0],
                                   start_src[y*iw + x + 1],
                                   start_src[y*iw + x + 2],
                                   start_src[y*iw + x + 3] };

                v_sum += w * src_vec;
                v_wsum += w;
            }

            for (; x <= end_x; x++)
            {
                float dx = ix_r0 - x;
                float w = ax*triangleCoeff(ax*dx) * ay*triangleCoeff(ay*dy);

                v_sum[0] += w * start_src[y*iw + x];
                v_wsum[0] += w;
            }
        }

        v_sum[0] = v_sum[0] + v_sum[1] + v_sum[2] + v_sum[3];
        v_wsum[0] = v_wsum[0] + v_wsum[1] + v_wsum[2] + v_wsum[3];

        start_dst[oy*ow + ox] = (!v_wsum[0]) ? (half)0.0f : (half)(v_sum[0] / v_wsum[0]);
    }
}
