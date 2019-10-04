// Copyright (C) 2019 Intel Corporation
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

    for (int ox = 0; ox < ow; ox++)
    {
        float ix_r0 = ox*fx + fx / 2.0f - 0.5f;
        float iy_r0 = oy*fy + fy / 2.0f - 0.5f;
        int ix_r1 = (int)(round(ix_r0));
        int iy_r1 = (int)(round(iy_r0));

        float wsum = 0.f;
        float sum = 0.f;

        for (int y = iy_r1 - ry; y <= iy_r1 + ry; y++)
        {
            for (int x = ix_r1 - rx; x <= ix_r1 + rx; x++)
            {
                if (y < 0 || x < 0) continue;
                if (y >= (int)ih || x >= (int)iw) continue;

                float dx = ix_r0 - x;
                float dy = iy_r0 - y;

                float w = ax*triangleCoeff(ax*dx) * ay*triangleCoeff(ay*dy);

                sum += w * start_src[y*iw + x];
                wsum += w;
            }
        }

        start_dst[oy*ow + ox] = (!wsum) ? (half)0.0f : (half)(sum / wsum);
    }
}
