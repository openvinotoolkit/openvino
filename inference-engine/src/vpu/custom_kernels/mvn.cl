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

void atomic_add_global(volatile __global float *source, const float operand)
{
    union
    {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union
    {
        unsigned int intVal;
        float floatVal;
    } prevVal;

    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

__kernel void reduction_mean(const __global half *  restrict src,
                                   __global float * mean,
                                   __global float * variance,
                                 int W,
                                 int H,
                                 int across_channels)
{
    int h = get_global_id(0);

    int c = get_global_id(1);
    int C = get_global_size(1);

    const int MAX_LOCAL_SIZE = 8;
    __local float mbuf[MAX_LOCAL_SIZE];
    __local float vbuf[MAX_LOCAL_SIZE];

    mbuf[get_local_id(0)] = 0;
    vbuf[get_local_id(0)] = 0;

    if (h < H)
    {
        float sum  = 0.f;
        float sum2 = 0.f;

        float8 sum4  = (float8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float8 sum24 = (float8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

        const __global half8* src_line = (const __global half8 *)(src + c*H*W + h*W);

        #pragma unroll 16
        for (size_t w = 0; w < W/8; w++)
        {
            half8 sh = src_line[w];
            float8 valf = convert_float8(sh);

            sum4 += valf;
            sum24 += valf*valf;
        }

        for (size_t w = W/8*8; w < W; w++)
        {
            float val = (float)src[c*H*W + h*W + w];

            sum  += val;
            sum2 += val*val;
        }

        mbuf[get_local_id(0)] = sum4.s0  + sum4.s1  + sum4.s2  + sum4.s3  + sum4.s4  + sum4.s5  + sum4.s6  + sum4.s7  + sum;
        vbuf[get_local_id(0)] = sum24.s0 + sum24.s1 + sum24.s2 + sum24.s3 + sum24.s4 + sum24.s5 + sum24.s6 + sum24.s7 + sum2;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_local_id(0) == 0)
    {
        float res  = mbuf[0]+mbuf[1]+mbuf[2]+mbuf[3]+mbuf[4]+mbuf[5]+mbuf[6]+mbuf[7];
        float res2 = vbuf[0]+vbuf[1]+vbuf[2]+vbuf[3]+vbuf[4]+vbuf[5]+vbuf[6]+vbuf[7];

// requires memory reset before layer execution
#if USE_ATOMIC
        int idx = (across_channels == 0) ? c : 0;

        atomic_add_global(mean + idx, res);
        atomic_add_global(variance + idx, res2);
#else
        int idx = c*get_num_groups(0) + get_group_id(0);

        mean[idx] = res;
        variance[idx] = res2;
#endif
    }
}

__kernel void mvn_scale(const __global half * restrict src_data,
                              __global float * mean_part,
                              __global float * power_mean,
                              __global half * restrict dst_data,
                        int W,
                        int H,
                        int across_channels,
                        int normalize_variance)
{
    int h = get_global_id(0);
    if (h >= H) return;

    int c = get_global_id(1);
    int C = get_global_size(1);

    int nparts = get_num_groups(0);

    int idx     = (across_channels == 0) ? nparts*c : 0;
    float scale = (across_channels == 0) ?      H*W : H*W*C;
    int total   = (across_channels == 0) ?   nparts : nparts*C;

    float mean = 0.f;
    float variance = 0.f;

    for (int i = 0; i < total; i++)
    {
        mean     += mean_part[idx+i];
        variance += power_mean[idx+i];
    }

    mean = mean/scale;
    variance = variance/scale;
    variance = variance - mean*mean;
    variance = native_sqrt(variance) + 1e-9f;

    half hmean = mean;
    half hvariance = (normalize_variance == 0) ? 1.f : (1.f / variance);

    const __global half8 * restrict src_data8 = (const __global half8 * restrict)(src_data + c*H*W + h*W);
    __global half8 * restrict dst_data8 = (__global half8 * restrict)(dst_data + c*H*W + h*W);

    #pragma unroll 16
    for (size_t w = 0; w < W/8; w++)
    {
        dst_data8[w] = (src_data8[w] - hmean) * hvariance;
    }
    for (size_t w = W/8*8; w < W; w++)
    {
        dst_data[c*H*W + h*W + w] = (src_data[c*H*W + h*W + w] - hmean) * hvariance;
    }
}
