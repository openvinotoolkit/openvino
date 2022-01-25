// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// alternative: https://github.com/OpenCL/ComplexMath/blob/master/clcomplex.h
typedef float2 cfloat;
#define real(a) ((a).s0)
#define imag(a) ((a).s1)
#define cmult(a, b) ((cfloat)(real(a) * real(b) - imag(a) * imag(b), real(a) * imag(b) + imag(a) * real(b)))
#define cadd(a, b) ((cfloat)(real(a) + real(b), imag(a) + imag(b)))
#define expi(x) ((cfloat)(cos(x), sin(x)))
#define expmi(x) ((cfloat)(cos(x), -sin(x)))
#define cload(p, offset, pitch) ((cfloat)((p)[offset], (p)[(offset)+(pitch)]))
#define cstore(p, offset, pitch, x) ((p)[offset] = real(x), (p)[(offset)+(pitch)] = imag(x))
#define czero() ((cfloat)(0))

// TODO: pregenerate e{r,i} array on host in macro. maybe it could be done with kernel which runs once?
KERNEL (dft_ref)(const __global INPUT0_TYPE *data, __global OUTPUT_TYPE *output)
{
    size_t g2 = get_global_id(2);
    ptrdiff_t y1 = g2 % OUTPUT_SIZES[1];
#if OUTPUT_DIMS > 2
    g2 /= OUTPUT_SIZES[1];
    ptrdiff_t y2 = g2 % OUTPUT_SIZES[2];
#if OUTPUT_DIMS > 3
    g2 /= OUTPUT_SIZES[2];
    ptrdiff_t y3 = g2 % OUTPUT_SIZES[3];
#if OUTPUT_DIMS > 4
    g2 /= OUTPUT_SIZES[3];
    ptrdiff_t y4 = g2 /*% OUTPUT_SIZES[4]*/;
#endif // OUTPUT_DIMS > 4
#endif // OUTPUT_DIMS > 3
#endif // OUTPUT_DIMS > 2
    cfloat y = czero(); // TODO: use OUTPUT_TYPE for intermediate calculations?
    const float pi2 = M_PI_F * 2;
    ptrdiff_t inputOffset = INPUT0_OFFSET;
#ifdef A1
    const float a1 = pi2 * y1 / OUTPUT_SIZES[1];
#else
    const ptrdiff_t x1 = y1;
    inputOffset += INPUT0_PITCHES[1] * x1;
#endif
#if OUTPUT_DIMS > 2
#ifdef A2
    const float a2 = pi2 * y2 / OUTPUT_SIZES[2];
#else
    const ptrdiff_t x2 = y2;
    inputOffset += INPUT0_PITCHES[2] * x2;
#endif
#if OUTPUT_DIMS > 3
#ifdef A3
    const float a3 = pi2 * y3 / OUTPUT_SIZES[3];
#else
    const ptrdiff_t x3 = y3;
    inputOffset += INPUT0_PITCHES[3] * x3;
#endif
#if OUTPUT_DIMS > 4
#ifdef A4
    const float a4 = pi2 * y4 / OUTPUT_SIZES[4];
#else
    const ptrdiff_t x4 = y4;
    inputOffset += INPUT0_PITCHES[4] * x4;
#endif
#ifdef A4
    for (ptrdiff_t x4 = 0; x4 < A4; x4 ++)
#endif
#endif // OUTPUT_DIMS > 4
#endif // OUTPUT_DIMS > 3
#endif // OUTPUT_DIMS > 2
    { // TODO: repeat upto maximum number of dimensions - 1(last one is complex pairs)
        const ptrdiff_t saveOffset = inputOffset;
#ifdef A3
        for (ptrdiff_t x3 = 0; x3 < A3; x3 ++)
#endif
        {
            const ptrdiff_t saveOffset = inputOffset;
#ifdef A2
            for (ptrdiff_t x2 = 0; x2 < A2; x2 ++)
#endif
            {
                const ptrdiff_t saveOffset = inputOffset;
#ifdef A1
                for (ptrdiff_t x1 = 0; x1 < A1; x1 ++)
#endif
                {
                    float a = 0;
#ifdef A1
                    a += a1 * x1;
#endif
#ifdef A2
                    a += a2 * x2;
#endif
#ifdef A3
                    a += a3 * x3;
#endif
#ifdef A4
                    a += a4 * x4;
#endif
                    const cfloat x = cload(data, inputOffset, INPUT0_PITCHES[0]);
                    const cfloat e = expmi(a);
                    y = cadd(y, cmult(x, e));
                    inputOffset += INPUT0_PITCHES[1];
                }
                inputOffset = saveOffset + INPUT0_PITCHES[2];
            }
            inputOffset = saveOffset + INPUT0_PITCHES[3];
        }
        inputOffset = saveOffset + INPUT0_PITCHES[4];
    }
    ptrdiff_t outputOffset = OUTPUT_OFFSET + OUTPUT_PITCHES[1] * y1;
#if OUTPUT_DIMS > 2
    outputOffset += OUTPUT_PITCHES[2] * y2;
#if OUTPUT_DIMS > 3
    outputOffset += OUTPUT_PITCHES[3] * y3;
#if OUTPUT_DIMS > 4
    outputOffset += OUTPUT_PITCHES[4] * y4;
#endif // OUTPUT_DIMS > 4
#endif // OUTPUT_DIMS > 3
#endif // OUTPUT_DIMS > 2
    cstore(output, outputOffset, OUTPUT_PITCHES[0], y);
}

#undef real
#undef imag
#undef cmult
#undef cadd
#undef expi
#undef expmi
#undef cload
#undef cstore
#undef czero
