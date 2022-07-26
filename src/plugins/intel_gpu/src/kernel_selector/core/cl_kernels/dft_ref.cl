// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define GET_INDEX(prefix, ORDER) CAT(prefix, _GET_INDEX)(ORDER)

// alternative: https://github.com/OpenCL/ComplexMath/blob/master/clcomplex.h
typedef float2 cfloat;
#define real(a)                     ((a).s0)
#define imag(a)                     ((a).s1)
#define cmult(a, b)                 ((cfloat)(real(a) * real(b) - imag(a) * imag(b), real(a) * imag(b) + imag(a) * real(b)))
#define crmult(a, b)                ((cfloat)(real(a) * (b), imag(a) * (b)))
#define cadd(a, b)                  ((cfloat)(real(a) + real(b), imag(a) + imag(b)))
#define expi(x)                     ((cfloat)(cos(x), sin(x)))
#define expmi(x)                    ((cfloat)(cos(x), -sin(x)))
#define cload(p, offset, pitch)     ((cfloat)((p)[offset], (p)[(offset) + (pitch)]))
#define cstore(p, offset, pitch, x) ((p)[offset] = real(x), (p)[(offset) + (pitch)] = imag(x))
#define czero()                     ((cfloat)(0))

// TODO: pregenerate e{r,i} array on host in macro. maybe it could be done with kernel which runs once?
KERNEL(dft_ref)(const __global INPUT0_TYPE* data, __global OUTPUT_TYPE* output) {
    const uint dim0 = get_global_id(0);
    const uint dim1 = get_global_id(1);
    const uint dim2 = get_global_id(2);

    const uint x = 0;
    const uint y = dim0;
#if OUTPUT_DIMS == 4
#    define ORDER   b, f, y, x
#    define ORDER_K kb, kf, ky, x
    const uint f = dim1;
    const uint b = dim2;
#elif OUTPUT_DIMS == 5
#    define ORDER   b, f, z, y, x
#    define ORDER_K kb, kf, kz, ky, x
    const uint z = dim1;
    const uint f = dim2 % OUTPUT_FEATURE_NUM;
    const uint b = dim2 / OUTPUT_FEATURE_NUM;
#elif OUTPUT_DIMS == 6
#    define ORDER   b, f, w, z, y, x
#    define ORDER_K kb, kf, kw, kz, ky, x
    const uint z = dim1 % OUTPUT_SIZE_Z;
    const uint w = dim1 / OUTPUT_SIZE_Z;
    const uint f = dim2 % OUTPUT_FEATURE_NUM;
    const uint b = dim2 / OUTPUT_FEATURE_NUM;
#endif

    // TODO: use OUTPUT_TYPE for intermediate calculations?
    // We don't use it for now as we will lose a lot of precision for f16 and tests won't pass
    cfloat Y = czero();
    const float PI2 = M_PI_F * 2;

#ifdef AXIS_Y
    const float ay = PI2 * y / OUTPUT_SIZE_Y;
#endif
#ifdef AXIS_Z
    const float az = PI2 * z / OUTPUT_SIZE_Z;
#endif
#ifdef AXIS_W
    const float aw = PI2 * w / OUTPUT_SIZE_W;
#endif
#ifdef AXIS_FEATURE
    const float af = PI2 * f / OUTPUT_FEATURE_NUM;
#endif
#ifdef AXIS_BATCH
    const float ab = PI2 * b / OUTPUT_BATCH_NUM;
#endif

#ifdef AXIS_BATCH
    for (uint kb = 0; kb < AXIS_BATCH; ++kb)
#else
#    define kb b
#endif
#ifdef AXIS_FEATURE
        for (uint kf = 0; kf < AXIS_FEATURE; ++kf)
#else
#    define kf f
#endif
#ifdef AXIS_W
            for (uint kw = 0; kw < AXIS_W; ++kw)
#else
#    define kw w
#endif
#ifdef AXIS_Z
                for (uint kz = 0; kz < AXIS_Z; ++kz)
#else
#    define kz z
#endif
#ifdef AXIS_Y
                    for (uint ky = 0; ky < AXIS_Y; ++ky)
#else
#    define ky y
#endif
                    {
                        float a = 0;
#ifdef AXIS_Y
                        a += ay * ky;
#endif
#ifdef AXIS_Z
                        a += az * kz;
#endif
#ifdef AXIS_W
                        a += aw * kw;
#endif
#ifdef AXIS_FEATURE
                        a += af * kf;
#endif
#ifdef AXIS_BATCH
                        a += ab * kb;
#endif
                        const cfloat X = cload(data, GET_INDEX(INPUT0, ORDER_K), INPUT0_X_PITCH);
#ifdef INVERSE_DFT_MULTIPLIER
                        const cfloat E = expi(a);
#else
        const cfloat E = expmi(a);
#endif
                        Y = cadd(Y, cmult(X, E));
                    }

#ifdef INVERSE_DFT_MULTIPLIER
    Y = crmult(Y, INVERSE_DFT_MULTIPLIER);
#endif

    cstore(output, GET_INDEX(OUTPUT, ORDER), OUTPUT_X_PITCH, Y);
}

#undef real
#undef imag
#undef cmult
#undef crmult
#undef cadd
#undef expi
#undef expmi
#undef cload
#undef cstore
#undef czero
#undef GET_INDEX
#undef ORDER
#undef ORDER_K
