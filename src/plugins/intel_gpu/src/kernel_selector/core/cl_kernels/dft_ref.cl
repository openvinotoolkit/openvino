// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define GET_INDEX(prefix, ORDER) CAT(prefix, _GET_INDEX)(ORDER)

// alternative: https://github.com/OpenCL/ComplexMath/blob/master/clcomplex.h
typedef float2 cfloat;
#define real(a)      ((a).s0)
#define imag(a)      ((a).s1)
#define cmult(a, b)  ((cfloat)(real(a) * real(b) - imag(a) * imag(b), real(a) * imag(b) + imag(a) * real(b)))
#define crmult(a, b) ((cfloat)(real(a) * (b), imag(a) * (b)))
#define cadd(a, b)   ((cfloat)(real(a) + real(b), imag(a) + imag(b)))
#define expi(x)      ((cfloat)(cos(x), sin(x)))
#define expmi(x)     ((cfloat)(cos(x), -sin(x)))
#define conj(x)      ((cfloat)(real(x), -imag(x)))
#define czero()      ((cfloat)(0))

// TODO: pregenerate e{r,i} array on host in macro. maybe it could be done with kernel which runs once?
KERNEL(dft_ref)(const __global INPUT0_TYPE* data, __global OUTPUT_TYPE* output) {
    const uint dim0 = get_global_id(0);
    const uint dim1 = get_global_id(1);
    const uint dim2 = get_global_id(2);

    uint y = dim0;
#if OUTPUT_DIMS == 4
#    define ORDER_REAL b, f, y, 0
#    define ORDER_IMAG b, f, y, 1
    uint f = dim1;
    uint b = dim2;
#elif OUTPUT_DIMS == 5
#    define ORDER_REAL b, f, z, y, 0
#    define ORDER_IMAG b, f, z, y, 1
    uint z = dim1;
    uint f = dim2 % OUTPUT_FEATURE_NUM;
    uint b = dim2 / OUTPUT_FEATURE_NUM;
#elif OUTPUT_DIMS == 6
#    define ORDER_REAL b, f, w, z, y, 0
#    define ORDER_IMAG b, f, w, z, y, 1
    uint z = dim1 % OUTPUT_SIZE_Z;
    uint w = dim1 / OUTPUT_SIZE_Z;
    uint f = dim2 % OUTPUT_FEATURE_NUM;
    uint b = dim2 / OUTPUT_FEATURE_NUM;
#endif

    const uint output_real_index = GET_INDEX(OUTPUT, ORDER_REAL);
#if !defined(REAL_DFT) || !defined(INVERSE_DFT_MULTIPLIER)
    const uint output_imag_index = GET_INDEX(OUTPUT, ORDER_IMAG);
#endif

    // TODO: use OUTPUT_TYPE for intermediate calculations?
    // We don't use it for now as we will lose a lot of precision for f16 and tests won't pass
#if defined(REAL_DFT) && defined(INVERSE_DFT_MULTIPLIER)
    float Y = 0;
#else
    cfloat Y = czero();
#endif
    const float PI2 = M_PI_F * 2;

#ifdef AXIS_Y
    const float ay = PI2 * y / SIGNAL_SIZE_Y;
#endif
#ifdef AXIS_Z
    const float az = PI2 * z / SIGNAL_SIZE_Z;
#endif
#ifdef AXIS_W
    const float aw = PI2 * w / SIGNAL_SIZE_W;
#endif
#ifdef AXIS_FEATURE
    const float af = PI2 * f / SIGNAL_SIZE_FEATURE;
#endif
#ifdef AXIS_BATCH
    const float ab = PI2 * b / SIGNAL_SIZE_BATCH;
#endif

#ifdef AXIS_BATCH
    for (b = 0; b < AXIS_BATCH; ++b)
#endif
#ifdef AXIS_FEATURE
        for (f = 0; f < AXIS_FEATURE; ++f)
#endif
#ifdef AXIS_W
            for (w = 0; w < AXIS_W; ++w)
#endif
#ifdef AXIS_Z
                for (z = 0; z < AXIS_Z; ++z)
#endif
#ifdef AXIS_Y
                    for (y = 0; y < AXIS_Y; ++y)
#endif
                    {
                        float a = 0;
#ifdef AXIS_Y
                        a += ay * y;
#endif
#ifdef AXIS_Z
                        a += az * z;
#endif
#ifdef AXIS_W
                        a += aw * w;
#endif
#ifdef AXIS_FEATURE
                        a += af * f;
#endif
#ifdef AXIS_BATCH
                        a += ab * b;
#endif

#ifdef REAL_DFT
#    ifdef INVERSE_DFT_MULTIPLIER
#        if OUTPUT_DIMS == 4
#            define SYMMETRIC_ORDER_REAL sb, sf, sy, 0
#            define SYMMETRIC_ORDER_IMAG sb, sf, sy, 1
#        elif OUTPUT_DIMS == 5
#            define SYMMETRIC_ORDER_REAL sb, sf, sz, sy, 0
#            define SYMMETRIC_ORDER_IMAG sb, sf, sz, sy, 1
#        elif OUTPUT_DIMS == 6
#            define SYMMETRIC_ORDER_REAL sb, sf, sw, sz, sy, 0
#            define SYMMETRIC_ORDER_IMAG sb, sf, sw, sz, sy, 1
#        endif
                        bool is_zero = false;
                        bool is_conj = false;
#        ifdef SYMMETRIC_AXIS_BATCH
                        uint sb = b;
                        if (sb > OUTPUT_BATCH_NUM / 2) {
                            sb = OUTPUT_BATCH_NUM - sb;
                            is_conj = true;
                        }
                        if (sb >= INPUT0_BATCH_NUM) {
                            is_zero = true;
                        }
#        else
#            define sb b
#        endif
#        ifdef SYMMETRIC_AXIS_FEATURE
                        uint sf = f;
                        if (sf > OUTPUT_FEATURE_NUM / 2) {
                            sf = OUTPUT_FEATURE_NUM - sf;
                            is_conj = true;
                        }
                        if (sf >= INPUT0_FEATURE_NUM) {
                            is_zero = true;
                        }
#        else
#            define sf f
#        endif
#        ifdef SYMMETRIC_AXIS_W
                        uint sw = w;
                        if (sw > OUTPUT_SIZE_W / 2) {
                            sw = OUTPUT_SIZE_W - sw;
                            is_conj = true;
                        }
                        if (sw >= INPUT0_SIZE_W) {
                            is_zero = true;
                        }
#        else
#            define sw w
#        endif
#        ifdef SYMMETRIC_AXIS_Z
                        uint sz = z;
                        if (sz > OUTPUT_SIZE_Z / 2) {
                            sz = OUTPUT_SIZE_Z - sz;
                            is_conj = true;
                        }
                        if (sz >= INPUT0_SIZE_Z) {
                            is_zero = true;
                        }
#        else
#            define sz z
#        endif
#        ifdef SYMMETRIC_AXIS_Y
                        uint sy = y;
                        if (sy > OUTPUT_SIZE_Y / 2) {
                            sy = OUTPUT_SIZE_Y - sy;
                            is_conj = true;
                        }
                        if (sy >= INPUT0_SIZE_Y) {
                            is_zero = true;
                        }
#        else
#            define sy y
#        endif
                        cfloat X = czero();
                        if (!is_zero) {
                            const uint input_real_index = GET_INDEX(INPUT0, SYMMETRIC_ORDER_REAL);
                            const uint input_imag_index = GET_INDEX(INPUT0, SYMMETRIC_ORDER_IMAG);
                            X = (cfloat)(data[input_real_index], data[input_imag_index]);
                            if (is_conj) {
                                X = conj(X);
                            }
                        }
#    else
                        const uint input_real_index = GET_INDEX(INPUT0, ORDER_REAL);
                        const float X = data[input_real_index];
#    endif
// clang-format off
#else
                        const uint input_real_index = GET_INDEX(INPUT0, ORDER_REAL);
                        const uint input_imag_index = GET_INDEX(INPUT0, ORDER_IMAG);
                        const cfloat X = (cfloat)(data[input_real_index], data[input_imag_index]);
#endif

#ifdef INVERSE_DFT_MULTIPLIER
// No need to calculate E for IRDFT case, as we will calculate manually later
#    ifndef REAL_DFT
                        const cfloat E = expi(a);
#    endif
#else
                        const cfloat E = expmi(a);
#endif

#ifdef REAL_DFT
#    ifdef INVERSE_DFT_MULTIPLIER
                        Y += real(X) * cos(a) - imag(X) * sin(a);
#    else
                        Y = cadd(Y, crmult(E, X));
#    endif
#else
                        Y = cadd(Y, cmult(X, E));
#endif
                    }
// clang-format on
#ifdef INVERSE_DFT_MULTIPLIER
#    ifdef REAL_DFT
    Y *= INVERSE_DFT_MULTIPLIER;
#    else
    Y = crmult(Y, INVERSE_DFT_MULTIPLIER);
#    endif
#endif

#if defined(REAL_DFT) && defined(INVERSE_DFT_MULTIPLIER)
    output[output_real_index] = Y;
#else
    output[output_real_index] = real(Y);
    output[output_imag_index] = imag(Y);
#endif
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
#undef ORDER_REAL
#undef ORDER_IMAG
