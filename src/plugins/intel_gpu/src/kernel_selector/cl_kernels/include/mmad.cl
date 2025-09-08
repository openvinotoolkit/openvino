// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

inline int FUNC(mmad_4)(char4 input, char4 weight, int acc) __attribute__((overloadable))
{
    acc += (input[0] * weight[0]);
    acc += (input[1] * weight[1]);
    acc += (input[2] * weight[2]);
    acc += (input[3] * weight[3]);
    return acc;
}

inline int FUNC(mmad_4)(char4 input, uchar4 weight, int acc) __attribute__((overloadable))
{
    acc += (input[0] * weight[0]);
    acc += (input[1] * weight[1]);
    acc += (input[2] * weight[2]);
    acc += (input[3] * weight[3]);
    return acc;
}

inline int FUNC(mmad_4)(uchar4 input, char4 weight, int acc) __attribute__((overloadable))
{
    acc += (input[0] * weight[0]);
    acc += (input[1] * weight[1]);
    acc += (input[2] * weight[2]);
    acc += (input[3] * weight[3]);
    return acc;
}

inline int FUNC(mmad_4)(uchar4 input, uchar4 weight, int acc) __attribute__((overloadable))
{
    acc += (input[0] * weight[0]);
    acc += (input[1] * weight[1]);
    acc += (input[2] * weight[2]);
    acc += (input[3] * weight[3]);
    return acc;
}

inline int FUNC(mmad8)(int8 A_scalars, int8 B_vectors, int acc) __attribute__((overloadable))
{
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[0]), as_char4(B_vectors[0]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[1]), as_char4(B_vectors[1]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[2]), as_char4(B_vectors[2]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[3]), as_char4(B_vectors[3]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[4]), as_char4(B_vectors[4]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[5]), as_char4(B_vectors[5]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[6]), as_char4(B_vectors[6]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[7]), as_char4(B_vectors[7]), acc);

    return acc;
}

inline int FUNC(mmad8)(int8 A_scalars, uint8 B_vectors, int acc) __attribute__((overloadable))
{
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[0]), as_uchar4(B_vectors[0]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[1]), as_uchar4(B_vectors[1]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[2]), as_uchar4(B_vectors[2]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[3]), as_uchar4(B_vectors[3]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[4]), as_uchar4(B_vectors[4]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[5]), as_uchar4(B_vectors[5]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[6]), as_uchar4(B_vectors[6]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[7]), as_uchar4(B_vectors[7]), acc);

    return acc;
}

inline int FUNC(mmad8)(uint8 A_scalars, int8 B_vectors, int acc) __attribute__((overloadable))
{
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[0]), as_char4(B_vectors[0]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[1]), as_char4(B_vectors[1]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[2]), as_char4(B_vectors[2]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[3]), as_char4(B_vectors[3]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[4]), as_char4(B_vectors[4]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[5]), as_char4(B_vectors[5]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[6]), as_char4(B_vectors[6]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[7]), as_char4(B_vectors[7]), acc);

    return acc;
}

inline int FUNC(mmad8)(uint8 A_scalars, uint8 B_vectors, int acc) __attribute__((overloadable))
{
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[0]), as_uchar4(B_vectors[0]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[1]), as_uchar4(B_vectors[1]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[2]), as_uchar4(B_vectors[2]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[3]), as_uchar4(B_vectors[3]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[4]), as_uchar4(B_vectors[4]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[5]), as_uchar4(B_vectors[5]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[6]), as_uchar4(B_vectors[6]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[7]), as_uchar4(B_vectors[7]), acc);

    return acc;
}

inline int FUNC(mmad16)(int16 A_scalars, int16 B_vectors, int acc) __attribute__((overloadable))
{
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[0]), as_char4(B_vectors[0]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[1]), as_char4(B_vectors[1]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[2]), as_char4(B_vectors[2]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[3]), as_char4(B_vectors[3]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[4]), as_char4(B_vectors[4]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[5]), as_char4(B_vectors[5]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[6]), as_char4(B_vectors[6]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[7]), as_char4(B_vectors[7]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[8]), as_char4(B_vectors[8]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[9]), as_char4(B_vectors[9]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[10]), as_char4(B_vectors[10]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[11]), as_char4(B_vectors[11]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[12]), as_char4(B_vectors[12]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[13]), as_char4(B_vectors[13]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[14]), as_char4(B_vectors[14]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[15]), as_char4(B_vectors[15]), acc);

    return acc;
}

inline int FUNC(mmad16)(int16 A_scalars, uint16 B_vectors, int acc) __attribute__((overloadable))
{
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[0]), as_uchar4(B_vectors[0]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[1]), as_uchar4(B_vectors[1]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[2]), as_uchar4(B_vectors[2]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[3]), as_uchar4(B_vectors[3]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[4]), as_uchar4(B_vectors[4]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[5]), as_uchar4(B_vectors[5]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[6]), as_uchar4(B_vectors[6]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[7]), as_uchar4(B_vectors[7]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[8]), as_uchar4(B_vectors[8]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[9]), as_uchar4(B_vectors[9]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[10]), as_uchar4(B_vectors[10]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[11]), as_uchar4(B_vectors[11]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[12]), as_uchar4(B_vectors[12]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[13]), as_uchar4(B_vectors[13]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[14]), as_uchar4(B_vectors[14]), acc);
    acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[15]), as_uchar4(B_vectors[15]), acc);

    return acc;
}

inline int FUNC(mmad16)(uint16 A_scalars, int16 B_vectors, int acc) __attribute__((overloadable))
{
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[0]), as_char4(B_vectors[0]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[1]), as_char4(B_vectors[1]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[2]), as_char4(B_vectors[2]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[3]), as_char4(B_vectors[3]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[4]), as_char4(B_vectors[4]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[5]), as_char4(B_vectors[5]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[6]), as_char4(B_vectors[6]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[7]), as_char4(B_vectors[7]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[8]), as_char4(B_vectors[8]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[9]), as_char4(B_vectors[9]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[10]), as_char4(B_vectors[10]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[11]), as_char4(B_vectors[11]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[12]), as_char4(B_vectors[12]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[13]), as_char4(B_vectors[13]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[14]), as_char4(B_vectors[14]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[15]), as_char4(B_vectors[15]), acc);

    return acc;
}

inline int FUNC(mmad16)(uint16 A_scalars, uint16 B_vectors, int acc) __attribute__((overloadable))
{
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[0]), as_uchar4(B_vectors[0]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[1]), as_uchar4(B_vectors[1]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[2]), as_uchar4(B_vectors[2]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[3]), as_uchar4(B_vectors[3]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[4]), as_uchar4(B_vectors[4]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[5]), as_uchar4(B_vectors[5]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[6]), as_uchar4(B_vectors[6]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[7]), as_uchar4(B_vectors[7]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[8]), as_uchar4(B_vectors[8]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[9]), as_uchar4(B_vectors[9]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[10]), as_uchar4(B_vectors[10]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[11]), as_uchar4(B_vectors[11]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[12]), as_uchar4(B_vectors[12]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[13]), as_uchar4(B_vectors[13]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[14]), as_uchar4(B_vectors[14]), acc);
    acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[15]), as_uchar4(B_vectors[15]), acc);

    return acc;
}

inline int4 FUNC(mmad4x8)(int4 A_vectors, int8 B_vectors, int4 acc) __attribute__((overloadable))
{
    int4 ret;
    for(uint i = 0; i < 4; i++)
    {
        int8 A_scalars;
        A_scalars.s0 = sub_group_broadcast(A_vectors[i], 0);
        A_scalars.s1 = sub_group_broadcast(A_vectors[i], 1);
        A_scalars.s2 = sub_group_broadcast(A_vectors[i], 2);
        A_scalars.s3 = sub_group_broadcast(A_vectors[i], 3);
        A_scalars.s4 = sub_group_broadcast(A_vectors[i], 4);
        A_scalars.s5 = sub_group_broadcast(A_vectors[i], 5);
        A_scalars.s6 = sub_group_broadcast(A_vectors[i], 6);
        A_scalars.s7 = sub_group_broadcast(A_vectors[i], 7);
        ret[i] = FUNC_CALL(mmad8)(A_scalars, B_vectors, acc[i]);
    }
    return ret;
}

inline int4 FUNC(mmad4x8)(int4 A_vectors, uint8 B_vectors, int4 acc) __attribute__((overloadable))
{
    int4 ret;
    for(uint i = 0; i < 4; i++)
    {
        int8 A_scalars;
        A_scalars.s0 = sub_group_broadcast(A_vectors[i], 0);
        A_scalars.s1 = sub_group_broadcast(A_vectors[i], 1);
        A_scalars.s2 = sub_group_broadcast(A_vectors[i], 2);
        A_scalars.s3 = sub_group_broadcast(A_vectors[i], 3);
        A_scalars.s4 = sub_group_broadcast(A_vectors[i], 4);
        A_scalars.s5 = sub_group_broadcast(A_vectors[i], 5);
        A_scalars.s6 = sub_group_broadcast(A_vectors[i], 6);
        A_scalars.s7 = sub_group_broadcast(A_vectors[i], 7);
        ret[i] = FUNC_CALL(mmad8)(A_scalars, B_vectors, acc[i]);
    }
    return ret;
}

inline int4 FUNC(mmad4x8)(uint4 A_vectors, int8 B_vectors, int4 acc) __attribute__((overloadable))
{
    int4 ret;
    for(uint i = 0; i < 4; i++)
    {
        uint8 A_scalars;
        A_scalars.s0 = sub_group_broadcast(A_vectors[i], 0);
        A_scalars.s1 = sub_group_broadcast(A_vectors[i], 1);
        A_scalars.s2 = sub_group_broadcast(A_vectors[i], 2);
        A_scalars.s3 = sub_group_broadcast(A_vectors[i], 3);
        A_scalars.s4 = sub_group_broadcast(A_vectors[i], 4);
        A_scalars.s5 = sub_group_broadcast(A_vectors[i], 5);
        A_scalars.s6 = sub_group_broadcast(A_vectors[i], 6);
        A_scalars.s7 = sub_group_broadcast(A_vectors[i], 7);
        ret[i] = FUNC_CALL(mmad8)(A_scalars, B_vectors, acc[i]);
    }
    return ret;
}

inline int4 FUNC(mmad4x8)(uint4 A_vectors, uint8 B_vectors, int4 acc) __attribute__((overloadable))
{
    int4 ret;
    for(uint i = 0; i < 4; i++)
    {
        uint8 A_scalars;
        A_scalars.s0 = sub_group_broadcast(A_vectors[i], 0);
        A_scalars.s1 = sub_group_broadcast(A_vectors[i], 1);
        A_scalars.s2 = sub_group_broadcast(A_vectors[i], 2);
        A_scalars.s3 = sub_group_broadcast(A_vectors[i], 3);
        A_scalars.s4 = sub_group_broadcast(A_vectors[i], 4);
        A_scalars.s5 = sub_group_broadcast(A_vectors[i], 5);
        A_scalars.s6 = sub_group_broadcast(A_vectors[i], 6);
        A_scalars.s7 = sub_group_broadcast(A_vectors[i], 7);
        ret[i] = FUNC_CALL(mmad8)(A_scalars, B_vectors, acc[i]);
    }
    return ret;
}

inline int8 FUNC(mmad8x8)(int8 A_vectors, int8 B_vectors, int8 acc) __attribute__((overloadable))
{
    int8 ret;
    for(uint i = 0; i < 8; i++)
    {
        int8 A_scalars;
        A_scalars.s0 = sub_group_broadcast(A_vectors[i], 0);
        A_scalars.s1 = sub_group_broadcast(A_vectors[i], 1);
        A_scalars.s2 = sub_group_broadcast(A_vectors[i], 2);
        A_scalars.s3 = sub_group_broadcast(A_vectors[i], 3);
        A_scalars.s4 = sub_group_broadcast(A_vectors[i], 4);
        A_scalars.s5 = sub_group_broadcast(A_vectors[i], 5);
        A_scalars.s6 = sub_group_broadcast(A_vectors[i], 6);
        A_scalars.s7 = sub_group_broadcast(A_vectors[i], 7);
        ret[i] = FUNC_CALL(mmad8)(A_scalars, B_vectors, acc[i]);
    }
    return ret;
}

inline int8 FUNC(mmad8x8)(int8 A_vectors, uint8 B_vectors, int8 acc) __attribute__((overloadable))
{
    int8 ret;
    for(uint i = 0; i < 8; i++)
    {
        int8 A_scalars;
        A_scalars.s0 = sub_group_broadcast(A_vectors[i], 0);
        A_scalars.s1 = sub_group_broadcast(A_vectors[i], 1);
        A_scalars.s2 = sub_group_broadcast(A_vectors[i], 2);
        A_scalars.s3 = sub_group_broadcast(A_vectors[i], 3);
        A_scalars.s4 = sub_group_broadcast(A_vectors[i], 4);
        A_scalars.s5 = sub_group_broadcast(A_vectors[i], 5);
        A_scalars.s6 = sub_group_broadcast(A_vectors[i], 6);
        A_scalars.s7 = sub_group_broadcast(A_vectors[i], 7);
        ret[i] = FUNC_CALL(mmad8)(A_scalars, B_vectors, acc[i]);
    }
    return ret;
}

inline int8 FUNC(mmad8x8)(uint8 A_vectors, int8 B_vectors, int8 acc) __attribute__((overloadable))
{
    int8 ret;
    for(uint i = 0; i < 8; i++)
    {
        uint8 A_scalars;
        A_scalars.s0 = sub_group_broadcast(A_vectors[i], 0);
        A_scalars.s1 = sub_group_broadcast(A_vectors[i], 1);
        A_scalars.s2 = sub_group_broadcast(A_vectors[i], 2);
        A_scalars.s3 = sub_group_broadcast(A_vectors[i], 3);
        A_scalars.s4 = sub_group_broadcast(A_vectors[i], 4);
        A_scalars.s5 = sub_group_broadcast(A_vectors[i], 5);
        A_scalars.s6 = sub_group_broadcast(A_vectors[i], 6);
        A_scalars.s7 = sub_group_broadcast(A_vectors[i], 7);
        ret[i] = FUNC_CALL(mmad8)(A_scalars, B_vectors, acc[i]);
    }
    return ret;
}

inline int8 FUNC(mmad8x8)(uint8 A_vectors, uint8 B_vectors, int8 acc) __attribute__((overloadable))
{
    int8 ret;
    for(uint i = 0; i < 8; i++)
    {
        uint8 A_scalars;
        A_scalars.s0 = sub_group_broadcast(A_vectors[i], 0);
        A_scalars.s1 = sub_group_broadcast(A_vectors[i], 1);
        A_scalars.s2 = sub_group_broadcast(A_vectors[i], 2);
        A_scalars.s3 = sub_group_broadcast(A_vectors[i], 3);
        A_scalars.s4 = sub_group_broadcast(A_vectors[i], 4);
        A_scalars.s5 = sub_group_broadcast(A_vectors[i], 5);
        A_scalars.s6 = sub_group_broadcast(A_vectors[i], 6);
        A_scalars.s7 = sub_group_broadcast(A_vectors[i], 7);
        ret[i] = FUNC_CALL(mmad8)(A_scalars, B_vectors, acc[i]);
    }
    return ret;
}

inline int16 FUNC(mmad16x16)(int16 A_vectors, int16 B_vectors, int16 acc) __attribute__((overloadable))
{
    int16 ret;
    for(uint i = 0; i < 16; i++)
    {
        int16 A_scalars;
        A_scalars.s0 = sub_group_broadcast(A_vectors[i], 0);
        A_scalars.s1 = sub_group_broadcast(A_vectors[i], 1);
        A_scalars.s2 = sub_group_broadcast(A_vectors[i], 2);
        A_scalars.s3 = sub_group_broadcast(A_vectors[i], 3);
        A_scalars.s4 = sub_group_broadcast(A_vectors[i], 4);
        A_scalars.s5 = sub_group_broadcast(A_vectors[i], 5);
        A_scalars.s6 = sub_group_broadcast(A_vectors[i], 6);
        A_scalars.s7 = sub_group_broadcast(A_vectors[i], 7);
        A_scalars.s8 = sub_group_broadcast(A_vectors[i], 8);
        A_scalars.s9 = sub_group_broadcast(A_vectors[i], 9);
        A_scalars.sa = sub_group_broadcast(A_vectors[i], 10);
        A_scalars.sb = sub_group_broadcast(A_vectors[i], 11);
        A_scalars.sc = sub_group_broadcast(A_vectors[i], 12);
        A_scalars.sd = sub_group_broadcast(A_vectors[i], 13);
        A_scalars.se = sub_group_broadcast(A_vectors[i], 14);
        A_scalars.sf = sub_group_broadcast(A_vectors[i], 15);
        ret[i] = FUNC_CALL(mmad16)(A_scalars, B_vectors, acc[i]);
    }
    return ret;
}

inline int16 FUNC(mmad16x16)(int16 A_vectors, uint16 B_vectors, int16 acc) __attribute__((overloadable))
{
    int16 ret;
    for(uint i = 0; i < 16; i++)
    {
        int16 A_scalars;
        A_scalars.s0 = sub_group_broadcast(A_vectors[i], 0);
        A_scalars.s1 = sub_group_broadcast(A_vectors[i], 1);
        A_scalars.s2 = sub_group_broadcast(A_vectors[i], 2);
        A_scalars.s3 = sub_group_broadcast(A_vectors[i], 3);
        A_scalars.s4 = sub_group_broadcast(A_vectors[i], 4);
        A_scalars.s5 = sub_group_broadcast(A_vectors[i], 5);
        A_scalars.s6 = sub_group_broadcast(A_vectors[i], 6);
        A_scalars.s7 = sub_group_broadcast(A_vectors[i], 7);
        A_scalars.s8 = sub_group_broadcast(A_vectors[i], 8);
        A_scalars.s9 = sub_group_broadcast(A_vectors[i], 9);
        A_scalars.sa = sub_group_broadcast(A_vectors[i], 10);
        A_scalars.sb = sub_group_broadcast(A_vectors[i], 11);
        A_scalars.sc = sub_group_broadcast(A_vectors[i], 12);
        A_scalars.sd = sub_group_broadcast(A_vectors[i], 13);
        A_scalars.se = sub_group_broadcast(A_vectors[i], 14);
        A_scalars.sf = sub_group_broadcast(A_vectors[i], 15);
        ret[i] = FUNC_CALL(mmad16)(A_scalars, B_vectors, acc[i]);
    }
    return ret;
}

inline int16 FUNC(mmad16x16)(uint16 A_vectors, int16 B_vectors, int16 acc) __attribute__((overloadable))
{
    int16 ret;
    for(uint i = 0; i < 16; i++)
    {
        uint16 A_scalars;
        A_scalars.s0 = sub_group_broadcast(A_vectors[i], 0);
        A_scalars.s1 = sub_group_broadcast(A_vectors[i], 1);
        A_scalars.s2 = sub_group_broadcast(A_vectors[i], 2);
        A_scalars.s3 = sub_group_broadcast(A_vectors[i], 3);
        A_scalars.s4 = sub_group_broadcast(A_vectors[i], 4);
        A_scalars.s5 = sub_group_broadcast(A_vectors[i], 5);
        A_scalars.s6 = sub_group_broadcast(A_vectors[i], 6);
        A_scalars.s7 = sub_group_broadcast(A_vectors[i], 7);
        A_scalars.s8 = sub_group_broadcast(A_vectors[i], 8);
        A_scalars.s9 = sub_group_broadcast(A_vectors[i], 9);
        A_scalars.sa = sub_group_broadcast(A_vectors[i], 10);
        A_scalars.sb = sub_group_broadcast(A_vectors[i], 11);
        A_scalars.sc = sub_group_broadcast(A_vectors[i], 12);
        A_scalars.sd = sub_group_broadcast(A_vectors[i], 13);
        A_scalars.se = sub_group_broadcast(A_vectors[i], 14);
        A_scalars.sf = sub_group_broadcast(A_vectors[i], 15);
        ret[i] = FUNC_CALL(mmad16)(A_scalars, B_vectors, acc[i]);
    }
    return ret;
}

inline int16 FUNC(mmad16x16)(uint16 A_vectors, uint16 B_vectors, int16 acc) __attribute__((overloadable))
{
    int16 ret;
    for(uint i = 0; i < 16; i++)
    {
        uint16 A_scalars;
        A_scalars.s0 = sub_group_broadcast(A_vectors[i], 0);
        A_scalars.s1 = sub_group_broadcast(A_vectors[i], 1);
        A_scalars.s2 = sub_group_broadcast(A_vectors[i], 2);
        A_scalars.s3 = sub_group_broadcast(A_vectors[i], 3);
        A_scalars.s4 = sub_group_broadcast(A_vectors[i], 4);
        A_scalars.s5 = sub_group_broadcast(A_vectors[i], 5);
        A_scalars.s6 = sub_group_broadcast(A_vectors[i], 6);
        A_scalars.s7 = sub_group_broadcast(A_vectors[i], 7);
        A_scalars.s8 = sub_group_broadcast(A_vectors[i], 8);
        A_scalars.s9 = sub_group_broadcast(A_vectors[i], 9);
        A_scalars.sa = sub_group_broadcast(A_vectors[i], 10);
        A_scalars.sb = sub_group_broadcast(A_vectors[i], 11);
        A_scalars.sc = sub_group_broadcast(A_vectors[i], 12);
        A_scalars.sd = sub_group_broadcast(A_vectors[i], 13);
        A_scalars.se = sub_group_broadcast(A_vectors[i], 14);
        A_scalars.sf = sub_group_broadcast(A_vectors[i], 15);
        ret[i] = FUNC_CALL(mmad16)(A_scalars, B_vectors, acc[i]);
    }
    return ret;
}

#define MMAD_8(A, B, C) FUNC_CALL(mmad8)(A, B, C)
#define MMAD_16(A, B, C) FUNC_CALL(mmad16)(A, B, C)
#define MMAD_4x8(A, B, C) FUNC_CALL(mmad4x8)(A, B, C)
#define MMAD_8x8(A, B, C) FUNC_CALL(mmad8x8)(A, B, C)
#define MMAD_16x16(A, B, C) FUNC_CALL(mmad16x16)(A, B, C)
