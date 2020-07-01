/*
// Copyright (c) 2016-2020 Intel Corporation
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
*/

void FUNC(intel_sub_group_block_write_4)( __local uint* p, uint4 data )
{
    p[ get_sub_group_local_id() ] = data.s0;
    p += 8;
    p[ get_sub_group_local_id() ] = data.s1;
    p += 8;
    p[ get_sub_group_local_id() ] = data.s2;
    p += 8;
    p[ get_sub_group_local_id() ] = data.s3;
}

uint4 FUNC(intel_sub_group_block_read_uint4)(const __local uint* p)
{
    uint4 ret;
    uint idx = get_sub_group_local_id();

    ret.s0 = p[idx]; idx += get_max_sub_group_size();
    ret.s1 = p[idx]; idx += get_max_sub_group_size();
    ret.s2 = p[idx]; idx += get_max_sub_group_size();
    ret.s3 = p[idx]; idx += get_max_sub_group_size();

    return ret;
}

uint8 FUNC(intel_sub_group_block_read_uint8)(const __local uint* p)
{
    uint8 ret;
    uint idx = get_sub_group_local_id();

    ret.s0 = p[idx]; idx += get_max_sub_group_size();
    ret.s1 = p[idx]; idx += get_max_sub_group_size();
    ret.s2 = p[idx]; idx += get_max_sub_group_size();
    ret.s3 = p[idx]; idx += get_max_sub_group_size();
    ret.s4 = p[idx]; idx += get_max_sub_group_size();
    ret.s5 = p[idx]; idx += get_max_sub_group_size();
    ret.s6 = p[idx]; idx += get_max_sub_group_size();
    ret.s7 = p[idx]; idx += get_max_sub_group_size();

    return ret;
}

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

inline void FUNC(sub_group_block_write_uchar16)(__global uchar* outPtr, uchar16 v)
{
#ifdef cl_intel_subgroups_char
    intel_sub_group_block_write_uc16(outPtr, v);
#else
    uint idx = get_sub_group_local_id();

    outPtr[idx] = v.s0; idx += get_max_sub_group_size();
    outPtr[idx] = v.s1; idx += get_max_sub_group_size();
    outPtr[idx] = v.s2; idx += get_max_sub_group_size();
    outPtr[idx] = v.s3; idx += get_max_sub_group_size();
    outPtr[idx] = v.s4; idx += get_max_sub_group_size();
    outPtr[idx] = v.s5; idx += get_max_sub_group_size();
    outPtr[idx] = v.s6; idx += get_max_sub_group_size();
    outPtr[idx] = v.s7; idx += get_max_sub_group_size();
    outPtr[idx] = v.s8; idx += get_max_sub_group_size();
    outPtr[idx] = v.s9; idx += get_max_sub_group_size();
    outPtr[idx] = v.sa; idx += get_max_sub_group_size();
    outPtr[idx] = v.sb; idx += get_max_sub_group_size();
    outPtr[idx] = v.sc; idx += get_max_sub_group_size();
    outPtr[idx] = v.sd; idx += get_max_sub_group_size();
    outPtr[idx] = v.se; idx += get_max_sub_group_size();
    outPtr[idx] = v.sf; idx += get_max_sub_group_size();
#endif
}

inline uchar16 FUNC(sub_group_block_read_uchar16)(const __global uchar* ptr) __attribute__((overloadable))
{
#ifdef cl_intel_subgroups_char
    // WA for compiler support
    // return intel_sub_group_block_read_uc16(ptr);
    return (uchar16)(intel_sub_group_block_read_uc8(ptr), intel_sub_group_block_read_uc8(ptr + 8 * get_max_sub_group_size()));
#else
    uint idx = get_sub_group_local_id();

    uchar16 ret;

    ret.s0 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s1 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s2 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s3 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s4 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s5 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s6 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s7 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s8 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s9 = ptr[idx]; idx += get_max_sub_group_size();
    ret.sa = ptr[idx]; idx += get_max_sub_group_size();
    ret.sb = ptr[idx]; idx += get_max_sub_group_size();
    ret.sc = ptr[idx]; idx += get_max_sub_group_size();
    ret.sd = ptr[idx]; idx += get_max_sub_group_size();
    ret.se = ptr[idx]; idx += get_max_sub_group_size();
    ret.sf = ptr[idx]; idx += get_max_sub_group_size();

    return ret;
#endif
}

inline uchar16 FUNC(sub_group_block_read_uchar16)(const __local uchar* ptr) __attribute__((overloadable))
{
#if defined(cl_intel_subgroup_local_block_io) && defined(cl_intel_subgroups_char)
    // WA for compiler support
    // return intel_sub_group_block_read_uc16(ptr);
    return (uchar16)(intel_sub_group_block_read_uc8(ptr), intel_sub_group_block_read_uc8(ptr + 8 * get_max_sub_group_size()));
#else
    uint idx = get_sub_group_local_id();

    uchar16 ret;

    ret.s0 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s1 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s2 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s3 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s4 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s5 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s6 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s7 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s8 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s9 = ptr[idx]; idx += get_max_sub_group_size();
    ret.sa = ptr[idx]; idx += get_max_sub_group_size();
    ret.sb = ptr[idx]; idx += get_max_sub_group_size();
    ret.sc = ptr[idx]; idx += get_max_sub_group_size();
    ret.sd = ptr[idx]; idx += get_max_sub_group_size();
    ret.se = ptr[idx]; idx += get_max_sub_group_size();
    ret.sf = ptr[idx]; idx += get_max_sub_group_size();

    return ret;
#endif
}

inline void FUNC(sub_group_block_write_uchar8)(__global uchar* outPtr, uchar8 v)
{
#ifdef cl_intel_subgroups_char
    intel_sub_group_block_write_uc8(outPtr, v);
#else
    uint idx = get_sub_group_local_id();

    outPtr[idx] = v.s0; idx += get_max_sub_group_size();
    outPtr[idx] = v.s1; idx += get_max_sub_group_size();
    outPtr[idx] = v.s2; idx += get_max_sub_group_size();
    outPtr[idx] = v.s3; idx += get_max_sub_group_size();
    outPtr[idx] = v.s4; idx += get_max_sub_group_size();
    outPtr[idx] = v.s5; idx += get_max_sub_group_size();
    outPtr[idx] = v.s6; idx += get_max_sub_group_size();
    outPtr[idx] = v.s7; idx += get_max_sub_group_size();
#endif
}

inline uchar8 FUNC(sub_group_block_read_uchar8)(const __global uchar* ptr) __attribute__((overloadable))
{
#ifdef cl_intel_subgroups_char
    return intel_sub_group_block_read_uc8(ptr);
#else
    uint idx = get_sub_group_local_id();

    uchar8 ret;

    ret.s0 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s1 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s2 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s3 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s4 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s5 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s6 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s7 = ptr[idx]; idx += get_max_sub_group_size();

    return ret;
#endif
}

inline uchar8 FUNC(sub_group_block_read_uchar8)(const __local uchar* ptr) __attribute__((overloadable))
{
#if defined(cl_intel_subgroup_local_block_io) && defined(cl_intel_subgroups_char)
    return intel_sub_group_block_read_uc8(ptr);
#else
    uint idx = get_sub_group_local_id();

    uchar8 ret;

    ret.s0 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s1 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s2 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s3 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s4 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s5 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s6 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s7 = ptr[idx]; idx += get_max_sub_group_size();

    return ret;
#endif
}

inline void FUNC(sub_group_block_write_uchar4)(__global uchar* outPtr, uchar4 v)
{
#ifdef cl_intel_subgroups_char
    intel_sub_group_block_write_uc4(outPtr, v);
#else
    uint idx = get_sub_group_local_id();

    outPtr[idx] = v.s0; idx += get_max_sub_group_size();
    outPtr[idx] = v.s1; idx += get_max_sub_group_size();
    outPtr[idx] = v.s2; idx += get_max_sub_group_size();
    outPtr[idx] = v.s3; idx += get_max_sub_group_size();
#endif
}

inline uchar4 FUNC(sub_group_block_read_uchar4)(const __global uchar* ptr) __attribute__((overloadable))
{
#ifdef cl_intel_subgroups_char
    return intel_sub_group_block_read_uc4(ptr);
#else
    uint idx = get_sub_group_local_id();

    uchar4 ret;

    ret.s0 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s1 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s2 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s3 = ptr[idx]; idx += get_max_sub_group_size();

    return ret;
#endif
}

inline uchar4 FUNC(sub_group_block_read_uchar4)(const __local uchar* ptr) __attribute__((overloadable))
{
#if defined(cl_intel_subgroup_local_block_io) && defined(cl_intel_subgroups_char)
    return intel_sub_group_block_read_uc4(ptr);
#else
    uint idx = get_sub_group_local_id();

    uchar4 ret;

    ret.s0 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s1 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s2 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s3 = ptr[idx]; idx += get_max_sub_group_size();

    return ret;
#endif
}

inline void FUNC(sub_group_block_write_uchar2)(__global uchar* outPtr, uchar2 v)
{
#ifdef cl_intel_subgroups_char
    intel_sub_group_block_write_uc2(outPtr, v);
#else
    uint idx = get_sub_group_local_id();

    outPtr[idx] = v.s0; idx += get_max_sub_group_size();
    outPtr[idx] = v.s1; idx += get_max_sub_group_size();
#endif
}

inline uchar2 FUNC(sub_group_block_read_uchar2)(const __global uchar* ptr) __attribute__((overloadable))
{
#ifdef cl_intel_subgroups_char
    return intel_sub_group_block_read_uc2(ptr);
#else
    uint idx = get_sub_group_local_id();

    uchar2 ret;

    ret.s0 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s1 = ptr[idx]; idx += get_max_sub_group_size();

    return ret;
#endif
}

inline uchar2 FUNC(sub_group_block_read_uchar2)(const __local uchar* ptr) __attribute__((overloadable))
{
#if defined(cl_intel_subgroup_local_block_io) && defined(cl_intel_subgroups_char)
    return intel_sub_group_block_read_uc2(ptr);
#else
    uint idx = get_sub_group_local_id();

    uchar2 ret;

    ret.s0 = ptr[idx]; idx += get_max_sub_group_size();
    ret.s1 = ptr[idx]; idx += get_max_sub_group_size();

    return ret;
#endif
}

inline void FUNC(sub_group_block_write_uchar)(__global uchar* outPtr, uchar v)
{
#ifdef cl_intel_subgroups_char
    intel_sub_group_block_write_uc(outPtr, v);
#else
    uint idx = get_sub_group_local_id();

    outPtr[idx] = v;
#endif
}

inline uchar FUNC(sub_group_block_read_uchar)(const __global uchar* ptr) __attribute__((overloadable))
{
#ifdef cl_intel_subgroups_char
    return intel_sub_group_block_read_uc(ptr);
#else
    uint idx = get_sub_group_local_id();

    uchar ret;

    ret = ptr[idx];

    return ret;
#endif
}

inline uchar FUNC(sub_group_block_read_uchar)(const __local uchar* ptr) __attribute__((overloadable))
{
#if defined(cl_intel_subgroup_local_block_io) && defined(cl_intel_subgroups_char)
    return intel_sub_group_block_read_uc(ptr);
#else
    uint idx = get_sub_group_local_id();

    uchar ret;

    ret = ptr[idx];

    return ret;
#endif
}

#define MMAD_8(A, B, C) FUNC_CALL(mmad8)(A, B, C)
#define MMAD_4x8(A, B, C) FUNC_CALL(mmad4x8)(A, B, C)
#define MMAD_8x8(A, B, C) FUNC_CALL(mmad8x8)(A, B, C)
#define MMAD_16x16(A, B, C) FUNC_CALL(mmad16x16)(A, B, C)
#define SLM_BLOCK_WRITE_4(A, B) (FUNC_CALL(intel_sub_group_block_write_4)(A, B))
#define SLM_BLOCK_READ_4(A) (FUNC_CALL(intel_sub_group_block_read_uint4)(A))
#define SLM_BLOCK_READ_8(A) (FUNC_CALL(intel_sub_group_block_read_uint8)(A))

#define BLOCK_READ_UC_1(ptr)  FUNC_CALL(sub_group_block_read_uchar)(ptr)
#define BLOCK_READ_UC_2(ptr)  FUNC_CALL(sub_group_block_read_uchar2)(ptr)
#define BLOCK_READ_UC_4(ptr)  FUNC_CALL(sub_group_block_read_uchar4)(ptr)
#define BLOCK_READ_UC_8(ptr)  FUNC_CALL(sub_group_block_read_uchar8)(ptr)
#define BLOCK_READ_UC_16(ptr) FUNC_CALL(sub_group_block_read_uchar16)(ptr)

#define BLOCK_WRITE_UC_1(ptr, val)  FUNC_CALL(sub_group_block_write_uchar)(ptr, val)
#define BLOCK_WRITE_UC_2(ptr, val)  FUNC_CALL(sub_group_block_write_uchar2)(ptr, val)
#define BLOCK_WRITE_UC_4(ptr, val)  FUNC_CALL(sub_group_block_write_uchar4)(ptr, val)
#define BLOCK_WRITE_UC_8(ptr, val)  FUNC_CALL(sub_group_block_write_uchar8)(ptr, val)
#define BLOCK_WRITE_UC_16(ptr, val) FUNC_CALL(sub_group_block_write_uchar16)(ptr, val)
