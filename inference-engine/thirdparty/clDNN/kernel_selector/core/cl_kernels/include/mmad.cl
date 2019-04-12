/*
// Copyright (c) 2016 Intel Corporation
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

inline int FUNC(mmad_4)(char4 input, char4 weight, int acc)
{
	acc += (input[0] * weight[0]);
	acc += (input[1] * weight[1]);
	acc += (input[2] * weight[2]);
	acc += (input[3] * weight[3]);
	return acc;
}

inline int FUNC(mmad8)(int8 A_scalars, int8 B_vectors, int acc)
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

inline int4 FUNC(mmad4x8)(int4 A_vectors, int8 B_vectors, int4 acc)
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

inline int8 FUNC(mmad8x8)(int8 A_vectors, int8 B_vectors, int8 acc)
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

// TODO: remove it when cl_intel_subgroups_char extension will work
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

inline uchar8 FUNC(sub_group_block_read_uchar8)(const __global uchar* ptr)
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

//


#define MMAD_8(A, B, C) FUNC_CALL(mmad8)(A, B, C)
#define MMAD_4x8(A, B, C) FUNC_CALL(mmad4x8)(A, B, C)
#define MMAD_8x8(A, B, C) FUNC_CALL(mmad8x8)(A, B, C)
#define SLM_BLOCK_WRITE_4(A, B) (FUNC_CALL(intel_sub_group_block_write_4)(A, B))
#define SLM_BLOCK_READ_4(A) (FUNC_CALL(intel_sub_group_block_read_uint4)(A))
#define SLM_BLOCK_READ_8(A) (FUNC_CALL(intel_sub_group_block_read_uint8)(A))
