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


#define MMAD_8(A, B, C) FUNC_CALL(mmad8)(A, B, C)
#define MMAD_4x8(A, B, C) FUNC_CALL(mmad4x8)(A, B, C)
#define MMAD_8x8(A, B, C) FUNC_CALL(mmad8x8)(A, B, C)
