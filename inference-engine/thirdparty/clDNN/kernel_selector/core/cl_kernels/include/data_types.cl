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

#if defined(cl_khr_fp16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

// TODO: currently we calculate on float32 because it's lot of "add" operation and it stuck on the value "8192.0f"
#if !defined(ACCUMULATOR_TYPE)
    #define ACCUMULATOR_TYPE float
#endif
    

#if FP16_UNIT_USED
    #ifndef UNIT_TYPE
    #define UNIT_TYPE half
    #endif

    #define UNIT_VAL_MAX HALF_MAX
    #define UNIT_VAL_MIN -UNIT_VAL_MAX
    #define UNIT_VAL_ONE 1.0h
    #define UNIT_VAL_ZERO 0.0h
    #define TO_UNIT_TYPE(v) convert_half(v)
    #define UNIT_MAX_FUNC fmax
    #define UNIT_MIN_FUNC fmin

#elif INT8_UNIT_USED
    #ifndef UNIT_TYPE
    #define UNIT_TYPE char
    #endif

    #define UNIT_VAL_MAX CHAR_MAX
    #define UNIT_VAL_MIN -UNIT_VAL_MAX
    #define UNIT_VAL_ONE (char) 1
    #define UNIT_VAL_ZERO (char) 0
    #define TO_UNIT_TYPE(v) convert_char(v)
    #define UNIT_MAX_FUNC max
    #define UNIT_MIN_FUNC min

#else
    #ifndef UNIT_TYPE
    #define UNIT_TYPE float
    #endif
    
    #define UNIT_VAL_MAX FLT_MAX
    #define UNIT_VAL_MIN -UNIT_VAL_MAX
    #define UNIT_VAL_ONE 1.0f
    #define UNIT_VAL_ZERO 0.0f
    #define TO_UNIT_TYPE(v) (float)(v)
    #define UNIT_MAX_FUNC fmax
    #define UNIT_MIN_FUNC fmin
#endif

// Creates vector type.
#define MAKE_VECTOR_TYPE(elem_type, size) CAT(elem_type, size)