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

typedef struct half1  { half s0; }                                                               half1;
typedef struct half5  { half s0; half s1; half s2; half s3; half s4; }                           half5;
typedef struct half6  { half s0; half s1; half s2; half s3; half s4; half s5; }                  half6;
typedef struct half7  { half s0; half s1; half s2; half s3; half s4; half s5; half s6; }         half7;
typedef struct half9  { half s0; half s1; half s2; half s3; half s4; half s5; half s6; half s7; 
                        half s8; }                                                               half9;
typedef struct half10 { half s0; half s1; half s2; half s3; half s4; half s5; half s6; half s7; 
                        half s8; half s9; }                                                      half10;
typedef struct half11 { half s0; half s1; half s2; half s3; half s4; half s5; half s6; half s7; 
                        half s8; half s9; half sa; }                                             half11;
typedef struct half12 { half s0; half s1; half s2; half s3; half s4; half s5; half s6; half s7; 
                        half s8;  half s9; half sa; half sb;}                                    half12;
typedef struct half13 { half s0; half s1; half s2; half s3; half s4; half s5; half s6; half s7; 
                        half s8;  half s9; half sa; half sb; half sc;}                           half13;
typedef struct half14 { half s0; half s1; half s2; half s3; half s4; half s5; half s6; half s7; 
                        half s8;  half s9; half sa; half sb; half sc; half se;}                  half14;
typedef struct half15 { half s0; half s1; half s2; half s3; half s4; half s5; half s6; half s7; 
                       half s8;  half s9; half sa; half sb; half sc; half se; half sf;}          half15;
typedef struct half0  { half s0; } half0; //never used but makes compiler happy.

typedef struct float1 { float s0; } float1;
typedef struct float5 { float s0; float s1; float s2; float s3; float s4; } float5;
typedef struct float6 { float s0; float s1; float s2; float s3; float s4; float s5; } float6;
typedef struct float7 { float s0; float s1; float s2; float s3; float s4; float s5; float s6; } float7;
typedef struct float9 { float s0; float s1; float s2; float s3; float s4; float s5; float s6; float s7; float s8; } float9;
typedef struct float10 { float s0; float s1; float s2; float s3; float s4; float s5;
                         float s6; float s7; float s8; float s9;} float10;
typedef struct float11 { float s0; float s1; float s2; float s3; float s4; float s5;
                         float s6; float s7; float s8; float s9; float sa;} float11;
typedef struct float12 { float s0; float s1; float s2; float s3; float s4; float s5;
                         float s6; float s7; float s8; float s9; float sa; float sb; } float12;
typedef struct float13 { float s0; float s1; float s2; float s3; float s4; float s5;
                         float s6; float s7; float s8; float s9; float sa; float sb; float sc;} float13;
typedef struct float14 { float s0; float s1; float s2; float s3; float s4; float s5;
                         float s6; float s7; float s8; float s9; float sa; float sb; float sc; float sd; } float14;
typedef struct float15 { float s0; float s1; float s2; float s3; float s4; float s5;
                         float s6; float s7; float s8; float s9; float sa; float sb; float sc; float sd; float se; } float15;
typedef struct float0 { float s0; } float0; //never used but makes compiler happy.

#if (KERNEL_WIDTH == 1)
__constant half1 half_zeros= (half1){0};
#elif (KERNEL_WIDTH == 2)
    __constant half2 half_zeros = (half2)(0);
#elif (KERNEL_WIDTH == 3)
    __constant half3 half_zeros = (half3)(0);
#elif (KERNEL_WIDTH == 4)
    __constant half4 half_zeros = (half4)(0);
#elif (KERNEL_WIDTH == 5)
    __constant half5 half_zeros = (half5){0, 0, 0, 0, 0};
#elif (KERNEL_WIDTH == 6)
    __constant half6 half_zeros = (half6){0, 0, 0, 0, 0, 0};
#elif (KERNEL_WIDTH == 7)
    __constant half7 half_zeros = (half7){0, 0, 0, 0, 0, 0, 0};
#elif (KERNEL_WIDTH == 8)
    __constant half8 half_zeros = (half8)(0);
#elif (KERNEL_WIDTH == 9)
    __constant half9 half_zeros = (half9){0, 0, 0, 0, 0, 0, 0, 0, 0};
#elif (KERNEL_WIDTH == 10)
    __constant half10 half_zeros = (half10){0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#elif (KERNEL_WIDTH == 11)
    __constant half11 half_zeros = (half11){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#elif (KERNEL_WIDTH == 12)
    __constant half12 half_zeros = (half12){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#elif (KERNEL_WIDTH == 13)
    __constant half13 half_zeros = (half13){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#elif (KERNEL_WIDTH == 14)
    __constant half14 half_zeros = (half14){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#elif (KERNEL_WIDTH == 15)
    __constant half15 half_zeros = (half15){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#elif (KERNEL_WIDTH == 16)
    __constant half16 half_zeros = (half16)(0);
#endif
