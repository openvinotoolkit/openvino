/*
// Copyright (c) 2018 Intel Corporation
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

inline int FUNC(imad_SW)(int acc, uchar4 input, char4 weight) __attribute__((overloadable)) {
    acc += input[0] * weight[0];
    acc += input[1] * weight[1];
    acc += input[2] * weight[2];
    acc += input[3] * weight[3];
    return acc;
}

inline int FUNC(imad_SW)(int acc, char4 input, char4 weight) __attribute__((overloadable)) {
    acc += input[0] * weight[0];
    acc += input[1] * weight[1];
    acc += input[2] * weight[2];
    acc += input[3] * weight[3];
    return acc;
}


#define IMAD(_O, _I, _W) FUNC_CALL(imad_SW)(_O, _I, _W)
