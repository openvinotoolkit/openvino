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

// TODO: currently we calculate on float32 because it's lot of "add" operation and it stuck on the value "8192.0f"
#if !defined(ACCUMULATOR_TYPE)
    #define ACCUMULATOR_TYPE float
    #define TO_ACCUMULATOR_TYPE(v) (float)(v)
    #define ACCUMULATOR_TYPE_ZERO 0.0f
#endif

// Creates vector type.
#define MAKE_VECTOR_TYPE(elem_type, size) CAT(elem_type, size)