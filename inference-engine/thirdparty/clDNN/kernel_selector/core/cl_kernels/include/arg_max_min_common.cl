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

/* Index and Value type that holds index and value used in this kernel */

#ifndef IAV_STRUCT_DEFINED
    typedef struct 
    {
        uint index; 
        INPUT0_TYPE value; 
    } iav_type;
    #define IAV_STRUCT_DEFINED
#endif