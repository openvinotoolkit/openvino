// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/* Index and Value type that holds index and value used in this kernel */

#ifndef IAV_STRUCT_DEFINED
    typedef struct 
    {
        uint index; 
        INPUT0_TYPE value; 
    } iav_type;
    #define IAV_STRUCT_DEFINED
#endif
