/*
 INTEL CONFIDENTIAL
 Copyright 2019 Intel Corporation.

 The source code contained or described herein and all documents related
 to the source code ("Material") are owned by Intel Corporation or its suppliers
 or licensors. Title to the Material remains with Intel Corporation or its suppliers
 and licensors. The Material may contain trade secrets and proprietary
 and confidential information of Intel Corporation and its suppliers and licensors,
 and is protected by worldwide copyright and trade secret laws and treaty provisions.
 No part of the Material may be used, copied, reproduced, modified, published,
 uploaded, posted, transmitted, distributed, or disclosed in any way without Intel's
 prior express written permission.

 No license under any patent, copyright, trade secret or other intellectual
 property right is granted to or conferred upon you by disclosure or delivery
 of the Materials, either expressly, by implication, inducement, estoppel
 or otherwise. Any license under such intellectual property rights must
 be express and approved by Intel in writing.

 Unless otherwise agreed by Intel in writing, you may not remove or alter this notice
 or any other notice embedded in Materials by Intel or Intel's suppliers or licensors
 in any way.
*/

#pragma once

#if defined(__GNUC__)
#define UNREFERENCED_PARAMETER(P) ((void)(P))
#else
#define WIN32_NO_STATUS
#include <windows.h>
#undef WIN32_NO_STATUS
#endif

// Enable safe functions compatibility
#if defined(__STDC_SECURE_LIB__)
#define __STDC_WANT_SECURE_LIB__ 1
#elif defined(__STDC_LIB_EXT1__)
#define STDC_WANT_LIB_EXT1 1
#else
#define memcpy_s(_Destination, _DestinationSize, _Source, _SourceSize) do {\
    memcpy(_Destination, _Source, _SourceSize);\
    UNREFERENCED_PARAMETER(_DestinationSize);\
} while(0);
#define memmove_s(_Destination, _DestinationSize, _Source, _SourceSize) do {\
    memmove(_Destination, _Source, _SourceSize);\
    UNREFERENCED_PARAMETER(_DestinationSize);\
} while(0);
#define strncpy_s(_Destination, _DestinationSize, _Source, _SourceSize) do {\
    strncpy(_Destination, _Source, _SourceSize);\
    UNREFERENCED_PARAMETER(_DestinationSize);\
} while(0);
#endif
