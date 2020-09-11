/*****************************************************************************\

Copyright 2016 Intel Corporation All Rights Reserved.

The source code contained or described herein and all documents related to
the source code ("Material") are owned by Intel Corporation or its suppliers
or licensors. Title to the Material remains with Intel Corporation or its
suppliers and licensors. The Material contains trade secrets and proprietary
and confidential information of Intel or its suppliers and licensors. The
Material is protected by worldwide copyright and trade secret laws and
treaty provisions. No part of the Material may be used, copied, reproduced,
modified, published, uploaded, posted, transmitted, distributed, or
disclosed in any way without Intel's prior express written permission.

No license under any patent, copyright, trade secret or other intellectual
property right is granted to or conferred upon you by disclosure or delivery
of the Materials, either expressly, by implication, inducement, estoppel or
otherwise. Any license under such intellectual property rights must be
express and approved by Intel in writing.

File Name: cl_intel_planar_yuv.h

Abstract:

Notes:

\*****************************************************************************/
#ifndef __CL_EXT_INTEL_PLANAR_YUV_H
#define __CL_EXT_INTEL_PLANAR_YUV_H

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>

/***************************************
* cl_intel_planar_yuv extension *
****************************************/
#define CL_NV12_INTEL                           0x410E

#define CL_MEM_NO_ACCESS_INTEL                  ( 1 << 24 )
#define CL_MEM_ACCESS_FLAGS_UNRESTRICTED_INTEL  ( 1 << 25 )

#ifdef __cplusplus
}
#endif


#endif /* __CL_EXT_INTEL_PLANAR_YUV_H */
