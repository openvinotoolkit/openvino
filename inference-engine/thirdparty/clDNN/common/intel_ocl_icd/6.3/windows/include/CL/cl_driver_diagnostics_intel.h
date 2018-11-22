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

File Name: cl_driver_diagnostics_intel.h

Abstract:

Notes:

\*****************************************************************************/
#ifndef __CL_DRIVER_DIAGNOSTICS_INTEL_H
#define __CL_DRIVER_DIAGNOSTICS_INTEL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>
#include <CL/cl_ext.h>

/****************************************
* cl_intel_driver_diagnostics extension *
*****************************************/
#define cl_intel_driver_diagnostics 1

typedef cl_bitfield                                 cl_diagnostics_verbose_level_intel;

#define CL_CONTEXT_SHOW_DIAGNOSTICS_INTEL           0x4106
#define CL_CONTEXT_DIAGNOSTICS_LEVEL_GOOD_INTEL     0x1
#define CL_CONTEXT_DIAGNOSTICS_LEVEL_BAD_INTEL      0x2
#define CL_CONTEXT_DIAGNOSTICS_LEVEL_NEUTRAL_INTEL  0x4

#ifdef __cplusplus
}
#endif

#endif /* __CL_DRIVER_DIAGNOSTICS_INTEL_H */