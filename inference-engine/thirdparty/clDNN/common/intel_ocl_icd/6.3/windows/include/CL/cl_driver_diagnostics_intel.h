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