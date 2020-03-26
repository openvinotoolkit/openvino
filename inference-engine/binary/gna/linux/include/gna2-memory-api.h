/*
 @copyright

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing,
 software distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions
 and limitations under the License.

 SPDX-License-Identifier: Apache-2.0
*/

/**************************************************************************//**
 @file gna2-memory-api.h
 @brief Gaussian and Neural Accelerator (GNA) 2.0 API Definition.
 @nosubgrouping

 ******************************************************************************
 @addtogroup GNA2_API
 @{

 ******************************************************************************

 @addtogroup GNA2_API_MEMORY Memory API

 API for managing memory used by GNA Tensors.

 @{
 *****************************************************************************/

#ifndef __GNA2_MEMORY_API_H
#define __GNA2_MEMORY_API_H

#include "gna2-common-api.h"

#include <stdint.h>

/**
 Allocates memory buffer, that can be used with GNA device.

 @param sizeRequested Buffer size desired by the caller.
 @param [out] sizeGranted Buffer size granted by GNA,
                      can be more then requested due to HW constraints.
 @param [out] memoryAddress Address of memory buffer
 @return Status of the operation.
 */
GNA2_API enum Gna2Status Gna2MemoryAlloc(
    uint32_t sizeRequested,
    uint32_t * sizeGranted,
    void ** memoryAddress);

/**
 Releases memory buffer.

 @param memory Memory buffer to be freed.
 @return Status of the operation.
 */
GNA2_API enum Gna2Status Gna2MemoryFree(
    void * memory);

#endif // __GNA2_MEMORY_API_H

/**
 @}
 @}
 */
