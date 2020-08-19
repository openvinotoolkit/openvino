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
 @file gna2-model-suecreek-header.h
 @brief Gaussian and Neural Accelerator (GNA) 1.0 SueCreek Header.
 @nosubgrouping

 ******************************************************************************

 @addtogroup GNA1_API Gaussian and Neural Accelerator (GNA) 1.0 API.
 @{

 ******************************************************************************

 @addtogroup GNA1_MODEL_EXPORT_API Embedded Model Export for SueCreek.

 @{
 *****************************************************************************/

#ifndef __GNA2_MODEL_SUECREEK_HEADER_H
#define __GNA2_MODEL_SUECREEK_HEADER_H

#if !defined(_WIN32)
#include <assert.h>
#endif
#include <stdint.h>

/**
 Header describing parameters of dumped model.

 Structure is partially filled by Gna2ModelExport() with parameters necessary for SueCreek,
 other fields are populated by user as necessary.
 */
struct Gna2ModelSueCreekHeader
{
    /**
     Offset in bytes of first layer descriptor in network.
     */
    uint32_t LayerDescriptorBaseOffset;

    /**
     Total size of model in bytes determined by Gna2ModelExport().
     Including hardware descriptors, model data and input/output buffers.
     */
    uint32_t ModelSize;

    /**
     Mode of GNA operation.
     + 1 = XNN mode (default),
     + 0 = GMM mode.
     */
    uint32_t OperationMode;

    /**
     Number of layers in model.
     */
    uint32_t NumberOfLayers;

    /**
     Network Input resolution in bytes.
     */
    uint32_t InputElementSize;

    /**
     Network Output resolution in bytes.
     */
    uint32_t OutputElementSize;

    /**
     Number of network input nodes.
     */
    uint32_t NumberOfInputNodes;

    /**
     Number of network output nodes.
     */
    uint32_t NumberOfOutputNodes;

    /**
     Offset in bytes of input pointer descriptor field.
     Need to be set for processing.
     */
    uint32_t InputDescriptorOffset;

    /**
     Offset in bytes of output pointer descriptor field.
     Need to be set for processing.
     */
    uint32_t OutputDescriptorOffset;

    /**
     Size in bytes of read-write region of statically linked GNA model.
     */
    uint32_t RwRegionSize;

    /**
     Scaling factor used for quantization of input values.
     */
    float InputScalingFactor;

    /**
     Scaling factor used for quantization of output values.
     */
    float OutputScalingFactor;

    /**
     Padding to 64B.
     */
    uint8_t Reserved[12];
};

static_assert(64 == sizeof(struct Gna2ModelSueCreekHeader), "Invalid size of struct Gna2ModelSueCreekHeader");

#endif // __GNA2_MODEL_SUECREEK_HEADER_H

/**
 @}
 @}
 */
