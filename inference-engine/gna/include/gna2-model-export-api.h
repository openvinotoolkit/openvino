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
 @file gna2-model-export-api.h
 @brief Gaussian and Neural Accelerator (GNA) 2.0 API Definition.
 @nosubgrouping

 ******************************************************************************

 @addtogroup GNA2_API
 @{
 ******************************************************************************

 @addtogroup GNA2_MODEL_API
 @{
 *****************************************************************************

 @addtogroup GNA2_MODEL_EXPORT_API Model Export API

 API for exporting GNA model for embedded devices.

 @{
 *****************************************************************************/

#ifndef __GNA2_MODEL_EXPORT_API_H
#define __GNA2_MODEL_EXPORT_API_H

#include "gna2-common-api.h"

#if !defined(_WIN32)
#include <assert.h>
#endif
#include <stdint.h>

/**
 Creates configuration for model exporting.

 Export configuration allows to configure all the parameters necessary
 to export components of one or more models.
 Use Gna2ModelExportConfigSet*() functions to configure parameters. Parameters
 can be modified/overridden for existing configuration to export model
 with modified properties.

 @warning
    User is responsible for releasing allocated memory buffers.

 @param userAllocator User provided memory allocator.
 @param [out] exportConfigId Identifier of created export configuration.
 @return Status of the operation.
 */
GNA2_API enum Gna2Status Gna2ModelExportConfigCreate(
    Gna2UserAllocator userAllocator,
    uint32_t * exportConfigId);

/**
 Releases export configuration and all its resources.

 @param exportConfigId Identifier of export configuration to release.
 @return Status of the operation.
 */
GNA2_API enum Gna2Status Gna2ModelExportConfigRelease(
    uint32_t exportConfigId);

/**
 Sets source model(s) to export.

 - Model will be validated against provided device.
 - Model(s) should be created through standard API Gna2ModelCreate() function.

 @param exportConfigId Identifier of export configuration to set.
 @param sourceDeviceIndex Id of the device on which the exported model was created.
    Use GNA2_DISABLED to export model from all available devices at one.
 @param sourceModelId Id of the source model, created previously with Gna2ModelCreate() function.
     Use GNA2_DISABLED to export all models from given device at one.
 @return Status of the operation.
 */
GNA2_API enum Gna2Status Gna2ModelExportConfigSetSource(
    uint32_t exportConfigId,
    uint32_t sourceDeviceIndex,
    uint32_t sourceModelId);

/**
 Sets version of the device that exported model will be used with.

 - Model will be validated against provided target device.

 @param exportConfigId Identifier of export configuration to set.
 @param targetDeviceVersion Device on which model will be used.
 @return Status of the operation.
 */
GNA2_API enum Gna2Status Gna2ModelExportConfigSetTarget(
    uint32_t exportConfigId,
    enum Gna2DeviceVersion targetDeviceVersion);

/**
 Determines the type of the component to export.
 */
enum Gna2ModelExportComponent
{
    /**
     Hardware layer descriptors will be exported.
     */
    Gna2ModelExportComponentLayerDescriptors = GNA2_DEFAULT,

    /**
     Header describing layer descriptors will be exported.
     */
    Gna2ModelExportComponentLayerDescriptorHeader = 1,

    /**
     Hardware layer descriptors and model data in legacy SueCreek format will be exported.
     */
    Gna2ModelExportComponentLegacySueCreekDump = 2,

    /**
     Header describing layer descriptors in legacy SueCreek format will be exported.
     */
    Gna2ModelExportComponentLegacySueCreekHeader = 3,
};

/**
 Exports the model(s) component.

 All exported model components are saved into memory allocated on user side by userAllocator.

 @warning
    User is responsible for releasing allocated memory buffers (exportBuffer).

 @param exportConfigId Identifier of export configuration used.
 @param componentType What component should be exported.
 @param [out] exportBuffer Memory allocated by userAllocator with exported layer descriptors.
 @param [out] exportBufferSize The size of exportBuffer in bytes.
 @return Status of the operation.
 */
GNA2_API enum Gna2Status Gna2ModelExport(
    uint32_t exportConfigId,
    enum Gna2ModelExportComponent componentType,
    void ** exportBuffer,
    uint32_t * exportBufferSize);

#endif // __GNA2_MODEL_EXPORT_API_H

/**
 @}
 @}
 @}
 */
