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
 @file gna2-capability-api.h
 @brief Gaussian and Neural Accelerator (GNA) 2.0 API Definition.
 @nosubgrouping

 ******************************************************************************

 @addtogroup GNA2_API
 @{
 ******************************************************************************

 @addtogroup GNA2_CAPABILITY_API Capability API

 API for querying capabilities of hardware devices and library.

 @{
 *****************************************************************************/

#ifndef __GNA2_CAPABILITY_API_H
#define __GNA2_CAPABILITY_API_H

#include "gna2-common-api.h"

#include <stdint.h>

/**
 List of device generations.

 Generation determines the set of capabilities common amongst multiple device versions.
 @see Gna2DeviceVersion.
 */
enum Gna2DeviceGeneration
{
    /**
     Legacy device supporting only Gaussian Mixture Models scoring.
     */
    Gna2DeviceGenerationGmm = 0x010,

    /**
     Initial GNA device generation with no CNN support.
     Backward compatible with ::Gna2DeviceGenerationGmm.
     */
    Gna2DeviceGeneration0_9 = 0x090,

    /**
     First fully featured GNA device generation.
     Backward compatible with ::Gna2DeviceGeneration0_9.
     */
    Gna2DeviceGeneration1_0 = 0x100,

    /**
     Fully featured second GNA device generation.
     Backward compatible with ::Gna2DeviceGeneration1_0.
     */
    Gna2DeviceGeneration2_0 = 0x200,

    /**
     Fully featured third GNA device generation.
     Partially compatible with ::Gna2DeviceGeneration2_0.
     */
    Gna2DeviceGeneration3_0 = 0x300,
};

/**
 Generation of device that is used by default by GNA Library in software mode,
 when no hardware device is available.

 @see
 GNA2_DEFAULT_DEVICE_VERSION.

 @note
 Usually it will be the latest existing GNA generation (excluding embedded)
 on the time of publishing the library, value may change with new release.
 */
#define GNA2_DEFAULT_DEVICE_GENERATION Gna2DeviceGeneration3_0

#endif // __GNA2_CAPABILITY_API_H

/**
 @}
 @}
 */

// / * *
// List of API versions.
// */
//enum Gna2ApiVersion
//{
//    / **
//     Previous GNA API 1.0.
//     */
//    GNA2_API_1_0 = 1,
//
//    /* *
//     Current GNA API 2.0.
//     */
//    GNA2_API_2_0 = 2,
//
//    /* *
//     Indicates that API version is not supported.
//     */
//    GNA2_API_VERSION_NOT_SUPPORTED = GNA2_NOT_SUPPORTED,
//};
//
//// Binary flags
//typedef enum _memory_mode
//{
//    GNA2_MEMORY_NOT_SUPPORTED = GNA2_NOT_SUPPORTED,
//    GNA2_MEMORY_MAPPED = 1,
//    GNA2_MEMORY_DIRECT = 2,
//    GNA2_MEMORY_DEDICATED = 4, // Server device built-in memory
//    GNA2_MEMORY_FANCY_MODE = 8, // ACE extension?
//} gna_memory_mode;

//typedef enum _api_property
//{
//    GNA2_API_VERSION,                            // GNA2_API_VERSION_T
//    GNA2_API_BUILD,                              // GNA2_UINT32_T
//    GNA2_API_THREAD_COUNT,                       // GNA2_UINT32_T
//    GNA2_API_THREAD_COUNT_MAX,                   // GNA2_UINT32_T
//
//    GNA2_API_PROPERTY_COUNT
//} gna_api_property;

//enum Gna2DevicePropertyType
//{
//    GNA2_DEVICE_AVAILABLE_COUNT,                 // GNA2_UINT32_T
//    GNA2_DEVICE_ACTIVE_COUNT,                    // GNA2_UINT32_T
//    GNA2_DEVICE_PROFILE,                         // GNA2_DEVICE_PROFILE_T
//    GNA2_DEVICE_VERSION,                         // GNA2_DEVICE_GENERATION_T
//    GNA2_DEVICE_DRIVER_BUILD,                    // GNA2_UINT32_T
//    GNA2_DEVICE_CLOCK_FREQUENCY,                 // GNA2_UINT32_T
//    GNA2_DEVICE_COMPUTE_ENGINE_COUNT,            // GNA2_UINT32_T
//    GNA2_DEVICE_ACTIVATION_ENGINE_COUNT,         // GNA2_UINT32_T
//    GNA2_DEVICE_POOLING_ENGINE_COUNT,            // GNA2_UINT32_T
//    GNA2_DEVICE_STREAM_COUNT,                    // GNA2_UINT32_T
//    GNA2_DEVICE_INPUT_BUFFER_SIZE,               // GNA2_UINT64_T
//    GNA2_DEVICE_MEMORY_MODE,                     // GNA2_MEMORY_MODE_T
//    GNA2_DEVICE_MEMORY_DEDICATED_SIZE,           // GNA2_UINT64_T
//    GNA2_DEVICE_MEMORY_REGIONS_COUNT,            // GNA2_UINT32_T
//    GNA2_DEVICE_MEMORY_SUPPORTED_SIZE,            // GNA2_UINT32_T
//    GNA2_DEVICE_MODEL_COUNT_MAX,                 // GNA2_UINT64_T
//    // ANNA
//    GNA2_DEVICE_EXT_,           // GNA2_UINT32_T
//};

///** Maximum number of requests that can be enqueued before retrieval */
//const uint32_t GNA2_REQUEST_QUEUE_LENGTH = 64;
//
///** Maximum supported time of waiting for a request in milliseconds. */
//const uint32_t GNA2_REQUEST_TIMEOUT_MAX = 180000;
//
//enum Gna2PropertyType
//{
//    /**
//     Determines if property is supported in given context.
//
//     A single char value, where 0 stands for False and 1 for True.
//     */
//    GNA2_PROPERTY_IS_SUPORTED = 0,
//
//    /**
//     Current value of the property
//
//     A single int64_t value.
//     */
//    GNA2_PROPERTY_CURRENT_VALUE = 1,
//
//    /**
//     Default value of a parameter, when not set by the user.
//
//     A single int64_t value.
//     */
//    GNA2_PROPERTY_DEFAULT_VALUE = 2,
//
//    /**
//     Minimal valid value (inclusive).
//
//     A single int64_t value.
//     */
//    GNA2_PROPERTY_MINIMUM = 3,
//
//    /**
//     Maximal valid value (inclusive).
//
//     A single int64_t value.
//     */
//    GNA2_PROPERTY_MAXIMUM = 4,
//
//    /**
//     Multiplicity (or step) of valid values.
//
//     A single int64_t value.
//     */
//    GNA2_PROPERTY_MULTIPLICITY = 5,
//
//    /**
//     Required alignment of data buffer pointers in bytes.
//
//     A single int64_t value.
//    */
//    GNA2_PROPERTY_ALIGNMENT = GNA2_PROPERTY_MULTIPLICITY,
//
//    /**
//     Set (array) of valid values, applicable mostly for enumerations.
//     @see GNA2_PROPERTY_VALUE_SET_SIZE.
//
//     An array of GNA2_PROPERTY_VALUE_SET_SIZE elements, each single uint64_t value.
//     */
//    GNA2_PROPERTY_VALUE_SET = 6,
//
//    /**
//     The size of the valid values set, in terms of elements.
//     @see GNA2_PROPERTY_VALUE_SET.
//
//     A single uint64_t value.
//     */
//    GNA2_PROPERTY_VALUE_SET_SIZE = 7,
//
//    /**
//     Special type, used where property is not applicable or unnecessary.
//     */
//    GNA2_PROPERTY_NONE = GNA2_DISABLED,
//};
//
///**
// Determines the parameters of GNA Properties in TLV-like format.
// */
//struct Gna2Property
//{
//    enum Gna2PropertyType Type;
//
//    uint32_t Size;
//
//    void * Value;
//};


//**
// * Test if given mode is set amongst flags
// *
// * @modeFlags   A value or bitwise OR of more values from GNA mode enumeration.
// * @mode        A tested mode value from GNA mode enumeration.
// * @return true if mode is set, false otherwise
//*/
//inline bool Gna2IsFlagSet(uint32_t modeFlags, uint32_t mode)
//{
//    if (modeFlags & mode || GNA2_NOT_SUPPORTED == mode)
//    {
//        return true;
//    }
//    return false;
//}
//
// TODO:enable querying some properties on non-available devices like SueScreek
//
//
// dedicated query functions
//GNA2_API enum Gna2Status Gna2GetApiProperty(
//    gna_api_property property,
//    void* poropertyValue,                       // [out] value of returned property, pointer to allocated 8Byte memory region
//    gna_property_type* propertyValueType);      // [out] type of returned property
//
//// optional
//GNA2_API enum Gna2Status Gna2SetApiProperty(
//    gna_api_property property,
//    void* poropertyValue);                      // value of property, pointer to allocated 8Byte memory region
//
//// e,g,     propertyString = "GNA2_LAYER_POOLING_MODE"
//GNA2_API enum Gna2Status Gna2ApiPropertyNameToString(
//    gna_api_property property,
//    char const ** propertyString);               // [out] c-string containing property name, allocated by GNA
//
//// e,g,     propertyString = "GNA2_POOLING_MAX | GNA2_POOLING_SUM"
//GNA2_API enum Gna2Status Gna2ApiPropertyValueToString(
//    gna_api_property property,
//    void* poropertyValue,                       // value of property
//    char const ** propertyString);               // [out] c-string containing property value, allocated by GNA

//GNA2_API enum Gna2Status Gna2GetDeviceProperty(
//    uint32_t device,                       // id/index of device <0;GNA2_DEVICE_AVAILABLE_COUNT-1>
//    enum Gna2DevicePropertyType capability,
//    enum Gna2PropertyType property,
//    struct Gna2Property * deviceProperty);
//
//GNA2_API enum Gna2Status Gna2GetDeviceProperty(
//    uint32_t device,                       // id/index of device <0;GNA2_DEVICE_AVAILABLE_COUNT-1>
//    enum Gna2DevicePropertyType capability,
//    struct Gna2Property * deviceProperties);
//
//GNA2_API enum Gna2Status Gna2SetDeviceProperty(
//    uint32_t device,
//    gna_device_property property,
//    void* poropertyValue);                      // value of property, pointer to allocated 8Byte memory region
//
//// e,g,     propertyString = "GNA2_LAYER_POOLING_MODE"
//GNA2_API enum Gna2Status Gna2DevicePropertyNameToString(
//    gna_device_property property,
//    char const * propertyString);               // [out] c-string containing property name, allocated by GNA
//
//// e,g,     propertyString = "GNA2_POOLING_MAX | GNA2_POOLING_SUM"
//GNA2_API enum Gna2Status Gna2DevicePropertyValueToString(
//    gna_device_property property,
//    void* poropertyValue,                       // value of property
//    char const * propertyString);               // [out] c-string containing property value, allocated by GNA
//
//GNA2_API enum Gna2Status Gna2GetLayerProperty(
//    uint32_t device,
//    Gna2OperationMode layerOperation,
//    gna_layer_property property,
//    void* poropertyValue,                       // [out] value of returned property, pointer to allocated 8Byte memory region
//    gna_property_type* propertyValueType);      // [out] type of returned property
//
//GNA2_API enum Gna2Status Gna2SetLayerProperty(
//    uint32_t device,
//    Gna2OperationMode layerOperation,
//    gna_layer_property property,
//    void* poropertyValue);                      // value of property, pointer to allocated 8Byte memory region
//
//// e,g,     propertyString = "GNA2_LAYER_POOLING_MODE"
//GNA2_API enum Gna2Status Gna2LayerPropertyNameToString(
//    gna_layer_property property,
//    char const * propertyString);               // [out] c-string containing property name, allocated by GNA
//
//// e,g,     propertyString = "GNA2_POOLING_MAX | GNA2_POOLING_SUM"
//GNA2_API enum Gna2Status Gna2LayerPropertyValueToString(
//    gna_layer_property property,
//    void* poropertyValue,                       // value of property
//    char const * propertyString);               // [out] c-string containing property value, allocated by GNA
//
//
//// Query hardware device properties even if not present in system, like SueCreek
//GNA2_API enum Gna2Status Gna2GetHardwareDeviceProperty(
//    enum Gna2DeviceGeneration generation,         // hardware device generation identifier, for not present devices
//    gna_device_property property,
//    void* poropertyValue,                       // [out] value of returned property, pointer to allocated 8Byte memory region
//    gna_property_type* propertyValueType);      // [out] type of returned property
//
//GNA2_API enum Gna2Status Gna2GetHardwareLayerProperty(
//    enum Gna2DeviceGeneration generation,         // hardware device generation identifier, for not present devices
//    Gna2OperationMode layerOperation,
//    gna_layer_property property,
//    void* poropertyValue,                       // [out] value of returned property, pointer to allocated 8Byte memory region
//    gna_property_type* propertyValueType);      // [out] type of returned property
