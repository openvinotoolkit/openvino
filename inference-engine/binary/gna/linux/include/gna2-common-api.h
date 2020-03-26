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
 @file gna2-common-api.h
 @brief Gaussian and Neural Accelerator (GNA) 2.0 API Definition.
 @nosubgrouping

 ******************************************************************************

 @addtogroup GNA2_API
 @{
 ******************************************************************************

 @addtogroup GNA2_COMMON_API Common API

 API with commonly used and auxiliary declarations.

 @{
 *****************************************************************************/

#ifndef __GNA2_COMMON_API_H
#define __GNA2_COMMON_API_H

#include <stdint.h>
#include <stdbool.h>

 /**
  C-linkage macro.
  */
#ifdef __cplusplus
#define GNA2_API_C extern "C"
#else
#define GNA2_API_C
#endif

  /**
   Library API import/export macro.
   */
#if !defined(GNA2_API_EXPORT)
#    if 1 == _WIN32
#       if 1 == INTEL_GNA_DLLEXPORT
#           define GNA2_API_EXPORT __declspec(dllexport)
#       else
#           define GNA2_API_EXPORT __declspec(dllimport)
#       endif
#    else
#        if __GNUC__ >= 4
#           define GNA2_API_EXPORT __attribute__ ((visibility ("default")))
#        else
#           define GNA2_API_EXPORT
#        endif
#    endif
#endif

   /**
    Library C-API import/export macro.
    */
#define GNA2_API GNA2_API_C GNA2_API_EXPORT

    /** Constant indicating that feature is disabled. */
#define GNA2_DISABLED (-1)

/** Constant indicating that value is default. */
#define GNA2_DEFAULT (0)

/** Constant indicating that feature is not available. */
#define GNA2_NOT_SUPPORTED (1u << 31)

/**
 List of device versions.

 Version determines concrete GNA device derivative.
 Devices of the same version are always single generation and have the same
 properties e.g. frequency, bandwidth.

 @see Gna2DeviceGeneration
 */
enum Gna2DeviceVersion
{
    /**
     Gaussian Mixture Models device.
     A ::Gna2DeviceGenerationGmm generation device.
     Code names: SkyLake (SKL), KabyLake (KBL), CoffeeLake (CFL).
     */
    Gna2DeviceVersionGMM = 0x01,

    /**
     GNA 0.9 device.
     A ::Gna2DeviceGeneration0_9 generation device.
     Code names: CannonLake (CNL).
     */
    Gna2DeviceVersion0_9 = 0x09,

    /**
     GNA 1.0 device.
     A ::Gna2DeviceGeneration1_0 generation device.
     Code names: GeminiLake (GLK), ElkhartLake (EHL), IceLake (ICL).
     */
    Gna2DeviceVersion1_0 = 0x10,

    /**
     GNA 2.0 device.
     A ::Gna2DeviceGeneration2_0 generation device.
     Code names: TigerLake (TGL).
     */
    Gna2DeviceVersion2_0 = 0x20,

    /**
     GNA 3.0 device.
     A ::Gna2DeviceGeneration3_0 generation device.
     Code names: AlderLake (ADL).
     */
    Gna2DeviceVersion3_0 = 0x30,

    /**
     GNA 1.0 embedded device.
     A ::Gna2DeviceGeneration1_0 generation device.
     Code names: SueCreek (SUE).
     */
    Gna2DeviceVersionEmbedded1_0 = 0x10E,

    /**
     GNA 2.1 embedded device.
     A ::Gna2DeviceGeneration2_0 generation device.
     Code names: JellyFish (JLF).
     */
    Gna2DeviceVersionEmbedded2_1 = 0x20E,

    /**
     GNA 3.0 embedded device on PCH/ACE.
     A ::Gna2DeviceGeneration3_0 generation device.
     Code names: AlderLake (ADL).
     */
    Gna2DeviceVersionEmbedded3_0 = 0x30E,

    /**
     GNA ANNA autonomous embedded device on ACE.
     A ::Gna2DeviceGeneration3_0 generation device.
     Code names: AlderLake (ADL).
     */
    Gna2DeviceVersionEmbedded3_1 = 0x31A,

    /**
     Value indicating no supported hardware device available.
     Software emulation (fall-back) will be used.

     @see ::GNA2_DEFAULT_DEVICE_VERSION and Gna2RequestConfigEnableHardwareConsistency().
     */
    Gna2DeviceVersionSoftwareEmulation = GNA2_DEFAULT,
};

/**
 Version of device that is used by default by GNA Library in software mode,
 when no hardware device is available.

 @see
 Gna2RequestConfigEnableHardwareConsistency() to change hardware device
 version in software mode.

 @note
 Usually it will be the latest existing GNA device (excluding embedded)
 on the time of publishing the library, value may change with new release.
 */
#define GNA2_DEFAULT_DEVICE_VERSION Gna2DeviceVersion3_0

 /**
  GNA API Status codes.
  */
enum Gna2Status
{
    /**
     Success: Operation completed successfully without errors or warnings.
     */
    Gna2StatusSuccess = GNA2_DEFAULT,

    /**
     Warning: Device is busy.
     GNA is still running, can not enqueue more requests.
     */
    Gna2StatusWarningDeviceBusy = 1,

    /**
     Warning: Arithmetic saturation.
     An arithmetic operation has resulted in saturation during calculation.
     */
    Gna2StatusWarningArithmeticSaturation = 2,

    /**
     Error: Unknown error occurred.
     */
    Gna2StatusUnknownError = -3,

    /**
     Error: Functionality not implemented yet.
     */
    Gna2StatusNotImplemented = -4,

    /**
     Error: Item identifier is invalid.
     Provided item (e.g. device or request) id or index is invalid.
    */
    Gna2StatusIdentifierInvalid = -5,

    /**
    Error: NULL argument is not allowed.
   */
    Gna2StatusNullArgumentNotAllowed = -6,

    /**
     Error: NULL argument is required.
    */
    Gna2StatusNullArgumentRequired = -7,

    /**
     Error: Unable to create new resources.
    */
    Gna2StatusResourceAllocationError = -8,

    /**
     Error: Device: not available.
    */
    Gna2StatusDeviceNotAvailable = -9,

    /**
     Error: Device failed to open, thread count is invalid.
    */
    Gna2StatusDeviceNumberOfThreadsInvalid = -10,
    /**
     Error: Device version is invalid.
    */
    Gna2StatusDeviceVersionInvalid = -11,

    /**
     Error: Queue can not create or enqueue more requests.
    */
    Gna2StatusDeviceQueueError = -12,

    /**
     Error: Failed to receive communication from the device driver.
    */
    Gna2StatusDeviceIngoingCommunicationError = -13,

    /**
     Error: Failed to sent communication to the device driver.
    */
    Gna2StatusDeviceOutgoingCommunicationError = -14,
    /**
     Error: Hardware device parameter out of range error occurred.
    */
    Gna2StatusDeviceParameterOutOfRange = -15,

    /**
     Error: Hardware device virtual address out of range error occurred.
    */
    Gna2StatusDeviceVaOutOfRange = -16,

    /**
     Error: Hardware device unexpected completion occurred during PCIe operation.
    */
    Gna2StatusDeviceUnexpectedCompletion = -17,

    /**
     Error: Hardware device DMA error occurred during PCIe operation.
    */
    Gna2StatusDeviceDmaRequestError = -18,

    /**
     Error: Hardware device MMU error occurred during PCIe operation.

    */
    Gna2StatusDeviceMmuRequestError = -19,

    /**
     Error: Hardware device break-point hit.
    */
    Gna2StatusDeviceBreakPointHit = -20,

    /**
     Error: Critical hardware device error occurred, device has been reset.
    */
    Gna2StatusDeviceCriticalFailure = -21,

    /**
     Error: Memory buffer alignment is invalid.
    */
    Gna2StatusMemoryAlignmentInvalid = -22,

    /**
     Error: Memory buffer size is invalid.
    */
    Gna2StatusMemorySizeInvalid = -23,

    /**
     Error: Model total memory size exceeded.
    */
    Gna2StatusMemoryTotalSizeExceeded = -24,

    /**
     Error: Memory buffer is invalid.
     E.g. outside of allocated memory or already released.
    */
    Gna2StatusMemoryBufferInvalid = -25,

    /**
     Error: Waiting for a request failed.
    */
    Gna2StatusRequestWaitError = -26,

    /**
    Error: Invalid number of active indices.
   */
    Gna2StatusActiveListIndicesInvalid = -27,

    /**
     Error: Acceleration mode is not supported on this computer.
    */
    Gna2StatusAccelerationModeNotSupported = -28,

    /**
     Error: Model configuration is not supported.
    */
    Gna2StatusModelConfigurationInvalid = -29,

    /**
     Error: Number is not a multiple of given number
    */
    Gna2StatusNotMultipleOf = -30,

    /* FIXME: this is just for gna_status_t compatibility, use model errors in the future */
    Gna2StatusBadFeatLength = -31,
    Gna2StatusDataModeInvalid = -32,
    Gna2StatusXnnErrorNetLyrNo = -33,
    Gna2StatusXnnErrorNetworkInputs = -34,
    Gna2StatusXnnErrorNetworkOutputs = -35,
    Gna2StatusXnnErrorLyrOperation = -36,
    Gna2StatusXnnErrorLyrCfg = -37,
    Gna2StatusXnnErrorLyrInvalidTensorOrder = -38,
    Gna2StatusXnnErrorLyrInvalidTensorDimensions = -39,
    Gna2StatusXnnErrorInvalidBuffer = -40,
    Gna2StatusXnnErrorNoFeedback = -41,
    Gna2StatusXnnErrorNoLayers = -42,
    Gna2StatusXnnErrorGrouping = -43,
    Gna2StatusXnnErrorInputBytes = -44,
    Gna2StatusXnnErrorInputVolume = -45,
    Gna2StatusXnnErrorOutputVolume = -46,
    Gna2StatusXnnErrorIntOutputBytes = -47,
    Gna2StatusXnnErrorOutputBytes = -48,
    Gna2StatusXnnErrorWeightBytes = -49,
    Gna2StatusXnnErrorWeightVolume = -50,
    Gna2StatusXnnErrorBiasBytes = -51,
    Gna2StatusXnnErrorBiasVolume = -52,
    Gna2StatusXnnErrorBiasMode = -53,
    Gna2StatusXnnErrorBiasMultiplier = -54,
    Gna2StatusXnnErrorBiasIndex = -55,
    Gna2StatusXnnErrorPwlSegments = -56,
    Gna2StatusXnnErrorPwlData = -57,
    Gna2StatusXnnErrorConvFltBytes = -58,
    Gna2StatusCnnErrorConvFltCount = -59,
    Gna2StatusCnnErrorConvFltVolume = -60,
    Gna2StatusCnnErrorConvFltStride = -61,
    Gna2StatusCnnErrorConvFltPadding = -62,
    Gna2StatusCnnErrorPoolStride = -63,
    Gna2StatusCnnErrorPoolSize = -64,
    Gna2StatusCnnErrorPoolType = -65,
    Gna2StatusGmmBadMeanWidth = -66,
    Gna2StatusGmmBadMeanOffset = -67,
    Gna2StatusGmmBadMeanSetoff = -68,
    Gna2StatusGmmBadMeanAlign = -69,
    Gna2StatusGmmBadVarWidth = -70,
    Gna2StatusGmmBadVarOffset = -71,
    Gna2StatusGmmBadVarSetoff = -72,
    Gna2StatusGmmBadVarsAlign = -73,
    Gna2StatusGmmBadGconstOffset = -74,
    Gna2StatusGmmBadGconstAlign = -75,
    Gna2StatusGmmBadMixCnum = -76,
    Gna2StatusGmmBadNumGmm = -77,
    Gna2StatusGmmBadMode = -78,
    Gna2StatusGmmCfgInvalidLayout = -79,
};

/**
 Verifies if status of the operation indicates it was successful.

 @param status The status code returned from API function.
 @return Successful status indicator.
    @retval true The status indicates success or warning.
    @retval false The status indicates some error.
 */
GNA2_API inline bool Gna2StatusIsSuccessful(enum Gna2Status status)
{
    return (Gna2StatusSuccess == status
        || Gna2StatusWarningArithmeticSaturation == status
        || Gna2StatusWarningDeviceBusy == status);
}

/**
 Gets message with detailed description of given status.

 @param status The status code returned from API function.
 @param [out] messageBuffer User allocated buffer for the message.
 @param [in] messageBufferSize The size of the messageBuffer in bytes.
        The message length varies depending on the status,
        the buffer of size Gna2StatusGetMaxMessageLength() is sufficient for every status.
 @return Status of fetching the message.
    @retval Gna2StatusSuccess The status was fully serialized into the messageBuffer.
    @retval Gna2StatusMemorySizeInvalid The messageBuffer is too small.
    @retval Gna2StatusNullArgumentNotAllowed The messageBuffer was NULL.
    @retval Gna2StatusIdentifierInvalid The status code is unknown.
 */
GNA2_API enum Gna2Status Gna2StatusGetMessage(enum Gna2Status status,
    char * messageBuffer, uint32_t messageBufferSize);

/**
 Gets maximal length of buffer needed by Gna2StatusGetMessage().

 @return Size [in bytes] of buffer needed.
 */
GNA2_API uint32_t Gna2StatusGetMaxMessageLength();

/**
 Rounds a number up, to the lowest multiple of significance.

 The function rounds the number up to the lowest possible value divisible
 by "significance".
 Used for calculating the memory sizes for GNA data buffers.

 @param number Memory size or a number to round up.
 @param significance The number that rounded value have to be divisible by.
 @return Rounded integer value.
 */

inline uint32_t Gna2RoundUp(uint32_t number, uint32_t significance)
{
    return ((uint32_t)((number)+significance - 1) / significance) * significance;
}

/**
 Rounds a number up, to the lowest multiple of 64.
 @see Gna2RoundUp().

 @param number Memory size or a number to round up.
 @return Rounded integer value.
 */
inline uint32_t Gna2RoundUpTo64(uint32_t number)
{
    return Gna2RoundUp(number, 64);
}

/**
 Definition of callback that is used to allocate "user owned" memory for model definition.

 Used for allocating "non-GNA" memory buffers used for model export or data-flow model
 structures (not model data).

 @warning
    User is responsible for releasing allocated memory buffers.

 @param size The size of the buffer to allocate, provided by GNA library.
 @return Allocated buffer.
 @retval NULL in case of allocation failure
 */
typedef void* (*Gna2UserAllocator)(uint32_t size);

#endif //ifndef __GNA2_COMMON_API_H

/**
 @}
 @}
*/
