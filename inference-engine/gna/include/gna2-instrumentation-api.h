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
 @file gna2-instrumentation-api.h
 @brief Gaussian and Neural Accelerator (GNA) 2.0 API Definition.
 @nosubgrouping

 ******************************************************************************

 @addtogroup GNA2_API
 @{
 ******************************************************************************

 @addtogroup GNA2_API_INSTRUMENTATION Instrumentation API

 API for querying inference performance statistics.

 @{
 *****************************************************************************/

#ifndef __GNA2_INSTRUMENTATION_API_H
#define __GNA2_INSTRUMENTATION_API_H

#include "gna2-common-api.h"

#include <stdint.h>

/**
 Inference request instrumentation points.
 */
enum Gna2InstrumentationPoint
{
    /**
     Request preprocessing start, from library instrumentation.
     */
    Gna2InstrumentationPointLibPreprocessing = 0,

    /**
     Request submission start, from library instrumentation.
     */
    Gna2InstrumentationPointLibSubmission = 1,

    /**
     Request processing start, from library instrumentation.
     */
    Gna2InstrumentationPointLibProcessing = 2,

    /**
     Request execution start, from library instrumentation.
     Actual software computation or issuing device request.
     */
    Gna2InstrumentationPointLibExecution = 3,

    /**
     Request ready to send to device, from library instrumentation.
     */
    Gna2InstrumentationPointLibDeviceRequestReady = 4,

    /**
     Request sent to device, from library instrumentation.
     */
    Gna2InstrumentationPointLibDeviceRequestSent = 5,

    /**
     Request completed by device, from library instrumentation.
     */
    Gna2InstrumentationPointLibDeviceRequestCompleted = 6,

    /**
     Request execution completed, from library instrumentation.
     Actual software computation done or device request notified.
     */
    Gna2InstrumentationPointLibCompletion = 7,

    /**
     Request received by user, from library instrumentation.
     */
    Gna2InstrumentationPointLibReceived = 8,

    /**
     Request preprocessing start, from driver instrumentation.
     */
    Gna2InstrumentationPointDrvPreprocessing = 9,

    /**
     Request processing started by hardware, from driver instrumentation.
     */
    Gna2InstrumentationPointDrvProcessing = 10,

    /**
     Request completed interrupt triggered by hardware, from driver instrumentation.
     */
    Gna2InstrumentationPointDrvDeviceRequestCompleted = 11,

    /**
     Request execution completed, from driver instrumentation.
     Driver completed interrupt and request handling.
     */
    Gna2InstrumentationPointDrvCompletion = 12,

    /**
     Total time spent on processing in hardware.
     Total = Compute + Stall
     @warning This event always provides time duration instead of time point.
     */
    Gna2InstrumentationPointHwTotalCycles = 13,

    /**
     Time hardware spent on waiting for data.
     @warning This event always provides time duration instead of time point.
     */
    Gna2InstrumentationPointHwStallCycles = 14,
};

/**
 Enables and configures instrumentation configuration.

 Instrumentation configurations have to be declared a priori to minimize the
 request preparation time and reduce processing latency.
 Configurations can be shared with multiple request configurations when the request
 with the current configuration has been completed and retrieved by Gna2RequestWait().

 @see
    Gna2RequestConfigSetInstrumentationUnit and ::Gna2InstrumentationUnitMicroseconds
    for description of result units.

 @see Gna2RequestConfigSetInstrumentationMode and Gna2InstrumentationMode
    for description of hardware instrumentation.

 @param numberOfInstrumentationPoints A number of selected instrumentation points.
 @param selectedInstrumentationPoints An array of selected instrumentation points.
 @param results Buffer to save instrumentation results to.
    Result buffer size have to be at least numberOfInstrumentationPoints * sizeof(uint64_t).
 @param [out] instrumentationConfigId Identifier of created instrumentation configuration.
 @return Status of the operation.
 */
GNA2_API enum Gna2Status Gna2InstrumentationConfigCreate(
    uint32_t numberOfInstrumentationPoints,
    enum Gna2InstrumentationPoint* selectedInstrumentationPoints,
    uint64_t * results,
    uint32_t * instrumentationConfigId);

/**
 Assigns instrumentation config to given request configuration.

 @see Gna2RequestConfigRelease()

 @param instrumentationConfigId Identifier of instrumentation config used.
 @param requestConfigId Request configuration to modify.
 @return Status of the operation.
 */
GNA2_API enum Gna2Status Gna2InstrumentationConfigAssignToRequestConfig(
    uint32_t instrumentationConfigId,
    uint32_t requestConfigId);

/**
 Units that instrumentation will count and report.
 */
enum Gna2InstrumentationUnit
{
    /**
     Microseconds.

     Uses std::chrono. @see http://www.cplusplus.com/reference/chrono/
     */
    Gna2InstrumentationUnitMicroseconds = GNA2_DEFAULT,

    /**
     Milliseconds.

     Uses std::chrono. @see http://www.cplusplus.com/reference/chrono/
     */
    Gna2InstrumentationUnitMilliseconds = 1,

    /**
     Processor cycles.

     Uses RDTSC. @see https://en.wikipedia.org/wiki/Time_Stamp_Counter
     */
    Gna2InstrumentationUnitCycles = 2,
};

/**
 Sets instrumentation unit for given configuration.

 Instrumentation results will represent a value in selected units.
 @note
    ::Gna2InstrumentationUnitMicroseconds is used when not set.

 @param instrumentationConfigId Instrumentation configuration to modify.
 @param instrumentationUnit Type of hardware performance statistic.
 */
GNA2_API enum Gna2Status Gna2InstrumentationConfigSetUnit(
    uint32_t instrumentationConfigId,
    enum Gna2InstrumentationUnit instrumentationUnit);

/**
 Mode of instrumentation for hardware performance counters.

 When performance counting is enabled, the total scoring cycles counter is always on.
 In addition one of several reasons for stall may be measured to allow
 identifying the bottlenecks in the scoring operation.
 */
enum Gna2InstrumentationMode
{
    /**
     Total Stall cycles.
     */
    Gna2InstrumentationModeTotalStall = GNA2_DEFAULT,

    /**
     Wait For Dma Completion cycles.
     */
    Gna2InstrumentationModeWaitForDmaCompletion = 1,

    /**
     Wait For Mmu Translation cycles.
     */
    Gna2InstrumentationModeWaitForMmuTranslation = 2,

    /**
     Descriptor Fetch Time cycles.
     */
    Gna2InstrumentationModeDescriptorFetchTime = 3,

    /**
     Input Buffer Fill From Memory cycles.
     */
    Gna2InstrumentationModeInputBufferFillFromMemory = 4,

    /**
     Output Buffer Full Stall cycles.
     */
    Gna2InstrumentationModeOutputBufferFullStall = 5,

    /**
     Output Buffer Wait For IOSF Stall cycles.
     */
    Gna2InstrumentationModeOutputBufferWaitForIosfStall = 6,

    /**
     Hardware performance counters are disabled.
     */
    Gna2InstrumentationModeDisabled = GNA2_DISABLED,
};

/**
 Sets hardware instrumentation mode for given configuration.

 @note
    ::Gna2InstrumentationModeTotalStall is used when not set.

 @param instrumentationConfigId Instrumentation configuration to modify.
 @param instrumentationMode Mode of hardware instrumentation.
 */
GNA2_API enum Gna2Status Gna2InstrumentationConfigSetMode(
    uint32_t instrumentationConfigId,
    enum Gna2InstrumentationMode instrumentationMode);

/**
 Releases instrumentation config and its resources.

 @note Please make sure all requests using this config are completed.

 @param instrumentationConfigId Identifier of affected instrumentation configuration.
 @return Status of the operation.
 */
GNA2_API enum Gna2Status Gna2InstrumentationConfigRelease(
    uint32_t instrumentationConfigId);

#endif // __GNA2_INSTRUMENTATION_API_H

/**
 @}
 @}
 */
