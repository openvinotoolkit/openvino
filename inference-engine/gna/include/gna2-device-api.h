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
 @file gna2-device-api.h
 @brief Gaussian and Neural Accelerator (GNA) 2.0 API Definition.
 @nosubgrouping

 ******************************************************************************

 @addtogroup GNA2_API
 @{
 ******************************************************************************

 @addtogroup GNA2_DEVICE_API Device API

 API for accessing and managing GNA hardware and software devices.

 @{
 *****************************************************************************/

#ifndef __GNA2_DEVICE_API_H
#define __GNA2_DEVICE_API_H

#include "gna2-common-api.h"

#include <stdint.h>

/**
 Gets number of available GNA devices on this computer.

 Number of opened devices is not relevant.
 If hardware device is present on a platform but is not available via GNA API,
 please verify if driver is properly installed and hardware is supported
 by your GNA software version.
 If no hardware device is available device number is set to 1,
 as software device still can be used.
 @see Gna2DeviceGetVersion() to determine version of available device.

 @param [out] deviceCount Number of available devices.
 @return Status of the operation.
 */
GNA2_API enum Gna2Status Gna2DeviceGetCount(
    uint32_t * deviceCount);

/**
 Retrieves hardware device version.

 Devices are zero-based indexed.
 Select desired device providing deviceIndex from 0 to Gna2DeviceGetCount() - 1.
 Corresponding device does not have to be opened.
 @see Gna2DeviceGetCount().

 @param deviceIndex Index of queried device.
 @param [out] deviceVersion Gna2DeviceVersion identifier.
    Set to Gna2DeviceVersionSoftwareEmulation when no hardware GNA device is available.
 @return Status of the operation.
 @retval Gna2StatusIdentifierInvalid The device with such index does not exits.
 */
GNA2_API enum Gna2Status Gna2DeviceGetVersion(
    uint32_t deviceIndex,
    enum Gna2DeviceVersion * deviceVersion);

/**
 Opens and initializes GNA device for processing.

 Device indexes are zero-based.
 Select desired device providing deviceIndex from 0 to Gna2DeviceGetCount - 1.
 If no hardware devices are available, software device can be still opened
 with deviceIndex = 0.

 @note
 - The device with same index cab be opened multiple times (up to 1024) e.g. by different threads.
 - The device has to be closed the same number of times it has been opened to prevent resource leakage.

 @param deviceIndex Index of the device to be opened.
 @return Status of the operation.
 @retval Gna2StatusIdentifierInvalid The device with such index does not exits.
 @retval Gna2StatusDeviceNotAvailable The device has been opened maximum number of times.
 */
GNA2_API enum Gna2Status Gna2DeviceOpen(
    uint32_t deviceIndex);

/**
 Closes GNA device.
 
 The device has to be closed the same number of times it has been opened to prevent resource leakage.
 When last handle to the device is closed releases the corresponding device resources.

 @param deviceIndex The device to be closed.
 @return Status of the operation.
 */
GNA2_API enum Gna2Status Gna2DeviceClose(
    uint32_t deviceIndex);

/**
 Sets number of software worker threads for given device.

 @note
    Must be called synchronously.

 Device indexes are zero-based.
 Select desired device providing deviceIndex from 0 to Gna2DeviceGetCount() - 1.

 @param deviceIndex Index of the affected device.
 @param numberOfThreads Number of software worker threads [1,127]. Default is 1.
 @return Status of the operation.
 */
GNA2_API enum Gna2Status Gna2DeviceSetNumberOfThreads(
    uint32_t deviceIndex,
    uint32_t numberOfThreads);

#endif // __GNA2_DEVICE_API_H

/**
 @}
 @}
 */
