// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///
/// @file
/// @brief     Application configuration Leon header
///

#ifndef _XLINK_H
#define _XLINK_H
#include "XLinkPublicDefines.h"

#ifdef __cplusplus
extern "C"
{
#endif

// ------------------------------------
// Device management. Begin.
// ------------------------------------

/**
 * @brief Initializes XLink and scheduler
 * @param handler[in] XLink global communication parameters
 * Now XLink can work with PCIe and USB simultaneously.
 */
XLinkError_t XLinkInitialize(XLinkGlobalHandler_t* handler);

/**
 * @brief Return Myriad device description which meets the requirements
 * @param[in]   state - state of device enum (booted, not booted or any state)
 * @param[in]   in_deviceRequirements - structure with device requirements (protocol, platform).
 * @note        If in_deviceRequirements has device name specified,
 *                  this function tries to get a device with that exact name
 *                  and fails if such device is unavailable
 * @param[out]  out_foundDevice - found device description
 */
XLinkError_t XLinkFindFirstSuitableDevice(XLinkDeviceState_t state,
                                          const deviceDesc_t in_deviceRequirements,
                                          deviceDesc_t *out_foundDevice);

/**
 * @brief Return all Myriad devices description which meets the requirements
 * @param[in]      state - state of device enum (booted, not booted or any state)
 * @param[in]      in_deviceRequirements - structure with device requirements (protocol, platform).
 * @param[in,out]  out_foundDevicesPtr - pointer to array with all found devices descriptions
 * @param[out]     out_foundDevicesCount - amount of found devices
 */
XLinkError_t XLinkFindAllSuitableDevices(XLinkDeviceState_t state,
                                         const deviceDesc_t in_deviceRequirements,
                                         deviceDesc_t *out_foundDevicesPtr,
                                         const unsigned int devicesArraySize,
                                         unsigned int *out_foundDevicesCount);

/**
 * @brief Connects to specific device, starts dispatcher and pings remote
 * @param[in,out] handler – XLink communication parameters (file path name for underlying layer)
 */
XLinkError_t XLinkConnect(XLinkHandler_t* handler);

/**
 * @brief Boots specified firmware binary to the remote device
 * @param deviceDesc - device description structure, obtained from XLinkFind* functions call
 * @param binaryPath - path to the *.mvcmd file
 */
XLinkError_t XLinkBoot(deviceDesc_t* deviceDesc, const char* binaryPath);

/**
 * @brief Reset the remote device and close all open local handles for this device
 * @warning This function should be used in a host application
 * @param[in] id – link Id obtained from XLinkConnect in the handler parameter
 */
XLinkError_t XLinkResetRemote(linkId_t id);

/**
 * @brief Close all and release all memory
 */
XLinkError_t XLinkResetAll();

/**
 * @brief Profiling funcs - keeping them global for now
 */
XLinkError_t XLinkProfStart();
XLinkError_t XLinkProfStop();
XLinkError_t XLinkProfPrint();

// ------------------------------------
// Device management. End.
// ------------------------------------




// ------------------------------------
// Device streams management. Begin.
// ------------------------------------

/**
 * @brief Opens a stream in the remote that can be written to by the local
 *        Allocates stream_write_size (aligned up to 64 bytes) for that stream
 * @param[in] id – link Id obtained from XLinkConnect in the handler parameter
 * @param[in] name – stream name
 * @param[in] stream_write_size – stream buffer size
 */
streamId_t XLinkOpenStream(linkId_t id, const char* name, int stream_write_size);

/**
 * @brief Close stream for any further data transfer
 *        Stream will be deallocated when all pending data has been released
 * @param[in] streamId - link Id obtained from XLinkOpenStream call
 */
XLinkError_t XLinkCloseStream(streamId_t streamId);

/**
 * @brief Send a package to initiate the writing of data to a remote stream
 * @warning Actual size of the written data is ALIGN_UP(size, 64)
 * @param[in] streamId – stream link Id obtained from XLinkOpenStream call
 * @param[in] buffer – data buffer to be transmitted
 * @param[in] size – size of the data to be transmitted
 */
XLinkError_t XLinkWriteData(streamId_t streamId, const uint8_t* buffer, int size);

/**
 * @brief Read data from local stream. Will only have something if it was written to by the remote
 * @param[in]   streamId – stream link Id obtained from XLinkOpenStream call
 * @param[out]  packet – structure containing output data buffer and received size
 */
XLinkError_t XLinkReadData(streamId_t streamId, streamPacketDesc_t** packet);

/**
 * @brief Release data from stream - This should be called after the data obtained from
 *  XlinkReadData is processed
 * @param[in] streamId – stream link Id obtained from XLinkOpenStream call
 */
XLinkError_t XLinkReleaseData(streamId_t streamId);

/**
 * @brief Read fill level of the local or remote queues
 * @param[in]   streamId – stream link Id obtained from XLinkOpenStream call
 * @param[in]   isRemote – 0 – local queue; any other value – remote queue
 * @param[out]  fillLevel – fill level of the selected queue
 */
XLinkError_t XLinkGetFillLevel(streamId_t streamId, int isRemote, int* fillLevel);

// ------------------------------------
// Device streams management. End.
// ------------------------------------




// ------------------------------------
// Deprecated API. Begin.
// ------------------------------------

XLinkError_t XLinkGetDeviceName(int index, char* name, int nameSize);
XLinkError_t XLinkGetDeviceNameExtended(int index, char* name, int nameSize, int pid);

XLinkError_t XLinkBootRemote(const char* deviceName, const char* binaryPath);
XLinkError_t XLinkDisconnect(linkId_t id);

XLinkError_t XLinkGetAvailableStreams(linkId_t id);

XLinkError_t XLinkWriteDataWithTimeout(streamId_t streamId, const uint8_t* buffer, int size, unsigned int timeout);
XLinkError_t XLinkAsyncWriteData();

XLinkError_t XLinkReadDataWithTimeOut(streamId_t streamId, streamPacketDesc_t** packet, unsigned int timeout);

XLinkError_t XLinkSetDeviceOpenTimeOutMsec(unsigned int msec);
XLinkError_t XLinkSetCommonTimeOutMsec(unsigned int msec);

// ------------------------------------
// Deprecated API. End.
// ------------------------------------

#ifdef __cplusplus
}
#endif

#endif
