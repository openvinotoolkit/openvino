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

// Set global common time out for all XLink operations.
XLinkError_t XLinkSetCommonTimeOutMsec(unsigned int msec);

// Set global device open time out for all XLink operations.
XLinkError_t XLinkSetDeviceOpenTimeOutMsec(unsigned int msec);

// Set global allocate graph time out for all XLink operations.
XLinkError_t XLinkSetAllocateGraphTimeOutMsec(unsigned int msec);

// Initializes XLink and scheduler
XLinkError_t XLinkInitialize(XLinkGlobalHandler_t* handler);

// Connects to specific device, starts dispatcher and pings remote
XLinkError_t XLinkConnect(XLinkHandler_t* handler);

// Opens a stream in the remote that can be written to by the local
// Allocates stream_write_size (aligned up to 64 bytes) for that stream
streamId_t XLinkOpenStream(linkId_t id, const char* name, int stream_write_size);

// Close stream for any further data transfer
// Stream will be deallocated when all pending data has been released
XLinkError_t XLinkCloseStream(streamId_t streamId);

// Currently useless
XLinkError_t XLinkGetAvailableStreams(linkId_t id);

/**
 * @brief Return Myriad device name
 * @param index Return device on index from suitable (matches pid argument) devices list
 * @param pid   0x2485 for MX, 0x2150 for M2, 0 for any, -1 for any not booted 
 */
XLinkError_t XLinkGetDeviceName(int index, char* name, int nameSize, int pid);

// Send a package to initiate the writing of data to a remote stream
// Note that the actual size of the written data is ALIGN_UP(size, 64)
XLinkError_t XLinkWriteData(streamId_t streamId, const uint8_t* buffer, int size);

// Send a package to initiate the writing of data to a remote stream with specific timeout
// Note that the actual size of the written data is ALIGN_UP(size, 64)
XLinkError_t XLinkWriteDataWithTimeout(streamId_t streamId, const uint8_t* buffer, int size, unsigned int timeout);

// Currently useless
XLinkError_t XLinkAsyncWriteData();

// Read data from local stream. Will only have something if it was written
// to by the remote
XLinkError_t XLinkReadData(streamId_t streamId, streamPacketDesc_t** packet);
XLinkError_t XLinkReadDataWithTimeOut(streamId_t streamId, streamPacketDesc_t** packet, unsigned int timeout);

// Release data from stream - This should be called after ReadData
XLinkError_t XLinkReleaseData(streamId_t streamId);

//Read fill level
XLinkError_t XLinkGetFillLevel(streamId_t streamId, int isRemote, int* fillLevel);

// Boot the remote (This is intended as an interface to boot the Myriad
// from PC)
XLinkError_t XLinkBootRemote(const char* deviceName, const char* binaryPath);

// Reset the remote
XLinkError_t XLinkResetRemote(linkId_t id);

// Close all and release all memory
XLinkError_t XLinkResetAll();

// Profiling funcs - keeping them global for now
XLinkError_t XLinkProfStart();
XLinkError_t XLinkProfStop();
XLinkError_t XLinkProfPrint();

XLinkError_t XLinkWriteGraphData(streamId_t streamId, const uint8_t* buffer, int size);

#ifdef __cplusplus
}
#endif

#endif
