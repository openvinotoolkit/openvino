/*
* Copyright 2017-2019 Intel Corporation.
* The source code, information and material ("Material") contained herein is
* owned by Intel Corporation or its suppliers or licensors, and title to such
* Material remains with Intel Corporation or its suppliers or licensors.
* The Material contains proprietary information of Intel or its suppliers and
* licensors. The Material is protected by worldwide copyright laws and treaty
* provisions.
* No part of the Material may be used, copied, reproduced, modified, published,
* uploaded, posted, transmitted, distributed or disclosed in any way without
* Intel's prior express written permission. No license under any patent,
* copyright or other intellectual property rights in the Material is granted to
* or conferred upon you, either expressly, by implication, inducement, estoppel
* or otherwise.
* Any license under such intellectual property rights must be express and
* approved by Intel in writing.
*/

#include "mvnc_data.h"
#include "mvnc_tool.h"

#define MVLOG_UNIT_NAME ncTool
#include "XLinkLog.h"
#include "XLinkStringUtils.h"
#include "XLink.h"

#include <string.h>
#include <stdio.h>

XLinkProtocol_t convertProtocolToXlink(
    const ncDeviceProtocol_t ncProtocol) {
    switch (ncProtocol) {
        case NC_ANY_PROTOCOL: return X_LINK_ANY_PROTOCOL;
        case NC_USB:          return X_LINK_USB_VSC;
        case NC_PCIE:         return X_LINK_PCIE;
        default:              return X_LINK_ANY_PROTOCOL;
    }
}

ncDeviceProtocol_t convertProtocolToNC(
    const XLinkProtocol_t xLinkProtocol) {
    switch (xLinkProtocol) {
        case X_LINK_ANY_PROTOCOL:   return NC_ANY_PROTOCOL;
        case X_LINK_USB_VSC:        return NC_USB;
        case X_LINK_PCIE:           return NC_PCIE;
        default:
            mvLog(MVLOG_WARN, "This convertation not supported, set to ANY_PROTOCOL");
            return NC_ANY_PROTOCOL;
    }
}

XLinkPlatform_t convertPlatformToXlink(
    const ncDevicePlatform_t ncProtocol) {
    switch (ncProtocol) {
        case NC_ANY_PLATFORM: return X_LINK_ANY_PLATFORM;
        case NC_MYRIAD_2:     return X_LINK_MYRIAD_2;
        case NC_MYRIAD_X:     return X_LINK_MYRIAD_X;
        default:           return X_LINK_ANY_PLATFORM;
    }
}

ncDevicePlatform_t convertPlatformToNC(
    const XLinkPlatform_t xLinkProtocol) {
    switch (xLinkProtocol) {
        case X_LINK_ANY_PLATFORM:   return NC_ANY_PLATFORM;
        case X_LINK_MYRIAD_2:       return NC_MYRIAD_2;
        case X_LINK_MYRIAD_X:       return NC_MYRIAD_X;
        default:
            mvLog(MVLOG_WARN, "This convertation not supported, set to NC_ANY_PLATFORM");
            return NC_ANY_PLATFORM;
    }
}

int copyNcDeviceDescrToXLink(const struct ncDeviceDescr_t *in_ncDeviceDesc,
                                    deviceDesc_t *out_deviceDesc) {
    CHECK_HANDLE_CORRECT(in_ncDeviceDesc);
    CHECK_HANDLE_CORRECT(out_deviceDesc);

    out_deviceDesc->protocol = convertProtocolToXlink(in_ncDeviceDesc->protocol);
    out_deviceDesc->platform = convertPlatformToXlink(in_ncDeviceDesc->platform);
    mv_strncpy(out_deviceDesc->name, XLINK_MAX_NAME_SIZE, in_ncDeviceDesc->name, XLINK_MAX_NAME_SIZE - 1);

    return NC_OK;
}

int copyXLinkDeviceDescrToNc(const deviceDesc_t *in_DeviceDesc,
                                    struct ncDeviceDescr_t *out_ncDeviceDesc) {
    CHECK_HANDLE_CORRECT(in_DeviceDesc);
    CHECK_HANDLE_CORRECT(out_ncDeviceDesc);

    out_ncDeviceDesc->protocol = convertProtocolToNC(in_DeviceDesc->protocol);
    out_ncDeviceDesc->platform = convertPlatformToNC(in_DeviceDesc->platform);
    mv_strncpy(out_ncDeviceDesc->name, XLINK_MAX_NAME_SIZE, in_DeviceDesc->name, XLINK_MAX_NAME_SIZE - 1);

    return NC_OK;
}

static ncStatus_t readFirmware(const char* binaryPath, char** out_firmware, size_t* out_length) {
    CHECK_HANDLE_CORRECT(binaryPath);
    CHECK_HANDLE_CORRECT(out_firmware);
    CHECK_HANDLE_CORRECT(out_length);

    *out_firmware = NULL;
    *out_length = 0;
    FILE* firmwareFile = fopen(binaryPath, "rb");

    if(firmwareFile == NULL) {
        mvLog(MVLOG_ERROR, "Fail to open file by path %s", binaryPath);
        return NC_ERROR;
    }

    fseek(firmwareFile, 0, SEEK_END);
    long fileSize = ftell(firmwareFile);
    if(fileSize <= 0) {
        mvLog(MVLOG_ERROR, "Fail to get file size or firmware is empty. fileSize = %ld", fileSize);
        fclose(firmwareFile);
        return NC_ERROR;
    }
    rewind(firmwareFile);

    char* firmware = malloc(fileSize * sizeof(char));
    if(firmware == NULL) {
        mvLog(MVLOG_ERROR, "Fail to allocate memory for firmware");
        fclose(firmwareFile);
        return NC_ERROR;
    }

    size_t readCount = fread(firmware, sizeof(char), fileSize, firmwareFile);
    if(readCount != fileSize)
    {
        mvLog(MVLOG_ERROR, "Fail to read firmware by path %s. readCount = %zu", binaryPath, readCount);
        fclose(firmwareFile);
        free(firmware);
        return NC_ERROR;
    }

    fclose(firmwareFile);

    *out_firmware = firmware;
    *out_length = (size_t )fileSize;

    return NC_OK;
}

static ncStatus_t patchFirmware(char **firmware, size_t *length, size_t commandLocationId,
                                const char command[], const size_t commandSize, const char value) {
    CHECK_HANDLE_CORRECT(firmware);
    CHECK_HANDLE_CORRECT(length);
    CHECK_HANDLE_CORRECT(command);

    char* currFirmware = *firmware;
    size_t currLength = *length;

    size_t patchedFirmwareLen = currLength + commandSize + 1;
    char* patchedFirmware = malloc(patchedFirmwareLen);
    if(patchedFirmware == NULL) {
        mvLog(MVLOG_ERROR, "Fail to allocate memory for patched firmware");
        return NC_ERROR;
    }

    memcpy(patchedFirmware, currFirmware, commandLocationId);

    memcpy(patchedFirmware + commandLocationId,
           command, commandSize);
    memcpy(patchedFirmware + commandLocationId + commandSize,
           &value, 1);

    size_t currentPos = commandLocationId + commandSize + 1;
    size_t tailSize = currLength - commandLocationId;
    memcpy(patchedFirmware + currentPos,
           currFirmware + commandLocationId, tailSize);

    free(currFirmware);
    *firmware = patchedFirmware;
    *length = patchedFirmwareLen;

    return NC_OK;
}

// 0x98 the write command for 8bit
// {0x00, 0x00, 0x20, 0x80} == 0x80200000 the address of watchdog flag
// 0x01 flag value
const char g_setWdSwitchCommandMX[] = {0x98, 0x00, 0x00, 0x20, 0x80};
const char g_executeCommand = 0xa4;

static ncStatus_t patchSetWdSwitchCommand(char **firmware, size_t *length, const char wdEnable) {
    CHECK_HANDLE_CORRECT(firmware);
    CHECK_HANDLE_CORRECT(length);

    char* currFirmware = *firmware;
    size_t currLength = *length;
    size_t executeCommandIdx = 0;
    char executeCommandFound = 0;
    size_t i = 0;

    for (i = currLength - 1; i >= 0; i--) {
        if(currFirmware[i] == g_executeCommand) {
            executeCommandIdx = i;
            executeCommandFound = 1;
            break;
        }
    }

    if(!executeCommandFound) {
        mvLog(MVLOG_WARN, "Fail to find execute command");
        return NC_ERROR;
    }
    return patchFirmware(firmware, length, executeCommandIdx,
                         g_setWdSwitchCommandMX, sizeof(g_setWdSwitchCommandMX), wdEnable);
}

// 0x98 the write command for 8bit
// {0x00, 0x0c, 0x20, 0x70} == 0x70200c00 the address of memory type for ddrInit application
const char g_setMemTypeCommandMX[] = {0x98, 0x00, 0x0c, 0x20, 0x70};
const char g_callCommand[] = {0xba, 0x78, 0xe9, 0x00, 0x70};

static ncStatus_t patchSetMemTypeCommand(char **firmware, size_t *length, const char memType) {
    CHECK_HANDLE_CORRECT(firmware);
    CHECK_HANDLE_CORRECT(length);

    char* currFirmware = *firmware;
    size_t currLength = *length;
    size_t callCommandIdx = 0;
    char callCommandFound = 0;
    size_t i = 0;
    size_t callCommandLen = sizeof(g_callCommand);

    for (i = 0; i < currLength; i++) {
        size_t j = 0;
        for (j = 0; j < callCommandLen; j++) {
            if(currFirmware[i + j] != g_callCommand[j]) {
                break;
            }
        }

        if(j == callCommandLen) {
            callCommandIdx = i;
            callCommandFound = 1;
        }
    }

    if(!callCommandFound) {
        mvLog(MVLOG_WARN, "Fail to find call command");
        return NC_ERROR;
    }
    return patchFirmware(firmware, length, callCommandIdx,
                         g_setMemTypeCommandMX, sizeof(g_setMemTypeCommandMX), memType);
}

ncStatus_t bootDevice(deviceDesc_t* deviceDescToBoot,
                      const char* mv_cmd_file_path, const bootOptions_t bootOptions) {

    CHECK_HANDLE_CORRECT(deviceDescToBoot);

    char* firmware = NULL;
    size_t length = 0;
    ncStatus_t sc = readFirmware(mv_cmd_file_path, &firmware, &length);
    if(sc) {
        mvLog(MVLOG_ERROR, "Fail to read firmware by path %s. sc = %d", mv_cmd_file_path, sc);
        return sc;
    }

    if(deviceDescToBoot->platform == X_LINK_MYRIAD_X) {
        if(deviceDescToBoot->protocol != X_LINK_PCIE) {
            sc = patchSetWdSwitchCommand(&firmware, &length, bootOptions.wdEnable);
            if(sc) {
                mvLog(MVLOG_WARN, "Fail to patch \"Set wd switch value\" command for firmware sc = %d", sc);
            }
            
            sc = patchSetMemTypeCommand(&firmware, &length, bootOptions.memType);
            if(sc) {
                mvLog(MVLOG_WARN, "Fail to patch \"Set memory type\" command for firmware sc = %d", sc);
            }
        }       
    }

    XLinkError_t rc = XLinkBootFirmware(deviceDescToBoot, firmware, (unsigned long)length);
    free(firmware);

    if(rc) {
        return NC_ERROR;
    }

    return NC_OK;
}
