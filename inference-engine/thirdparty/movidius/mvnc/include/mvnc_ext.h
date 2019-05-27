#ifndef __NC_EXT_H_INCLUDED__
#define __NC_EXT_H_INCLUDED__
#include <mvnc.h>
#ifdef __cplusplus
extern "C"
{
#endif

typedef enum {
    NC_NORESET = 0,
    NC_RESET = 1
} ncInitReset_t;

/*
 * @brief Boot device with firmware without creating handler for it
 * @param devicePlatform Platform to boot
 * @param customFirmwareDir Path to directory with firmware to load. If NULL, use default
 */
MVNC_EXPORT_API ncStatus_t ncDeviceLoadFirmware(const ncDevicePlatform_t devicePlatform, const char* customFirmwareDir);
MVNC_EXPORT_API ncStatus_t ncDeviceLoadFirmwareWithPath(const char* unbooted_device_name, const char* fw_path);
MVNC_EXPORT_API ncStatus_t ncPlatformInit(ncInitReset_t reset);
MVNC_EXPORT_API ncStatus_t ncDeviceOpenBooted(struct ncDeviceHandle_t **deviceHandle, const char* deviceID);

MVNC_EXPORT_API ncStatus_t ncFifoWriteIonElem(struct ncFifoHandle_t* fifo, const void *inputTensor,
                          unsigned int *inputTensorLength, void *userParam);

MVNC_EXPORT_API ncStatus_t ncFifoReadIonElem(struct ncFifoHandle_t* fifo, int output_shared_fd,
                          unsigned int *outputDataLen, void **userParam);

/*
 * @brief Reset all devices
 */
MVNC_EXPORT_API ncStatus_t ncDeviceResetAll();

MVNC_EXPORT_API ncStatus_t ncGraphGetInfoSize(const void* graphFile, size_t graphFileLength, ncUserGetInfo_t option, void* data, unsigned int* dataLength);

MVNC_EXPORT_API ncStatus_t ncDeviceGetUnbootedName(struct ncDeviceHandle_t* deviceHandle, void* devAddr);

MVNC_EXPORT_API ncStatus_t ncDeviceGetId(struct ncDeviceHandle_t* deviceHandle, void* deviceId);

MVNC_EXPORT_API ncStatus_t ncDeviceHWReset(struct ncDeviceHandle_t* deviceHandle);

#ifdef __cplusplus
}
#endif

#endif  // __NC_EXT_H_INCLUDED__
