// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <new>
#include <stdlib.h>
#include <assert.h>

#include <iostream>
#include <string>
#include <sstream>
#include <utility>
#include <vector>
#include <fstream>
#include <memory>
#include <iomanip>

#include "level_zero/ze_api.h"

#define CHECK_ZE_STATUS(err, msg)                                                                                  \
    if (err < 0) {                                                                                                 \
        printf("ERROR: %s failed with err = 0x%08x, in function %s, line %d\n", msg, err, __FUNCTION__, __LINE__); \
        exit(0);                                                                                                   \
    } else {                                                                                                       \
        /*printf("INFO[ZE]: %s succeed\n", msg);    */                                                             \
    }

void queryP2P(ze_device_handle_t dev0, ze_device_handle_t dev1);

class lzContext {
private:
    const ze_device_type_t type = ZE_DEVICE_TYPE_GPU;
    ze_driver_handle_t pDriver = nullptr;
    ze_device_handle_t pDevice = nullptr;
    ze_context_handle_t context;
    ze_command_list_handle_t command_list = nullptr;
    ze_command_queue_handle_t command_queue = nullptr;
    ze_device_properties_t deviceProperties = {};

    ze_event_pool_handle_t eventPool = nullptr;
    ze_event_handle_t kernelTsEvent = nullptr;
    void *timestampBuffer = nullptr;

    const char *kernelSpvFile;
    const char *kernelFuncName;
    std::vector<char> kernelSpvBin;
    ze_module_handle_t module = nullptr;
    ze_kernel_handle_t function = nullptr;

    ze_device_handle_t findDevice(ze_driver_handle_t pDriver, ze_device_type_t type, uint32_t devIdx);
    void initTimeStamp();
    int readKernel();
    int initKernel();

public:
    lzContext(/* args */);
    ~lzContext();

    ze_device_handle_t device() { return pDevice; }

    int initZe(int devIdx);
    void* createBuffer(size_t elem_count, int offset);
    void readBuffer(std::vector<uint32_t> &hostDst, void *devSrc, size_t size);

    void writeBuffer(std::vector<uint32_t> hostSrc, void *devDst, size_t size);
    void runKernel(const char *spvFile, const char *funcName, void *remoteBuf, void *devBuf, size_t elemCount,
        int srcOffsetX, int srcOffsetY, int strideX, int strideY, int width);
    void *createFromHandle(uint64_t handle, size_t bufSize);
    void printBuffer(void* ptr, size_t count = 16);
};
