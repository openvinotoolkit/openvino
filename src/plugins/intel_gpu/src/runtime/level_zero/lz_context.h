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
    // template <typename T>
    void* createBuffer(size_t elem_count, int offset);
    void readBuffer(std::vector<uint32_t> &hostDst, void *devSrc, size_t size);
    // template <typename T>
    // void readBuffer(std::vector<T> &hostDst, void *devSrc, size_t size) {
    //     ze_result_t result;
    //     result = zeCommandListAppendMemoryCopy(command_list, hostDst.data(), devSrc, size, nullptr, 0, nullptr);
    //     CHECK_ZE_STATUS(result, "zeCommandListAppendMemoryCopy");
    //     result = zeCommandListClose(command_list);
    //     CHECK_ZE_STATUS(result, "zeCommandListClose");
    //     result = zeCommandQueueExecuteCommandLists(command_queue, 1, &command_list, nullptr);
    //     CHECK_ZE_STATUS(result, "zeCommandQueueExecuteCommandLists");
    //     result = zeCommandQueueSynchronize(command_queue, UINT64_MAX);
    //     CHECK_ZE_STATUS(result, "zeCommandQueueSynchronize");
    //     result = zeCommandListReset(command_list);
    //     CHECK_ZE_STATUS(result, "zeCommandListReset");
    // }

    // void readBuffer(std::vector<uint32_t> &hostDst, void *devSrc, size_t size);
    void writeBuffer(std::vector<uint32_t> hostSrc, void *devDst, size_t size);
    void runKernel(const char *spvFile, const char *funcName, void *remoteBuf, void *devBuf, size_t elemCount);
    void *createFromHandle(uint64_t handle, size_t bufSize);
    void printBuffer(void* ptr, size_t count = 16);
    // void printBuffer(void* ptr, std::vector<T>& outBuf, size_t count = 16);
    // template <typename T>
    // void printBuffer(void *ptr, std::vector<T>& outBuf, size_t count = 16) {
    //     readBuffer(outBuf, ptr, count*sizeof(T));
    //     printf("The first %ld elements in level-zero ptr = %p are: \n", count, ptr);
    //     for (size_t i = 0; i < count; i++) {
    //         // printf("%d, ", outBuf[i]);
    //         std::cout << "[" << outBuf[i] << "] ";
    //         if (i && i % 16 == 0)
    //             printf("\n");
    //     }
    //     printf("\n");
    // }
};
