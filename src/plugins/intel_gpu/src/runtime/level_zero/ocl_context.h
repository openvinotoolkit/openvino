// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common.h"

class oclContext {
private:
    cl_platform_id platform_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    uint32_t device_idx = -1;

public:
    oclContext(/* args */);
    // oclContext(cl_device_id device);
    ~oclContext();

    cl_device_id device() { return device_; }
    cl_context context() { return context_; }
    cl_command_queue queue() { return queue_; }

    int get_device_idx(cl_device_id device_idx);
    void init(int devIdx);
    void *initUSM(size_t elem_count, int offset);
    void readUSM(void *ptr, std::vector<uint32_t> &outBuf, size_t size);
    void freeUSM(void *ptr);
    void runKernel(char *programFile, char *kernelName, void *ptr0, void *ptr1, size_t elemCount);

    cl_mem createBuffer(size_t size, const std::vector<float> &inbuf = std::vector<float>{});
    uint64_t deriveHandle(cl_mem clbuf);
    void readBuffer(cl_mem clbuf, std::vector<uint32_t> &outBuf, size_t size);
    void freeBuffer(cl_mem clbuf);
    void printBuffer(cl_mem clbuf, size_t count = 16);
};
