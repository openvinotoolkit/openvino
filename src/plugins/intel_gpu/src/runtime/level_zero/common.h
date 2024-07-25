// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <iostream>

#include <CL/cl.h>
#include <CL/cl_ext.h>

#define CL_MEM_ALLOCATION_HANDLE_INTEL 0x10050

static std::map<int, std::string> oclErrorCode = {
    {0, "CL_SUCCESS"},
    {-1, "CL_DEVICE_NOT_FOUND"},
    {-2, "CL_DEVICE_NOT_AVAILABLE"},
    {-3, "CL_COMPILER_NOT_AVAILABLE"},
    {-4, "CL_MEM_OBJECT_ALLOCATION_FAILURE"},
    {-5, "CL_OUT_OF_RESOURCES"},
    {-6, "CL_OUT_OF_HOST_MEMORY"},
    {-7, "CL_PROFILING_INFO_NOT_AVAILABLE"},
    {-8, "CL_MEM_COPY_OVERLAP"},
    {-9, "CL_IMAGE_FORMAT_MISMATCH"},
    {-10, "CL_IMAGE_FORMAT_NOT_SUPPORTED"},
    {-11, "CL_BUILD_PROGRAM_FAILURE"},
    {-12, "CL_MAP_FAILURE"},
    {-13, "CL_MISALIGNED_SUB_BUFFER_OFFSET"},
    {-14, "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"},
    {-15, "CL_COMPILE_PROGRAM_FAILURE"},
    {-16, "CL_LINKER_NOT_AVAILABLE"},
    {-17, "CL_LINK_PROGRAM_FAILURE"},
    {-18, "CL_DEVICE_PARTITION_FAILED"},
    {-19, "CL_KERNEL_ARG_INFO_NOT_AVAILABLE"},
    {-30, "CL_INVALID_VALUE"},
    {-31, "CL_INVALID_DEVICE_TYPE"},
    {-32, "CL_INVALID_PLATFORM"},
    {-33, "CL_INVALID_DEVICE"},
    {-34, "CL_INVALID_CONTEXT"},
    {-35, "CL_INVALID_QUEUE_PROPERTIES"},
    {-36, "CL_INVALID_COMMAND_QUEUE"},
    {-37, "CL_INVALID_HOST_PTR"},
    {-38, "CL_INVALID_MEM_OBJECT"},
    {-39, "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"},
    {-40, "CL_INVALID_IMAGE_SIZE"},
    {-41, "CL_INVALID_SAMPLER"},
    {-42, "CL_INVALID_BINARY"},
    {-43, "CL_INVALID_BUILD_OPTIONS"},
    {-44, "CL_INVALID_PROGRAM"},
    {-45, "CL_INVALID_PROGRAM_EXECUTABLE"},
    {-46, "CL_INVALID_KERNEL_NAME"},
    {-47, "CL_INVALID_KERNEL_DEFINITION"},
    {-48, "CL_INVALID_KERNEL"},
    {-49, "CL_INVALID_ARG_INDEX"},
    {-50, "CL_INVALID_ARG_VALUE"},
    {-51, "CL_INVALID_ARG_SIZE"},
    {-52, "CL_INVALID_KERNEL_ARGS"},
    {-53, "CL_INVALID_WORK_DIMENSION"},
    {-54, "CL_INVALID_WORK_GROUP_SIZE"},
    {-55, "CL_INVALID_WORK_ITEM_SIZE"},
    {-56, "CL_INVALID_GLOBAL_OFFSET"},
    {-57, "CL_INVALID_EVENT_WAIT_LIST"},
    {-58, "CL_INVALID_EVENT"},
    {-59, "CL_INVALID_OPERATION"},
    {-60, "CL_INVALID_GL_OBJECT"},
    {-61, "CL_INVALID_BUFFER_SIZE"},
    {-62, "CL_INVALID_MIP_LEVEL"},
    {-63, "CL_INVALID_GLOBAL_WORK_SIZE"},
    {-64, "CL_INVALID_PROPERTY"},
    {-65, "CL_INVALID_IMAGE_DESCRIPTOR"},
    {-66, "CL_INVALID_COMPILER_OPTIONS"},
    {-67, "CL_INVALID_LINKER_OPTIONS"},
    {-68, "CL_INVALID_DEVICE_PARTITION_COUNT"},
    {-69, "CL_INVALID_PIPE_SIZE"},
    {-70, "CL_INVALID_DEVICE_QUEUE"},
    {-71, "CL_INVALID_SPEC_ID"},
    {-72, "CL_MAX_SIZE_RESTRICTION_EXCEEDED"},
};

static std::map<int, const char*> oclChannelOrder = {
    {0x10B0, "CL_R"},
    {0x10B1, "CL_A"},
    {0x10B2, "CL_RG"},
    {0x10B3, "CL_RA"},
    {0x10B4, "CL_RGB"},
    {0x10B5, "CL_RGBA"},
    {0x10B6, "CL_BGRA"},
    {0x10B7, "CL_ARGB"},
    {0x10B8, "CL_INTENSITY"},
    {0x10B9, "CL_LUMINANCE"},
    {0x10BA, "CL_Rx"},
    {0x10BB, "CL_RGx"},
    {0x10BC, "CL_RGBx"},
    {0x10BD, "CL_DEPTH"},
    {0x10BE, "CL_DEPTH_STENCIL"},
    {0x10BF, "CL_sRGB"},
    {0x10C0, "CL_sRGBx"},
    {0x10C1, "CL_sRGBA"},
    {0x10C2, "CL_sBGRA"},
    {0x10C3, "CL_ABGR"},
};

static std::map<int, const char*> oclChannelType = {
    {0x10D0, "CL_SNORM_INT8"},
    {0x10D1, "CL_SNORM_INT16"},
    {0x10D2, "CL_UNORM_INT8"},
    {0x10D3, "CL_UNORM_INT16"},
    {0x10D4, "CL_UNORM_SHORT_565"},
    {0x10D5, "CL_UNORM_SHORT_555"},
    {0x10D6, "CL_UNORM_INT_101010"},
    {0x10D7, "CL_SIGNED_INT8"},
    {0x10D8, "CL_SIGNED_INT16"},
    {0x10D9, "CL_SIGNED_INT32"},
    {0x10DA, "CL_UNSIGNED_INT8"},
    {0x10DB, "CL_UNSIGNED_INT16"},
    {0x10DC, "CL_UNSIGNED_INT32"},
    {0x10DD, "CL_HALF_FLOAT"},
    {0x10DE, "CL_FLOAT"},
    {0x10DF, "CL_UNORM_INT24"},
    {0x10E0, "CL_UNORM_INT_101010_2"},
};

#define CHECK_OCL_ERROR(err, msg) \
    if (err < 0) { \
        std::string errstr = (oclErrorCode.find(err) != oclErrorCode.end()) ? oclErrorCode[err] : "Unknown"; \
        printf("ERROR: oclContext::%s, line = %d, %s! err = %d (%s)\n", __FUNCTION__, __LINE__, msg, err, errstr.c_str()); \
    }

#define CHECK_OCL_ERROR_RETURN(err, msg, ret) \
    if (err < 0) { \
        std::string errstr = (oclErrorCode.find(err) != oclErrorCode.end()) ? oclErrorCode[err] : "Unknown"; \
        printf("ERROR: oclContext::%s, line = %d, %s! err = %d (%s)\n", __FUNCTION__, __LINE__, msg, err, errstr.c_str()); \
        return ret; \
    }

#define CHECK_OCL_ERROR_EXIT(err, msg) \
    if (err < 0) { \
        std::string errstr = (oclErrorCode.find(err) != oclErrorCode.end()) ? oclErrorCode[err] : "Unknown"; \
        printf("ERROR: oclContext::%s, line = %d, %s! err = %d (%s)\n", __FUNCTION__, __LINE__, msg, err, errstr.c_str()); \
        exit(1); \
    }

#define INTEL_GFX_VENDOR_ID 0x8086
typedef enum {
    FORCE_ADAPTER_UNKNOWN = 0,
    FORCE_ADAPTER_DISCRETE = 1,
    FORCE_ADAPTER_INTEGRATE = 2,
    FORCE_ADAPTER_CARD0 = 3
} ForceAdapterType;

const unsigned int short_discrete_devices[] = {
    0x02,  // ATS
    0x49,  // DG1
    0x56,  // DG2
    0x4F   // DG2 Val-Only
};

