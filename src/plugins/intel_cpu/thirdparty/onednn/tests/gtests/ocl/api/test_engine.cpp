/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl_ocl.h"
#include "oneapi/dnnl/dnnl_ocl.hpp"

#include <string>
#include <CL/cl.h>

extern "C" bool dnnl_impl_gpu_mayiuse_ngen_kernels(dnnl_engine_t engine);

namespace dnnl {
namespace {

enum class dev_kind { null, cpu, gpu };
enum class ctx_kind { null, cpu, gpu };

} // namespace

struct ocl_engine_test_t_params_t {
    dev_kind adev_kind;
    ctx_kind actx_kind;
    dnnl_status_t expected_status;
};

class ocl_engine_test_t
    : public ::testing::TestWithParam<ocl_engine_test_t_params_t> {
protected:
    void SetUp() override {
        gpu_ocl_dev = find_ocl_device(CL_DEVICE_TYPE_GPU);
        cpu_ocl_dev = find_ocl_device(CL_DEVICE_TYPE_CPU);

        cl_int err;
        if (gpu_ocl_dev) {
            gpu_ocl_ctx = clCreateContext(
                    nullptr, 1, &gpu_ocl_dev, nullptr, nullptr, &err);
            TEST_OCL_CHECK(err);
        }

        if (cpu_ocl_dev) {
            cpu_ocl_ctx = clCreateContext(
                    nullptr, 1, &cpu_ocl_dev, nullptr, nullptr, &err);
            TEST_OCL_CHECK(err);
        }
    }

    void TearDown() override {
        if (gpu_ocl_ctx) { clReleaseContext(gpu_ocl_ctx); }
        if (cpu_ocl_ctx) { clReleaseContext(cpu_ocl_ctx); }
    }

    cl_context gpu_ocl_ctx = nullptr;
    cl_device_id gpu_ocl_dev = nullptr;

    cl_context cpu_ocl_ctx = nullptr;
    cl_device_id cpu_ocl_dev = nullptr;
};

TEST_P(ocl_engine_test_t, BasicInteropC) {
    auto p = GetParam();
    cl_device_id ocl_dev = (p.adev_kind == dev_kind::gpu)
            ? gpu_ocl_dev
            : (p.adev_kind == dev_kind::cpu) ? cpu_ocl_dev : nullptr;

    cl_context ocl_ctx = (p.actx_kind == ctx_kind::gpu)
            ? gpu_ocl_ctx
            : (p.actx_kind == ctx_kind::cpu) ? cpu_ocl_ctx : nullptr;

    SKIP_IF(p.adev_kind != dev_kind::null && !ocl_dev,
            "Required OpenCL device not found.");
    SKIP_IF(p.actx_kind != ctx_kind::null && !ocl_ctx,
            "Required OpenCL context not found.");
    SKIP_IF(cpu_ocl_dev == gpu_ocl_dev
                    && (p.adev_kind == dev_kind::cpu
                            || p.actx_kind == ctx_kind::cpu),
            "OpenCL CPU-only device not found.");

    dnnl_engine_t eng = nullptr;
    dnnl_status_t s = dnnl_ocl_interop_engine_create(&eng, ocl_dev, ocl_ctx);

    ASSERT_EQ(s, p.expected_status);

    if (s == dnnl_success) {
        cl_device_id dev = nullptr;
        cl_context ctx = nullptr;

        DNNL_CHECK(dnnl_ocl_interop_get_device(eng, &dev));
        DNNL_CHECK(dnnl_ocl_interop_engine_get_context(eng, &ctx));

        ASSERT_EQ(dev, ocl_dev);
        ASSERT_EQ(ctx, ocl_ctx);

        cl_uint ref_count;
        TEST_OCL_CHECK(clGetContextInfo(ocl_ctx, CL_CONTEXT_REFERENCE_COUNT,
                sizeof(ref_count), &ref_count, nullptr));
        int i_ref_count = int(ref_count);
        ASSERT_EQ(i_ref_count, 2);

        DNNL_CHECK(dnnl_engine_destroy(eng));

        TEST_OCL_CHECK(clGetContextInfo(ocl_ctx, CL_CONTEXT_REFERENCE_COUNT,
                sizeof(ref_count), &ref_count, nullptr));
        i_ref_count = int(ref_count);
        ASSERT_EQ(i_ref_count, 1);

        // Check if device can be partitioned
        cl_uint max_sub_dev;
        TEST_OCL_CHECK(
                clGetDeviceInfo(ocl_dev, CL_DEVICE_PARTITION_MAX_SUB_DEVICES,
                        sizeof(max_sub_dev), &max_sub_dev, nullptr));
        if (max_sub_dev > 0) {
            std::vector<cl_device_id> sub_dev(max_sub_dev);

            cl_device_partition_property properties[3]
                    = {CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN,
                            CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE, 0};

            TEST_OCL_CHECK(clCreateSubDevices(
                    ocl_dev, properties, max_sub_dev, sub_dev.data(), nullptr));

            cl_int err;
            // Use only first sub-device to create a context (and engine).
            cl_context sub_ctx = clCreateContext(
                    nullptr, 1, sub_dev.data(), nullptr, nullptr, &err);

            TEST_OCL_CHECK(err);

            dnnl_engine_t sub_eng = nullptr;
            DNNL_CHECK(dnnl_ocl_interop_engine_create(
                    &sub_eng, sub_dev[0], sub_ctx));

            TEST_OCL_CHECK(
                    clGetDeviceInfo(sub_dev[0], CL_DEVICE_REFERENCE_COUNT,
                            sizeof(ref_count), &ref_count, nullptr));
            i_ref_count = int(ref_count);
            ASSERT_EQ(i_ref_count, 2);

            DNNL_CHECK(dnnl_engine_destroy(sub_eng));

            TEST_OCL_CHECK(
                    clGetDeviceInfo(sub_dev[0], CL_DEVICE_REFERENCE_COUNT,
                            sizeof(ref_count), &ref_count, nullptr));
            i_ref_count = int(ref_count);
            ASSERT_EQ(i_ref_count, 1);

            for (auto dev : sub_dev)
                clReleaseDevice(dev);
        }
    }
}

TEST_P(ocl_engine_test_t, BasicInteropCpp) {
    auto p = GetParam();
    cl_device_id ocl_dev = (p.adev_kind == dev_kind::gpu)
            ? gpu_ocl_dev
            : (p.adev_kind == dev_kind::cpu) ? cpu_ocl_dev : nullptr;

    cl_context ocl_ctx = (p.actx_kind == ctx_kind::gpu)
            ? gpu_ocl_ctx
            : (p.actx_kind == ctx_kind::cpu) ? cpu_ocl_ctx : nullptr;

    SKIP_IF(p.adev_kind != dev_kind::null && !ocl_dev,
            "Required OpenCL device not found.");
    SKIP_IF(p.actx_kind != ctx_kind::null && !ocl_ctx,
            "Required OpenCL context not found.");
    SKIP_IF(cpu_ocl_dev == gpu_ocl_dev
                    && (p.adev_kind == dev_kind::cpu
                            || p.actx_kind == ctx_kind::cpu),
            "OpenCL CPU-only device not found.");

    catch_expected_failures(
            [&]() {
                {
                    auto eng = ocl_interop::make_engine(ocl_dev, ocl_ctx);
                    if (p.expected_status != dnnl_success) {
                        FAIL() << "Success not expected";
                    }

                    cl_device_id dev = ocl_interop::get_device(eng);
                    cl_context ctx = ocl_interop::get_context(eng);
                    ASSERT_EQ(dev, ocl_dev);
                    ASSERT_EQ(ctx, ocl_ctx);

                    cl_uint ref_count;
                    TEST_OCL_CHECK(clGetContextInfo(ocl_ctx,
                            CL_CONTEXT_REFERENCE_COUNT, sizeof(ref_count),
                            &ref_count, nullptr));
                    int i_ref_count = int(ref_count);
                    ASSERT_EQ(i_ref_count, 2);
                }

                cl_uint ref_count;
                TEST_OCL_CHECK(
                        clGetContextInfo(ocl_ctx, CL_CONTEXT_REFERENCE_COUNT,
                                sizeof(ref_count), &ref_count, nullptr));
                int i_ref_count = int(ref_count);
                ASSERT_EQ(i_ref_count, 1);

                // Check if device can be partitioned
                cl_uint max_sub_dev;
                TEST_OCL_CHECK(clGetDeviceInfo(ocl_dev,
                        CL_DEVICE_PARTITION_MAX_SUB_DEVICES,
                        sizeof(max_sub_dev), &max_sub_dev, nullptr));

                if (max_sub_dev > 0) {

                    cl_device_partition_property properties[3] = {
                            CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN,
                            CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE, 0};

                    std::vector<cl_device_id> sub_dev(max_sub_dev);

                    TEST_OCL_CHECK(clCreateSubDevices(ocl_dev, properties,
                            max_sub_dev, sub_dev.data(), nullptr));

                    // Use only first sub-device to create a context (and
                    // engine).
                    cl_int err;
                    cl_context sub_ctx = clCreateContext(
                            nullptr, 1, sub_dev.data(), nullptr, nullptr, &err);
                    TEST_OCL_CHECK(err);
                    {
                        engine eng
                                = ocl_interop::make_engine(sub_dev[0], sub_ctx);
                        cl_uint ref_count;
                        TEST_OCL_CHECK(clGetDeviceInfo(sub_dev[0],
                                CL_DEVICE_REFERENCE_COUNT, sizeof(ref_count),
                                &ref_count, nullptr));
                        int i_ref_count = int(ref_count);
                        ASSERT_EQ(i_ref_count, 2);
                    }

                    cl_uint ref_count;
                    TEST_OCL_CHECK(clGetDeviceInfo(sub_dev[0],
                            CL_DEVICE_REFERENCE_COUNT, sizeof(ref_count),
                            &ref_count, nullptr));
                    int i_ref_count = int(ref_count);
                    ASSERT_EQ(i_ref_count, 1);

                    for (auto dev : sub_dev)
                        clReleaseDevice(dev);
                }
            },
            p.expected_status != dnnl_success, p.expected_status);
}

TEST_P(ocl_engine_test_t, BinaryKernels) {
    auto p = GetParam();
    cl_device_id ocl_dev = (p.adev_kind == dev_kind::gpu)
            ? gpu_ocl_dev
            : (p.adev_kind == dev_kind::cpu) ? cpu_ocl_dev : nullptr;

    cl_context ocl_ctx = (p.actx_kind == ctx_kind::gpu)
            ? gpu_ocl_ctx
            : (p.actx_kind == ctx_kind::cpu) ? cpu_ocl_ctx : nullptr;

    SKIP_IF(p.adev_kind != dev_kind::null && !ocl_dev,
            "Required OpenCL device not found.");
    SKIP_IF(p.actx_kind != ctx_kind::null && !ocl_ctx,
            "Required OpenCL context not found.");
    SKIP_IF(cpu_ocl_dev == gpu_ocl_dev
                    && (p.adev_kind == dev_kind::cpu
                            || p.actx_kind == ctx_kind::cpu),
            "OpenCL CPU-only device not found.");

    dnnl_engine_t eng = nullptr;
    dnnl_status_t s = dnnl_ocl_interop_engine_create(&eng, ocl_dev, ocl_ctx);

    ASSERT_EQ(s, p.expected_status);

//DNNL_ENABLE_MEM_DEBUG forces allocation fail, causing mayiuse to fail
#ifndef DNNL_ENABLE_MEM_DEBUG
    if (s == dnnl_success) {
        ASSERT_EQ(dnnl_impl_gpu_mayiuse_ngen_kernels(eng), true);
    }
#endif
}

INSTANTIATE_TEST_SUITE_P(Simple, ocl_engine_test_t,
        ::testing::Values(ocl_engine_test_t_params_t {
                dev_kind::gpu, ctx_kind::gpu, dnnl_success}));

INSTANTIATE_TEST_SUITE_P(InvalidArgs, ocl_engine_test_t,
        ::testing::Values(ocl_engine_test_t_params_t {dev_kind::cpu,
                                  ctx_kind::cpu, dnnl_invalid_arguments},
                ocl_engine_test_t_params_t {
                        dev_kind::gpu, ctx_kind::cpu, dnnl_invalid_arguments},
                ocl_engine_test_t_params_t {
                        dev_kind::null, ctx_kind::gpu, dnnl_invalid_arguments},
                ocl_engine_test_t_params_t {dev_kind::gpu, ctx_kind::null,
                        dnnl_invalid_arguments}));

} // namespace dnnl
