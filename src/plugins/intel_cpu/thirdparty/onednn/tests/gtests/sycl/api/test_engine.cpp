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

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"

#include <memory>
#include <CL/sycl.hpp>

using namespace cl::sycl;

namespace dnnl {

enum class dev_kind { cpu, cpu_only, gpu, gpu_only, host };
enum class ctx_kind { cpu, cpu_only, gpu, gpu_only, host };

namespace {

std::string to_string(dev_kind kind) {
    static const char *strs[] = {"CPU", "CPU-only", "GPU", "GPU-only", "Host"};
    return strs[static_cast<int>(kind)];
}

std::string to_string(ctx_kind kind) {
    static const char *strs[] = {"CPU", "CPU-only", "GPU", "GPU-only", "Host"};
    return strs[static_cast<int>(kind)];
}

} // namespace

struct sycl_engine_test_params {
    dev_kind adev_kind;
    ctx_kind actx_kind;
    dnnl_status_t expected_status;
};

class sycl_engine_test
    : public ::testing::TestWithParam<sycl_engine_test_params> {
protected:
    virtual void SetUp() {
        for (auto &plat : platform::get_platforms()) {
            for (auto &dev : plat.get_devices()) {
                if (dev.is_gpu()) {
                    if (!gpu_dev) {
                        gpu_dev.reset(new device(dev));
                        gpu_ctx.reset(new context(*gpu_dev));
                    }
                    if (!gpu_only_dev && !dev.is_cpu() && !dev.is_host()) {
                        gpu_only_dev.reset(new device(dev));
                        gpu_only_ctx.reset(new context(*gpu_only_dev));
                    }
                } else if (dev.is_cpu()) {
                    if (!cpu_dev) {
                        cpu_dev.reset(new device(dev));
                        cpu_ctx.reset(new context(*cpu_dev));
                    }
                    if (!cpu_only_dev && !dev.is_gpu() && !dev.is_host()) {
                        cpu_only_dev.reset(new device(dev));
                        cpu_only_ctx.reset(new context(*cpu_only_dev));
                    }
                } else if (dev.is_host()) {
                    if (!host_dev) {
                        host_dev.reset(new device(dev));
                        host_ctx.reset(new context(*host_dev));
                    }
                }
            }
        }
    }

    virtual void TearDown() {}

    std::unique_ptr<device> cpu_dev, cpu_only_dev;
    std::unique_ptr<device> gpu_dev, gpu_only_dev;
    std::unique_ptr<device> host_dev;
    std::unique_ptr<context> cpu_ctx, cpu_only_ctx;
    std::unique_ptr<context> gpu_ctx, gpu_only_ctx;
    std::unique_ptr<context> host_ctx;
};

TEST_P(sycl_engine_test, BasicInterop) {
    auto param = GetParam();
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_NONE
    SKIP_IF(param.adev_kind != dev_kind::gpu
                    && param.adev_kind != dev_kind::gpu_only,
            "Skip non-GPU engine kinds for GPU only configuration");
#endif

    device *dev_ptr = nullptr;
    switch (param.adev_kind) {
        case dev_kind::cpu: dev_ptr = cpu_dev.get(); break;
        case dev_kind::cpu_only: dev_ptr = cpu_only_dev.get(); break;
        case dev_kind::gpu: dev_ptr = gpu_dev.get(); break;
        case dev_kind::gpu_only: dev_ptr = gpu_only_dev.get(); break;
        case dev_kind::host: dev_ptr = host_dev.get(); break;
    }
    context *ctx_ptr = nullptr;
    switch (param.actx_kind) {
        case ctx_kind::cpu: ctx_ptr = cpu_ctx.get(); break;
        case ctx_kind::cpu_only: ctx_ptr = cpu_only_ctx.get(); break;
        case ctx_kind::gpu: ctx_ptr = gpu_ctx.get(); break;
        case ctx_kind::gpu_only: ctx_ptr = gpu_only_ctx.get(); break;
        case ctx_kind::host: ctx_ptr = host_ctx.get(); break;
    }

    SKIP_IF(!dev_ptr, to_string(param.adev_kind) + " device not found");
    SKIP_IF(!ctx_ptr, to_string(param.actx_kind) + " context not found");

    auto &dev = *dev_ptr;
    auto &ctx = *ctx_ptr;

    catch_expected_failures(
            [&]() {
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
                if (!impl::utils::one_of(param.adev_kind, dev_kind::gpu,
                            dev_kind::gpu_only)) {
                    engine eng(engine::kind::cpu, 0);
                    EXPECT_ANY_THROW(sycl_interop::make_engine(dev, ctx));
                    EXPECT_ANY_THROW(sycl_interop::get_device(eng));
                    EXPECT_ANY_THROW(sycl_interop::get_context(eng));

                    return;
                }
#endif
                auto eng = sycl_interop::make_engine(dev, ctx);
                if (param.expected_status != dnnl_success) {
                    FAIL() << "Success not expected";
                }

                EXPECT_EQ(sycl_interop::get_device(eng), dev);
                EXPECT_EQ(sycl_interop::get_context(eng), ctx);
            },
            param.expected_status != dnnl_success, param.expected_status);
}

TEST(sycl_engine_test, HostDevice) {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_NONE
    SKIP_IF(true, "Skip test for host device for GPU only configuration");
#endif

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
    SKIP_IF(true, "Skip this test for classic CPU runtime");
#endif

    device dev(host_selector {});
    context ctx(dev);

    auto eng = sycl_interop::make_engine(dev, ctx);

    memory::dims tz = {2, 3, 4, 5};
    memory::desc mem_d(tz, memory::data_type::f32, memory::format_tag::nchw);
    auto mem = test::make_memory(mem_d, eng);

    {
        auto *ptr = mem.map_data<float>();
        GTEST_EXPECT_NE(ptr, nullptr);
        for (size_t i = 0; i < mem_d.get_size() / sizeof(float); ++i)
            ptr[i] = float(i) * (i % 2 == 0 ? 1 : -1);
        mem.unmap_data(ptr);
    }

    auto eltwise_d = eltwise_forward::desc(
            prop_kind::forward, algorithm::eltwise_relu, mem_d, 0.0f);
    auto eltwise = eltwise_forward({eltwise_d, eng});

    stream s(eng);
    eltwise.execute(s, {{DNNL_ARG_SRC, mem}, {DNNL_ARG_DST, mem}});
    s.wait();

    {
        auto *ptr = mem.map_data<float>();
        GTEST_EXPECT_NE(ptr, nullptr);
        for (size_t i = 0; i < mem_d.get_size() / sizeof(float); ++i)
            ASSERT_EQ(ptr[i], (i % 2 == 0 ? i : 0));
        mem.unmap_data(ptr);
    }
}

TEST_P(sycl_engine_test, SubDevice) {
    auto param = GetParam();
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_NONE
    SKIP_IF(param.adev_kind != dev_kind::gpu
                    && param.adev_kind != dev_kind::gpu_only,
            "Skip non-GPU engine kinds for GPU only configuration");
#endif

    SKIP_IF(param.expected_status != dnnl_success,
            "Don't test for failed scenarios");
    SKIP_IF(!gpu_dev.get(), "Non GPU doesn't support sub-devices");

    auto &dev = *gpu_dev.get();
    auto max_sub_devices
            = dev.get_info<info::device::partition_max_sub_devices>();
    SKIP_IF(max_sub_devices < 2, "This GPU doesn't support sub-devices");

    auto sub_dev = dev.create_sub_devices<
            info::partition_property::partition_by_affinity_domain>(
            info::partition_affinity_domain::next_partitionable);
    context sub_ctx(sub_dev);

    catch_expected_failures(
            [&]() {
                for (const auto &sub_dev_i : sub_dev) {
                    engine eng;
                    ASSERT_NO_THROW(eng
                            = sycl_interop::make_engine(sub_dev_i, sub_ctx));

                    memory::dims tz = {2, 3, 4, 5};
                    memory::desc mem_d(tz, memory::data_type::f32,
                            memory::format_tag::nchw);
                    auto mem = test::make_memory(mem_d, eng);

                    auto eltwise_d = eltwise_forward::desc(prop_kind::forward,
                            algorithm::eltwise_relu, mem_d, 0.0f);
                    auto eltwise = eltwise_forward({eltwise_d, eng});

                    stream s(eng);
                    eltwise.execute(
                            s, {{DNNL_ARG_SRC, mem}, {DNNL_ARG_DST, mem}});
                    s.wait();
                }
            },
            param.expected_status != dnnl_success, param.expected_status);
}

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_NONE
TEST_P(sycl_engine_test, gpu_only) {
    device dev(cpu_selector {});
    context ctx(dev);
    EXPECT_ANY_THROW(sycl_interop::make_engine(dev, ctx));
}
#endif

INSTANTIATE_TEST_SUITE_P(Simple, sycl_engine_test,
        ::testing::Values(sycl_engine_test_params {dev_kind::gpu, ctx_kind::gpu,
                                  dnnl_success},
                sycl_engine_test_params {
                        dev_kind::cpu, ctx_kind::cpu, dnnl_success},
                sycl_engine_test_params {
                        dev_kind::host, ctx_kind::host, dnnl_success}));

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
INSTANTIATE_TEST_SUITE_P(InvalidArgs, sycl_engine_test,
        ::testing::Values(sycl_engine_test_params {dev_kind::cpu_only,
                ctx_kind::gpu_only, dnnl_invalid_arguments}));
#endif

} // namespace dnnl
