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

#include "oneapi/dnnl/dnnl.h"
#include "oneapi/dnnl/dnnl_sycl.hpp"

#include <memory>
#include <CL/cl.h>
#include <CL/sycl.hpp>

using namespace cl::sycl;

namespace dnnl {
class sycl_stream_test : public ::testing::TestWithParam<engine::kind> {
protected:
    virtual void SetUp() {
        if (engine::get_count(engine::kind::cpu) > 0) {
            cpu_eng = engine(engine::kind::cpu, 0);
        }
        if (engine::get_count(engine::kind::gpu) > 0) {
            gpu_eng = engine(engine::kind::gpu, 0);
        }
    }

    bool has(engine::kind eng_kind) const {
        switch (eng_kind) {
            case engine::kind::cpu: return bool(cpu_eng);
            case engine::kind::gpu: return bool(gpu_eng);
            default: assert(!"Not expected");
        }
        return false;
    }

    engine get_engine(engine::kind eng_kind) const {
        switch (eng_kind) {
            case engine::kind::cpu: return cpu_eng;
            case engine::kind::gpu: return gpu_eng;
            default: assert(!"Not expected");
        }
        return {};
    }

    device get_device(engine::kind eng_kind) const {
        switch (eng_kind) {
            case engine::kind::cpu: return sycl_interop::get_device(cpu_eng);
            case engine::kind::gpu: return sycl_interop::get_device(gpu_eng);
            default: assert(!"Not expected");
        }
        return {};
    }

    context get_context(engine::kind eng_kind) const {
        switch (eng_kind) {
            case engine::kind::cpu: return sycl_interop::get_context(cpu_eng);
            case engine::kind::gpu: return sycl_interop::get_context(gpu_eng);
            default: assert(!"Not expected");
        }
        return context();
    }

    engine cpu_eng;
    engine gpu_eng;
};

TEST_P(sycl_stream_test, Create) {
    engine::kind kind = GetParam();
    SKIP_IF(!has(kind), "Device not found.");

    stream s(get_engine(kind));

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
    if (kind == engine::kind::cpu) {
        EXPECT_ANY_THROW(sycl_interop::get_queue(s));
        return;
    }
#endif
    queue sycl_queue = sycl_interop::get_queue(s);

    auto queue_dev = sycl_queue.get_device();
    auto queue_ctx = sycl_queue.get_context();

    EXPECT_EQ(get_device(kind), queue_dev);
    EXPECT_EQ(get_context(kind), queue_ctx);
}

TEST_P(sycl_stream_test, BasicInterop) {
    engine::kind kind = GetParam();
    SKIP_IF(!has(kind), "Device not found.");

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
    if (kind == engine::kind::cpu) {
        cl::sycl::queue dummy;
        EXPECT_ANY_THROW(sycl_interop::make_stream(get_engine(kind), dummy));
        return;
    }
#endif
    queue interop_queue(get_context(kind), get_device(kind));
    stream s = sycl_interop::make_stream(get_engine(kind), interop_queue);

    EXPECT_EQ(interop_queue, sycl_interop::get_queue(s));
}

TEST_P(sycl_stream_test, InteropIncompatibleQueue) {
    engine::kind kind = GetParam();
    SKIP_IF(!has(engine::kind::cpu) || !has(engine::kind::gpu),
            "CPU or GPU device not found.");

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
    SKIP_IF(true, "Skip this test for classic CPU runtime");
#endif

    auto other_kind = (kind == engine::kind::gpu) ? engine::kind::cpu
                                                  : engine::kind::gpu;
    queue interop_queue(get_context(other_kind), get_device(other_kind));

    catch_expected_failures(
            [&] { sycl_interop::make_stream(get_engine(kind), interop_queue); },
            true, dnnl_invalid_arguments);
}

// TODO: Enable the test below after sycl_stream_t is fixed to not reuse the
// service stream. Now it ignores the input stream flags and reuses the service
// stream which is constructed without any flags.
#if 0
TEST_P(sycl_stream_test, Flags) {
    engine::kind kind = GetParam();
    SKIP_IF(!has(kind), "Device not found.");

    stream in_order_stream(get_engine(kind), stream::flags::in_order);
    auto in_order_queue = sycl_interop::get_queue(in_order_stream);
    EXPECT_TRUE(in_order_queue.is_in_order());

    stream out_of_order_stream(get_engine(kind), stream::flags::out_of_order);
    auto out_of_order_queue = sycl_interop::get_queue(out_of_order_stream);
    EXPECT_TRUE(!out_of_order_queue.is_in_order());
}
#endif

namespace {
struct PrintToStringParamName {
    template <class ParamType>
    std::string operator()(
            const ::testing::TestParamInfo<ParamType> &info) const {
        switch (info.param) {
            case engine::kind::cpu: return "cpu";
            case engine::kind::gpu: return "gpu";
            default: assert(!"Not expected");
        }
        return {};
    }
};
} // namespace

INSTANTIATE_TEST_SUITE_P(AllEngineKinds, sycl_stream_test,
        ::testing::Values(engine::kind::cpu, engine::kind::gpu),
        PrintToStringParamName());

} // namespace dnnl
