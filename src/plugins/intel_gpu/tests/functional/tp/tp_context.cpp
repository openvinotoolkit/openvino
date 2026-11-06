// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// What the GPU plugin adds to the collective contract (the shared L3 conformance tests,
// behavior/tp/collective_contract.hpp): its backend, its options, its limits and its messages.

#include <gtest/gtest.h>

#include <future>
#include <string>
#include <vector>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/runtime/tp/collective_contract.hpp"
#include "ov_ops/tp_collective.hpp"

namespace {

using ::testing::HasSubstr;
using ov::op::internal::ReduceOp;
using ov::op::internal::TPAllReduce;
using ov::tp::CollectiveGroup;
using ov::tp::IExchange;
using ov::tp::InProcessExchange;

constexpr size_t elements = 4096;

std::shared_ptr<ov::Model> make_allreduce_model(int rank,
                                                int world_size,
                                                int group_id = 0,
                                                ReduceOp reduce_op = ReduceOp::SUM) {
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, elements});
    auto reduce = std::make_shared<TPAllReduce>(x, rank, world_size, group_id, reduce_op);
    return std::make_shared<ov::Model>(ov::OutputVector{reduce}, ov::ParameterVector{x});
}

class GPUTPContextTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        try {
            gpu_count = ov::Core{}.get_property("GPU", ov::available_devices).size();
        } catch (const std::exception& e) {
            gpu_error = e.what();
        }
    }

    void SetUp() override {
        if (gpu_count < 2)
            GTEST_SKIP() << "need >= 2 GPUs, found " << gpu_count << gpu_error;
    }

    static ov::AnyMap rank_params(const std::vector<std::shared_ptr<IExchange>>& group,
                                  int rank,
                                  const std::string& backend = ov::tp::auto_backend) {
        return {{ov::tp::internal::device_group.name(), std::vector<std::string>{"GPU.0", "GPU.1"}},
                {ov::tp::internal::groups.name(),
                 std::map<int, CollectiveGroup>{{0, CollectiveGroup{group[static_cast<size_t>(rank)], backend}}}},
                {ov::tp::internal::communication_timeout_ms.name(), 5000}};
    }

    std::vector<ov::RemoteContext> create_contexts(const ov::AnyMap& extra = {},
                                                   const std::string& backend = ov::tp::auto_backend) {
        auto group = InProcessExchange::create_group(2, std::chrono::milliseconds{5000});
        std::vector<ov::RemoteContext> contexts;
        for (int rank = 0; rank < 2; ++rank) {
            auto params = rank_params(group, rank, backend);
            for (const auto& entry : extra)
                params[entry.first] = entry.second;
            contexts.push_back(core.create_context(rank == 0 ? "GPU.0" : "GPU.1", params));
        }
        return contexts;
    }

    void expect_two_ranks_reduce(std::vector<ov::RemoteContext>& contexts) {
        std::vector<ov::InferRequest> requests;
        for (int rank = 0; rank < 2; ++rank) {
            requests.push_back(core.compile_model(make_allreduce_model(rank, 2), contexts[static_cast<size_t>(rank)])
                                   .create_infer_request());
            ov::Tensor input(ov::element::f32, ov::Shape{1, elements});
            auto* data = input.data<float>();
            for (size_t i = 0; i < elements; ++i)
                data[i] = static_cast<float>(rank + 1) * static_cast<float>(i % 7);
            requests.back().set_input_tensor(input);
        }
        std::vector<std::future<void>> ranks;
        for (auto& request : requests)
            ranks.push_back(std::async(std::launch::async, [&request] {
                request.infer();
            }));
        for (auto& rank : ranks)
            rank.get();
        for (auto& request : requests) {
            const auto* data = request.get_output_tensor().data<const float>();
            for (size_t i = 0; i < elements; ++i)
                ASSERT_FLOAT_EQ(data[i], 3.0f * static_cast<float>(i % 7)) << "element " << i;
        }
    }

    std::vector<std::string> backends(const std::vector<std::string>& devices) {
        return core
            .get_property("GPU",
                          ov::tp::internal::collective_backends.name(),
                          {{ov::tp::internal::device_group.name(), devices}})
            .as<std::vector<std::string>>();
    }

    std::string support(const std::vector<std::string>& devices, const std::string& backend) {
        return core
            .get_property("GPU",
                          ov::tp::internal::collective_support.name(),
                          {{ov::tp::internal::device_group.name(), devices},
                           {ov::tp::internal::collective_backend.name(), backend}})
            .as<std::string>();
    }

    ov::Core core;
    static size_t gpu_count;
    static std::string gpu_error;
};

size_t GPUTPContextTest::gpu_count = 0;
std::string GPUTPContextTest::gpu_error;

TEST_F(GPUTPContextTest, TheBackendIsL0P2P) {
    EXPECT_EQ(backends({"GPU.0", "GPU.1"}), std::vector<std::string>{"L0_P2P"});
    EXPECT_EQ(support({"GPU.0", "GPU.1"}, "L0_P2P"), "");
    auto contexts = create_contexts({}, "L0_P2P");
    expect_two_ranks_reduce(contexts);
}

TEST_F(GPUTPContextTest, ExplainsWhatItDoesNotSupport) {
    EXPECT_THAT(support({"GPU.0", "GPU.1"}, "NO_SUCH_BACKEND"),
                HasSubstr("Unknown collective backend \"NO_SUCH_BACKEND\"; supported for GPU.0,GPU.1: L0_P2P"));
    EXPECT_TRUE(backends({"GPU.0", "GPU.0"}).empty());
    EXPECT_THAT(support({"GPU.0", "GPU.0"}, ov::tp::auto_backend), HasSubstr("appears in the group more than once"));
    EXPECT_TRUE(backends({"GPU.0", "GPU.99"}).empty());
    EXPECT_THAT(support({"GPU.0", "GPU.99"}, ov::tp::auto_backend), HasSubstr("does not exist"));
}

TEST_F(GPUTPContextTest, TakesItsCollectiveOptionsAsContextParameters) {
    auto contexts = create_contexts({{"GPU_TP_ENABLE_HALVING", false}});
    expect_two_ranks_reduce(contexts);
}

TEST_F(GPUTPContextTest, RejectsWhatTheContextDoesNotSupport) {
    auto group = InProcessExchange::create_group(2, std::chrono::milliseconds{5000});

    auto params = rank_params(group, 0);
    params[ov::tp::internal::channels.name()] = uint32_t{2};
    OV_EXPECT_THROW(core.create_context("GPU.0", params), ov::Exception, HasSubstr("only one channel"));

    params = rank_params(group, 0);
    params["GPU_TP_NO_SUCH_OPTION"] = true;
    OV_EXPECT_THROW(core.create_context("GPU.0", params), ov::Exception, HasSubstr("GPU_TP_NO_SUCH_OPTION"));

    params = rank_params(group, 0);
    params[ov::enable_profiling.name()] = true;
    OV_EXPECT_THROW(core.create_context("GPU.0", params),
                    ov::Exception,
                    HasSubstr("unsupported parameter of a TP context"));

    params = rank_params(group, 0, "NO_SUCH_BACKEND");
    OV_EXPECT_THROW(core.create_context("GPU.0", params),
                    ov::Exception,
                    HasSubstr("Unknown collective backend \"NO_SUCH_BACKEND\"; supported for GPU.0,GPU.1: L0_P2P"));

    params = rank_params(group, 0);
    params[ov::tp::internal::device_group.name()] = std::vector<std::string>{"GPU.0", "GPU.0"};
    OV_EXPECT_THROW(core.create_context("GPU.0", params), ov::Exception, HasSubstr("more than once"));

    OV_EXPECT_THROW(core.create_context("GPU.1", rank_params(group, 0)),
                    ov::Exception,
                    HasSubstr("rank 0 runs on GPU.0 by the group"));
}

TEST_F(GPUTPContextTest, RejectsWhatACollectiveCannotBeCompiledFor) {
    OV_EXPECT_THROW(core.compile_model(make_allreduce_model(0, 2), "GPU.0"),
                    ov::Exception,
                    HasSubstr("needs a TP context"));

    auto contexts = create_contexts();
    OV_EXPECT_THROW(core.compile_model(make_allreduce_model(1, 2), contexts[0]),
                    ov::Exception,
                    HasSubstr("is for rank 1 of 2, the context is rank 0 of 2"));
    OV_EXPECT_THROW(core.compile_model(make_allreduce_model(0, 2, 0, ReduceOp::MAX), contexts[0]),
                    ov::Exception,
                    HasSubstr("only SUM is supported"));
    OV_EXPECT_THROW(core.compile_model(make_allreduce_model(0, 2, 1), contexts[0]),
                    ov::Exception,
                    HasSubstr("only collective group 0 is supported"));
}

// One model per channel: the GPU plugin binds a compiled model's stream network to the channel.
TEST_F(GPUTPContextTest, RejectsASecondModelOnTheChannel) {
    auto contexts = create_contexts();
    auto first = core.compile_model(make_allreduce_model(0, 2), contexts[0]);
    OV_EXPECT_THROW(core.compile_model(make_allreduce_model(0, 2), contexts[0]),
                    ov::Exception,
                    HasSubstr("already bound to a model"));
}

// The ranks check at the first binding that they were given the same fixed staging.
TEST_F(GPUTPContextTest, RejectsRanksThatDisagreeOnTheStagingSize) {
    auto group = InProcessExchange::create_group(2, std::chrono::milliseconds{5000});
    auto params = rank_params(group, 0);
    params[ov::tp::internal::comm_buffer_size.name()] = size_t{8192};
    auto rank0 = core.create_context("GPU.0", params);
    params = rank_params(group, 1);
    params[ov::tp::internal::comm_buffer_size.name()] = size_t{16384};
    auto rank1 = core.create_context("GPU.1", params);
    OV_EXPECT_THROW(core.compile_model(make_allreduce_model(0, 2), rank0),
                    ov::Exception,
                    HasSubstr("must be the same on every rank"));
}

// What the TP contexts hold, the collectives' fixed staging included, is part of the device's memory
// statistics while they live, and is given back with them.
TEST_F(GPUTPContextTest, DeviceMemoryStatisticsCountTheTPContexts) {
    auto usm_device = [this] {
        return core.get_property("GPU.0", ov::intel_gpu::memory_statistics).at("usm_device");
    };
    core.get_default_context("GPU.0");
    core.get_default_context("GPU.1");
    const auto before = usm_device();

    constexpr size_t staging = size_t{8} << 20;
    {
        auto contexts = create_contexts({{ov::tp::internal::comm_buffer_size.name(), staging}});
        expect_two_ranks_reduce(contexts);
        EXPECT_GE(usm_device(), before + staging);
    }
    EXPECT_EQ(usm_device(), before);
}

}  // namespace
