// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// The rank contexts through the collective contract, without the TP plugin in
// between: the GPU plugin builds a TP context from the ov::tp::internal
// parameters, binds every model compiled with it, and refuses what does not
// fit -- a collective without a TP context, a rank of a different context, a
// second model on the one channel.

#include <gtest/gtest.h>

#include <future>
#include <string>
#include <vector>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/tp/collective_contract.hpp"
#include "ov_ops/tp_collective.hpp"

namespace ov::tp::tests {
namespace {

using ::testing::HasSubstr;

constexpr size_t elements = 4096;

/// x -> TPAllReduce -> result, for `rank` of `world_size`.
std::shared_ptr<ov::Model> make_allreduce_model(int rank, int world_size) {
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, elements});
    auto reduce = std::make_shared<ov::op::internal::TPAllReduce>(x, rank, world_size, 0, ov::op::internal::ReduceOp::SUM);
    return std::make_shared<ov::Model>(ov::OutputVector{reduce}, ov::ParameterVector{x});
}

class TPContextTest : public ::testing::Test {
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

    /// The TP parameters of one rank of {GPU.0, GPU.1}.
    static ov::AnyMap rank_params(const std::vector<std::shared_ptr<IExchange>>& group, int rank) {
        return {{ov::tp::internal::device_group.name(), std::vector<std::string>{"GPU.0", "GPU.1"}},
                {ov::tp::internal::groups.name(),
                 std::map<int, CollectiveGroup>{{0, CollectiveGroup{group[static_cast<size_t>(rank)], auto_backend}}}},
                {ov::tp::internal::communication_timeout_ms.name(), 5000}};
    }

    static std::vector<ov::RemoteContext> create_contexts(ov::Core& core, size_t comm_buffer_size = 0) {
        auto group = InProcessExchange::create_group(2, std::chrono::milliseconds{5000});
        std::vector<ov::RemoteContext> contexts;
        for (int rank = 0; rank < 2; ++rank) {
            auto params = rank_params(group, rank);
            if (comm_buffer_size != 0) {
                params[ov::tp::internal::comm_buffer_size.name()] = comm_buffer_size;
            }
            contexts.push_back(core.create_context(rank == 0 ? "GPU.0" : "GPU.1", params));
        }
        return contexts;
    }

    /// Compiles x -> TPAllReduce on both contexts, runs both ranks and checks the sum.
    static void expect_two_ranks_reduce(ov::Core& core, std::vector<ov::RemoteContext>& contexts) {
        std::vector<ov::CompiledModel> compiled;
        for (int rank = 0; rank < 2; ++rank) {
            compiled.push_back(core.compile_model(make_allreduce_model(rank, 2), contexts[static_cast<size_t>(rank)]));
        }

        std::vector<ov::InferRequest> requests;
        for (int rank = 0; rank < 2; ++rank) {
            requests.push_back(compiled[static_cast<size_t>(rank)].create_infer_request());
            ov::Tensor input(ov::element::f32, ov::Shape{1, elements});
            auto* data = input.data<float>();
            for (size_t i = 0; i < elements; ++i) {
                data[i] = static_cast<float>(rank + 1) * static_cast<float>(i % 7);
            }
            requests.back().set_input_tensor(input);
        }

        // Every rank has to be in its collective at the same time.
        std::vector<std::future<void>> ranks;
        for (auto& request : requests) {
            ranks.push_back(std::async(std::launch::async, [&request] {
                request.infer();
            }));
        }
        for (auto& rank : ranks) {
            rank.get();
        }

        for (auto& request : requests) {
            const auto output = request.get_output_tensor();
            const auto* data = output.data<const float>();
            for (size_t i = 0; i < elements; ++i) {
                ASSERT_FLOAT_EQ(data[i], 3.0f * static_cast<float>(i % 7)) << "element " << i;
            }
        }
    }

    static size_t gpu_count;
    static std::string gpu_error;
};

size_t TPContextTest::gpu_count = 0;
std::string TPContextTest::gpu_error;

TEST_F(TPContextTest, CollectiveNeedsATPContext) {
    ov::Core core;
    OV_EXPECT_THROW(core.compile_model(make_allreduce_model(0, 2), "GPU.0"),
                    ov::Exception,
                    HasSubstr("needs a TP context"));
}

TEST_F(TPContextTest, TwoRanksReduceThroughTheirContexts) {
    ov::Core core;
    auto contexts = create_contexts(core);
    expect_two_ranks_reduce(core, contexts);
}

// 8 KiB of staging is smaller than the message, which therefore goes in chunks.
TEST_F(TPContextTest, TwoRanksReduceThroughAFixedStagingInChunks) {
    ov::Core core;
    auto contexts = create_contexts(core, 8192);
    expect_two_ranks_reduce(core, contexts);
}

TEST_F(TPContextTest, RejectsRanksThatDisagreeOnTheStagingSize) {
    ov::Core core;
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

TEST_F(TPContextTest, RejectsACollectiveOfAnotherRank) {
    ov::Core core;
    auto contexts = create_contexts(core);
    OV_EXPECT_THROW(core.compile_model(make_allreduce_model(1, 2), contexts[0]),
                    ov::Exception,
                    HasSubstr("is for rank 1 of 2, the context is rank 0 of 2"));
}

TEST_F(TPContextTest, RejectsASecondModelOnTheChannel) {
    ov::Core core;
    auto contexts = create_contexts(core);
    auto first = core.compile_model(make_allreduce_model(0, 2), contexts[0]);
    OV_EXPECT_THROW(core.compile_model(make_allreduce_model(0, 2), contexts[0]),
                    ov::Exception,
                    HasSubstr("already bound to a model"));
}

TEST_F(TPContextTest, RejectsAContextOnAnotherDeviceThanItsRank) {
    ov::Core core;
    auto group = InProcessExchange::create_group(2, std::chrono::milliseconds{5000});
    OV_EXPECT_THROW(core.create_context("GPU.1", rank_params(group, 0)),
                    ov::Exception,
                    HasSubstr("rank 0 runs on GPU.0 by the group"));
}

// The queries a TP plugin runs before it creates any context: they answer for
// a device group without throwing, whatever is wrong with it.
TEST_F(TPContextTest, ListsTheCollectiveBackendsOfAGroup) {
    ov::Core core;
    auto backends = [&](const std::vector<std::string>& devices) {
        return core
            .get_property("GPU",
                          ov::tp::internal::collective_backends.name(),
                          {{ov::tp::internal::device_group.name(), devices}})
            .as<std::vector<std::string>>();
    };
    EXPECT_EQ(backends({"GPU.0", "GPU.1"}), std::vector<std::string>{"L0_P2P"});
    EXPECT_TRUE(backends({"GPU.0", "GPU.0"}).empty());
    EXPECT_TRUE(backends({"GPU.0", "GPU.99"}).empty());
}

TEST_F(TPContextTest, ReportsCollectiveSupportWithoutThrowing) {
    ov::Core core;
    auto support = [&](const std::vector<std::string>& devices, const std::string& backend) {
        return core
            .get_property("GPU",
                          ov::tp::internal::collective_support.name(),
                          {{ov::tp::internal::device_group.name(), devices},
                           {ov::tp::internal::collective_backend.name(), backend}})
            .as<std::string>();
    };
    EXPECT_EQ(support({"GPU.0", "GPU.1"}, auto_backend), "");
    EXPECT_EQ(support({"GPU.0", "GPU.1"}, "L0_P2P"), "");
    EXPECT_THAT(support({"GPU.0", "GPU.1"}, "NO_SUCH_BACKEND"),
                HasSubstr("Unknown collective backend \"NO_SUCH_BACKEND\"; supported for GPU.0,GPU.1: L0_P2P"));
    EXPECT_THAT(support({"GPU.0", "GPU.0"}, auto_backend), HasSubstr("appears in the group more than once"));
    EXPECT_THAT(support({"GPU.0", "GPU.99"}, auto_backend), HasSubstr("does not exist"));
}

TEST_F(TPContextTest, TwoRanksReduceThroughTheNamedBackend) {
    ov::Core core;
    auto group = InProcessExchange::create_group(2, std::chrono::milliseconds{5000});
    std::vector<ov::RemoteContext> contexts;
    for (int rank = 0; rank < 2; ++rank) {
        auto params = rank_params(group, rank);
        params[ov::tp::internal::groups.name()] =
            std::map<int, CollectiveGroup>{{0, CollectiveGroup{group[static_cast<size_t>(rank)], "L0_P2P"}}};
        contexts.push_back(core.create_context(rank == 0 ? "GPU.0" : "GPU.1", params));
    }
    expect_two_ranks_reduce(core, contexts);
}

TEST_F(TPContextTest, RejectsWhatItDoesNotSupport) {
    ov::Core core;
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

    params = rank_params(group, 0);
    params[ov::tp::internal::groups.name()] =
        std::map<int, CollectiveGroup>{{0, CollectiveGroup{group[0], "NO_SUCH_BACKEND"}}};
    OV_EXPECT_THROW(core.create_context("GPU.0", params),
                    ov::Exception,
                    HasSubstr("Unknown collective backend \"NO_SUCH_BACKEND\"; supported for GPU.0,GPU.1: L0_P2P"));

    params = rank_params(group, 0);
    params[ov::tp::internal::device_group.name()] = std::vector<std::string>{"GPU.0", "GPU.0"};
    OV_EXPECT_THROW(core.create_context("GPU.0", params), ov::Exception, HasSubstr("more than once"));
}

}  // namespace
}  // namespace ov::tp::tests
