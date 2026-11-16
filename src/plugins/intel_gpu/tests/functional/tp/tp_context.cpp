// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// What the GPU plugin adds to the collective contract (the shared L3 conformance tests,
// behavior/tp/collective_contract.hpp): its backend, its options, its limits and its messages.

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "behavior/tp/collective_contract.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/test_constants.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/runtime/tp/collective_contract.hpp"

namespace {

using ::testing::HasSubstr;
using ov::op::internal::ReduceOp;

class GPUTPContextTest : public ov::test::behavior::TPCollectiveContractTest {
protected:
    /// How the plugin lists the group in its messages: "GPU.0,GPU.1".
    std::string group_list() const {
        return m_devices[0] + "," + m_devices[1];
    }
};

TEST_P(GPUTPContextTest, TheBackendIsL0P2P) {
    EXPECT_EQ(backends(m_devices), std::vector<std::string>{"L0_P2P"});
    EXPECT_EQ(support(m_devices, "L0_P2P"), "");
    const auto group = make_group();
    Contexts contexts;
    for (uint32_t rank = 0; rank < ranks(); ++rank)
        contexts.push_back(m_core.create_context(m_devices[rank], rank_params(group, rank, "L0_P2P")));
    auto requests = compile_allreduce(contexts);
    expect_exact_sum(requests);
}

TEST_P(GPUTPContextTest, ExplainsWhatItDoesNotSupport) {
    const auto& d0 = m_devices[0];
    EXPECT_THAT(support(m_devices, "NO_SUCH_BACKEND"),
                HasSubstr("Unknown collective backend \"NO_SUCH_BACKEND\"; supported for " + group_list() + ": L0_P2P"));
    EXPECT_TRUE(backends({d0, d0}).empty());
    EXPECT_THAT(support({d0, d0}, ov::tp::auto_backend), HasSubstr("appears in the group more than once"));
    const auto missing = m_device_type + ".99";
    EXPECT_TRUE(backends({d0, missing}).empty());
    EXPECT_THAT(support({d0, missing}, ov::tp::auto_backend), HasSubstr("does not exist"));
}

TEST_P(GPUTPContextTest, TakesItsCollectiveOptionsAsContextParameters) {
    auto contexts = create_contexts([](ov::AnyMap& params) {
        params["GPU_TP_ENABLE_HALVING"] = false;
    });
    auto requests = compile_allreduce(contexts);
    expect_exact_sum(requests);
}

TEST_P(GPUTPContextTest, RejectsWhatTheContextDoesNotSupport) {
    const auto group = make_group();

    auto params = rank_params(group, 0);
    params[ov::tp::internal::channels.name()] = uint32_t{2};
    OV_EXPECT_THROW(m_core.create_context(m_devices[0], params), ov::Exception, HasSubstr("only one channel"));

    params = rank_params(group, 0);
    params["GPU_TP_NO_SUCH_OPTION"] = true;
    OV_EXPECT_THROW(m_core.create_context(m_devices[0], params), ov::Exception, HasSubstr("GPU_TP_NO_SUCH_OPTION"));

    params = rank_params(group, 0);
    params[ov::enable_profiling.name()] = true;
    OV_EXPECT_THROW(m_core.create_context(m_devices[0], params),
                    ov::Exception,
                    HasSubstr("unsupported parameter of a TP context"));

    params = rank_params(group, 0, "NO_SUCH_BACKEND");
    OV_EXPECT_THROW(
        m_core.create_context(m_devices[0], params),
        ov::Exception,
        HasSubstr("Unknown collective backend \"NO_SUCH_BACKEND\"; supported for " + group_list() + ": L0_P2P"));

    params = rank_params(group, 0);
    params[ov::tp::internal::device_group.name()] = std::vector<std::string>{m_devices[0], m_devices[0]};
    OV_EXPECT_THROW(m_core.create_context(m_devices[0], params), ov::Exception, HasSubstr("more than once"));

    OV_EXPECT_THROW(m_core.create_context(m_devices[1], rank_params(group, 0)),
                    ov::Exception,
                    HasSubstr("rank 0 runs on " + m_devices[0] + " by the group"));
}

TEST_P(GPUTPContextTest, RejectsWhatACollectiveCannotBeCompiledFor) {
    OV_EXPECT_THROW(m_core.compile_model(make_allreduce_model(0, ranks()), m_devices[0]),
                    ov::Exception,
                    HasSubstr("needs a TP context"));

    auto contexts = create_contexts();
    const auto world = std::to_string(ranks());
    OV_EXPECT_THROW(m_core.compile_model(make_allreduce_model(1, ranks()), contexts[0]),
                    ov::Exception,
                    HasSubstr("is for rank 1 of " + world + ", the context is rank 0 of " + world));
    OV_EXPECT_THROW(m_core.compile_model(make_allreduce_model(0, ranks(), 0, ReduceOp::MAX), contexts[0]),
                    ov::Exception,
                    HasSubstr("only SUM is supported"));
    OV_EXPECT_THROW(m_core.compile_model(make_allreduce_model(0, ranks(), 1), contexts[0]),
                    ov::Exception,
                    HasSubstr("only collective group 0 is supported"));
}

// One model per channel: the GPU plugin binds a compiled model's stream network to the channel.
TEST_P(GPUTPContextTest, RejectsASecondModelOnTheChannel) {
    auto contexts = create_contexts();
    auto first = m_core.compile_model(make_allreduce_model(0, ranks()), contexts[0]);
    OV_EXPECT_THROW(m_core.compile_model(make_allreduce_model(0, ranks()), contexts[0]),
                    ov::Exception,
                    HasSubstr("already bound to a model"));
}

// A staging the device cannot allocate is named in the error rather than left to a Level Zero code.
TEST_P(GPUTPContextTest, ExplainsAStagingItCannotAllocate) {
    const auto group = make_group();
    auto params = rank_params(group, 0);
    params[ov::tp::internal::comm_buffer_size.name()] = size_t{1} << 60;
    OV_EXPECT_THROW(m_core.create_context(m_devices[0], params),
                    ov::Exception,
                    HasSubstr("rank 0 cannot allocate its collective staging of 1152921504606846976 bytes"));
}

// The ranks check at the first binding that they were given the same fixed staging.
TEST_P(GPUTPContextTest, RejectsRanksThatDisagreeOnTheStagingSize) {
    const auto group = make_group();
    auto params = rank_params(group, 0);
    params[ov::tp::internal::comm_buffer_size.name()] = size_t{8192};
    auto rank0 = m_core.create_context(m_devices[0], params);
    params = rank_params(group, 1);
    params[ov::tp::internal::comm_buffer_size.name()] = size_t{16384};
    auto rank1 = m_core.create_context(m_devices[1], params);
    OV_EXPECT_THROW(m_core.compile_model(make_allreduce_model(0, ranks()), rank0),
                    ov::Exception,
                    HasSubstr("must be the same on every rank"));
}

// What the TP contexts hold, the collectives' fixed staging included, is part of the device's memory
// statistics while they live, and is given back with them.
TEST_P(GPUTPContextTest, DeviceMemoryStatisticsCountTheTPContexts) {
    auto usm_device = [this] {
        return m_core.get_property(m_devices[0], ov::intel_gpu::memory_statistics).at("usm_device");
    };
    for (const auto& device : m_devices)
        m_core.get_default_context(device);
    const auto before = usm_device();

    constexpr size_t staging = size_t{8} << 20;
    {
        auto contexts = create_contexts([&](ov::AnyMap& params) {
            params[ov::tp::internal::comm_buffer_size.name()] = staging;
        });
        auto requests = compile_allreduce(contexts);
        expect_exact_sum(requests);
        EXPECT_GE(usm_device(), before + staging);
    }
    EXPECT_EQ(usm_device(), before);
}

// The messages name two devices; the rest of the contract is covered on every world size by the shared suite.
INSTANTIATE_TEST_SUITE_P(smoke_TP,
                         GPUTPContextTest,
                         ::testing::Combine(::testing::Values(std::string(ov::test::utils::DEVICE_GPU)),
                                            ::testing::Values(2u)),
                         GPUTPContextTest::get_test_case_name);

}  // namespace
