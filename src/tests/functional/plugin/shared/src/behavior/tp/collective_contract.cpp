// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/tp/collective_contract.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <functional>
#include <future>
#include <limits>
#include <random>
#include <sstream>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tp/collective_contract.hpp"
#include "ov_ops/tp_collective.hpp"

namespace ov {
namespace test {
namespace behavior {
namespace {

using ov::op::internal::ReduceOp;
using ov::op::internal::TPAllReduce;
using ov::op::internal::TPGather;
using ov::tp::CollectiveGroup;
using ov::tp::InProcessExchange;

constexpr size_t elements = 4096;

using Contexts = std::vector<ov::RemoteContext>;
using Requests = std::vector<ov::InferRequest>;

std::shared_ptr<ov::op::v0::Parameter> make_input(const ov::Shape& shape = {1, elements}) {
    return std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
}

/// x -> TPAllReduce -> result, for `rank` of `world_size`.
std::shared_ptr<ov::Model> make_allreduce_model(int rank,
                                                int world_size,
                                                int group_id = 0,
                                                ReduceOp reduce_op = ReduceOp::SUM) {
    auto x = make_input();
    auto reduce = std::make_shared<TPAllReduce>(x, rank, world_size, group_id, reduce_op);
    return std::make_shared<ov::Model>(ov::OutputVector{reduce}, ov::ParameterVector{x});
}

/// Rank r's input: (r + 1) * (i % 7) -- small integers, so every sum is exact in f32, f16 and bf16.
ov::Tensor exact_input(size_t rank, const ov::Shape& shape = {1, elements}) {
    ov::Tensor tensor(ov::element::f32, shape);
    auto* data = tensor.data<float>();
    for (size_t i = 0; i < tensor.get_size(); ++i)
        data[i] = static_cast<float>(rank + 1) * static_cast<float>(i % 7);
    return tensor;
}

ov::Tensor random_input(uint32_t seed, const ov::Shape& shape = {1, elements}) {
    ov::Tensor tensor(ov::element::f32, shape);
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    auto* data = tensor.data<float>();
    for (size_t i = 0; i < tensor.get_size(); ++i)
        data[i] = distribution(generator);
    return tensor;
}

/// Every rank has to be in its collective at the same time.
void run_ranks(Requests& requests) {
    std::vector<std::future<void>> ranks;
    for (auto& request : requests)
        ranks.push_back(std::async(std::launch::async, [&request] {
            request.infer();
        }));
    for (auto& rank : ranks)
        rank.get();
}

void expect_values(const ov::Tensor& actual, const std::vector<float>& expected, float tolerance, size_t rank) {
    ASSERT_EQ(actual.get_size(), expected.size()) << "rank " << rank;
    const auto* data = actual.data<const float>();
    for (size_t i = 0; i < expected.size(); ++i)
        ASSERT_NEAR(data[i], expected[i], tolerance) << "rank " << rank << ", element " << i;
}

void expect_same_bits(Requests& requests) {
    const auto first = requests.front().get_output_tensor();
    for (size_t rank = 1; rank < requests.size(); ++rank) {
        const auto other = requests[rank].get_output_tensor();
        ASSERT_EQ(other.get_byte_size(), first.get_byte_size());
        EXPECT_EQ(std::memcmp(other.data(), first.data(), first.get_byte_size()), 0)
            << "rank " << rank << " differs from rank 0";
    }
}

}  // namespace

std::string TPCollectiveContractTest::get_test_case_name(
    const ::testing::TestParamInfo<TPCollectiveContractParams>& info) {
    std::ostringstream name;
    name << "device=" << std::get<0>(info.param) << "_ranks=" << std::get<1>(info.param);
    return name.str();
}

void TPCollectiveContractTest::SetUp() {
    std::tie(m_device_type, m_world_size) = GetParam();
    std::vector<std::string> ids;
    try {
        ids = m_core.get_property(m_device_type, ov::available_devices);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "no " << m_device_type << " devices: " << e.what();
    }
    if (ids.size() < m_world_size)
        GTEST_SKIP() << "need " << m_world_size << " " << m_device_type << " devices, found " << ids.size();
    for (size_t rank = 0; rank < m_world_size; ++rank)
        m_devices.push_back(m_device_type + "." + ids[rank]);
}

int TPCollectiveContractTest::ranks() const {
    return static_cast<int>(m_world_size);
}

TPCollectiveContractTest::Group TPCollectiveContractTest::make_group(int timeout_ms) const {
    return InProcessExchange::create_group(ranks(), std::chrono::milliseconds{timeout_ms});
}

ov::AnyMap TPCollectiveContractTest::rank_params(const Group& group,
                                                 int rank,
                                                 const std::string& backend,
                                                 int timeout_ms) const {
    return {{ov::tp::internal::device_group.name(), m_devices},
            {ov::tp::internal::groups.name(),
             std::map<int, CollectiveGroup>{{0, CollectiveGroup{group[static_cast<size_t>(rank)], backend}}}},
            {ov::tp::internal::communication_timeout_ms.name(), timeout_ms}};
}

TPCollectiveContractTest::Contexts TPCollectiveContractTest::create_contexts(
    const std::function<void(ov::AnyMap&)>& adjust,
    int timeout_ms) {
    const auto group = make_group(timeout_ms);
    Contexts contexts;
    for (int rank = 0; rank < ranks(); ++rank) {
        auto params = rank_params(group, rank, ov::tp::auto_backend, timeout_ms);
        if (adjust)
            adjust(params);
        contexts.push_back(m_core.create_context(m_devices[static_cast<size_t>(rank)], params));
    }
    return contexts;
}

TPCollectiveContractTest::Requests TPCollectiveContractTest::compile(
    Contexts& contexts,
    const std::function<std::shared_ptr<ov::Model>(int)>& make_model,
    const ov::AnyMap& config) {
    std::vector<ov::CompiledModel> compiled;
    for (int rank = 0; rank < ranks(); ++rank)
        compiled.push_back(m_core.compile_model(make_model(rank), contexts[static_cast<size_t>(rank)], config));
    Requests requests;
    for (auto& model : compiled)
        requests.push_back(model.create_infer_request());
    return requests;
}

TPCollectiveContractTest::Requests TPCollectiveContractTest::compile_allreduce(Contexts& contexts,
                                                                               const ov::AnyMap& config) {
    return compile(
        contexts,
        [this](int rank) {
            return make_allreduce_model(rank, ranks());
        },
        config);
}

void TPCollectiveContractTest::expect_exact_sum(Requests& requests) {
    for (size_t rank = 0; rank < requests.size(); ++rank)
        requests[rank].set_input_tensor(exact_input(rank));
    run_ranks(requests);
    const auto factor = static_cast<float>(m_world_size * (m_world_size + 1) / 2);
    std::vector<float> expected(elements);
    for (size_t i = 0; i < elements; ++i)
        expected[i] = factor * static_cast<float>(i % 7);
    for (size_t rank = 0; rank < requests.size(); ++rank)
        expect_values(requests[rank].get_output_tensor(), expected, 0.0f, rank);
}

std::vector<ov::element::Type> TPCollectiveContractTest::supported_precisions() {
    const auto capabilities = m_core.get_property(m_devices.front(), ov::device::capabilities);
    std::vector<ov::element::Type> precisions;
    const std::pair<const char*, ov::element::Type> known[] = {{ov::device::capability::FP32, ov::element::f32},
                                                               {ov::device::capability::FP16, ov::element::f16},
                                                               {ov::device::capability::BF16, ov::element::bf16}};
    for (const auto& [capability, type] : known)
        if (std::find(capabilities.begin(), capabilities.end(), capability) != capabilities.end())
            precisions.push_back(type);
    return precisions;
}

std::vector<std::string> TPCollectiveContractTest::backends(const std::vector<std::string>& devices) {
    return m_core
        .get_property(m_device_type,
                      ov::tp::internal::collective_backends.name(),
                      {{ov::tp::internal::device_group.name(), devices}})
        .as<std::vector<std::string>>();
}

std::string TPCollectiveContractTest::support(const std::vector<std::string>& devices, const std::string& backend) {
    return m_core
        .get_property(m_device_type,
                      ov::tp::internal::collective_support.name(),
                      {{ov::tp::internal::device_group.name(), devices},
                       {ov::tp::internal::collective_backend.name(), backend}})
        .as<std::string>();
}

namespace {

ov::tp::internal::DeviceMemory device_memory(ov::Core& core, const std::string& device) {
    return core.get_property(device, ov::tp::internal::device_memory.name(), {}).as<ov::tp::internal::DeviceMemory>();
}

// ---- D1: TP context creation ----

TEST_P(TPCollectiveContractTest, D1_ContextsAreCreatedOneAfterAnotherWithoutWaiting) {
    const auto started = std::chrono::steady_clock::now();
    {
        auto contexts = create_contexts();
        EXPECT_EQ(contexts.size(), m_world_size);
    }
    // Publishing does not wait for the peers, so nothing here waits for the timeout.
    EXPECT_LT(std::chrono::steady_clock::now() - started, std::chrono::milliseconds{default_timeout_ms});
}

TEST_P(TPCollectiveContractTest, D1_RejectsAContextOnAnotherDeviceThanItsRank) {
    const auto group = make_group();
    OV_EXPECT_THROW_HAS_SUBSTRING(m_core.create_context(m_devices[1], rank_params(group, 0)), ov::Exception, "");
}

TEST_P(TPCollectiveContractTest, D1_RejectsAnUnknownBackendListingTheValidOnes) {
    const auto valid = backends(m_devices);
    ASSERT_FALSE(valid.empty());
    const auto group = make_group();
    try {
        m_core.create_context(m_devices[0], rank_params(group, 0, "NO_SUCH_BACKEND"));
        FAIL() << "an unknown backend was accepted";
    } catch (const ov::Exception& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("NO_SUCH_BACKEND"), std::string::npos) << message;
        for (const auto& name : valid)
            EXPECT_NE(message.find(name), std::string::npos) << "the error does not list " << name << ": " << message;
    }
}

TEST_P(TPCollectiveContractTest, D1_RejectsAnUnknownParameter) {
    const auto group = make_group();
    auto params = rank_params(group, 0);
    params["NO_SUCH_TP_PARAMETER"] = 1;
    OV_EXPECT_THROW_HAS_SUBSTRING(m_core.create_context(m_devices[0], params), ov::Exception, "NO_SUCH_TP_PARAMETER");
}

TEST_P(TPCollectiveContractTest, D1_RejectsAContextWithoutItsGroup) {
    const auto group = make_group();
    auto params = rank_params(group, 0);
    params.erase(ov::tp::internal::groups.name());
    OV_EXPECT_THROW_HAS_SUBSTRING(m_core.create_context(m_devices[0], params), ov::Exception, "");
}

// ---- D2: compile and import with the context ----

TEST_P(TPCollectiveContractTest, D2_ACollectiveNeedsATPContext) {
    OV_EXPECT_THROW_HAS_SUBSTRING(m_core.compile_model(make_allreduce_model(0, ranks()), m_devices[0]),
                                  ov::Exception,
                                  "");
}

TEST_P(TPCollectiveContractTest, D2_RanksReduceThroughTheirContexts) {
    auto contexts = create_contexts();
    auto requests = compile_allreduce(contexts);
    expect_exact_sum(requests);
}

TEST_P(TPCollectiveContractTest, D2_ImportedRanksReduceThroughFreshContexts) {
    std::vector<std::string> blobs;
    {
        auto contexts = create_contexts();
        for (int rank = 0; rank < ranks(); ++rank) {
            auto compiled =
                m_core.compile_model(make_allreduce_model(rank, ranks()), contexts[static_cast<size_t>(rank)]);
            std::stringstream blob;
            compiled.export_model(blob);
            blobs.push_back(blob.str());
        }
    }
    auto contexts = create_contexts();
    Requests requests;
    for (int rank = 0; rank < ranks(); ++rank) {
        std::stringstream blob(blobs[static_cast<size_t>(rank)]);
        requests.push_back(m_core.import_model(blob, contexts[static_cast<size_t>(rank)]).create_infer_request());
    }
    expect_exact_sum(requests);
}

TEST_P(TPCollectiveContractTest, D2_RejectsACollectiveOfAnotherRank) {
    auto contexts = create_contexts();
    OV_EXPECT_THROW_HAS_SUBSTRING(m_core.compile_model(make_allreduce_model(1, ranks()), contexts[0]),
                                  ov::Exception,
                                  "");
}

TEST_P(TPCollectiveContractTest, D2_RejectsACollectiveOfAnotherWorldSize) {
    auto contexts = create_contexts();
    OV_EXPECT_THROW_HAS_SUBSTRING(m_core.compile_model(make_allreduce_model(0, ranks() + 1), contexts[0]),
                                  ov::Exception,
                                  "");
}

// ---- D3: the only binding route ----

TEST_P(TPCollectiveContractTest, D3_TheContextParametersAreNotACompileConfig) {
    const auto group = make_group();
    OV_EXPECT_THROW_HAS_SUBSTRING(
        m_core.compile_model(make_allreduce_model(0, ranks()), m_devices[0], rank_params(group, 0)),
        ov::Exception,
        "");
}

// ---- D4: collective ops and the op contract ----

TEST_P(TPCollectiveContractTest, D4_AllReduceIsWithinTheToleranceAndBitwiseEqualOnAllRanks) {
    for (const auto precision : supported_precisions()) {
        SCOPED_TRACE(precision.get_type_name());
        auto contexts = create_contexts();
        auto requests = compile_allreduce(contexts, {ov::hint::inference_precision(precision)});
        std::vector<float> expected(elements, 0.0f);
        for (size_t rank = 0; rank < m_world_size; ++rank) {
            auto input = random_input(static_cast<uint32_t>(rank + 1));
            const auto* data = input.data<const float>();
            for (size_t i = 0; i < elements; ++i)
                expected[i] += data[i];
            requests[rank].set_input_tensor(input);
        }
        run_ranks(requests);
        // Each input is rounded once to the precision and the sum about once per rank.
        const float epsilon = precision == ov::element::f32    ? std::numeric_limits<float>::epsilon()
                              : precision == ov::element::f16  ? 9.77e-4f
                                                               : 7.82e-3f;
        const float tolerance = epsilon * static_cast<float>(m_world_size * (m_world_size + 1));
        for (size_t rank = 0; rank < m_world_size; ++rank)
            expect_values(requests[rank].get_output_tensor(), expected, tolerance, rank);
        expect_same_bits(requests);
    }
}

// MAX and MIN may be unsupported; then compilation fails instead of computing something else.
TEST_P(TPCollectiveContractTest, D4_ReduceOpsAreExactOrRejected) {
    for (const auto reduce_op : {ReduceOp::MAX, ReduceOp::MIN}) {
        SCOPED_TRACE(reduce_op == ReduceOp::MAX ? "MAX" : "MIN");
        auto contexts = create_contexts();
        Requests requests;
        try {
            requests = compile(contexts, [&](int rank) {
                return make_allreduce_model(rank, ranks(), 0, reduce_op);
            });
        } catch (const ov::Exception&) {
            continue;
        }
        for (size_t rank = 0; rank < m_world_size; ++rank)
            requests[rank].set_input_tensor(exact_input(rank));
        run_ranks(requests);
        std::vector<float> expected(elements);
        for (size_t i = 0; i < elements; ++i)
            expected[i] =
                static_cast<float>(reduce_op == ReduceOp::MAX ? m_world_size : 1) * static_cast<float>(i % 7);
        for (size_t rank = 0; rank < m_world_size; ++rank)
            expect_values(requests[rank].get_output_tensor(), expected, 0.0f, rank);
    }
}

TEST_P(TPCollectiveContractTest, D4_RejectsACollectiveOfAGroupTheContextDoesNotHave) {
    auto contexts = create_contexts();
    OV_EXPECT_THROW_HAS_SUBSTRING(m_core.compile_model(make_allreduce_model(0, ranks(), 1), contexts[0]),
                                  ov::Exception,
                                  "");
}

// The root gets the slices of all ranks in rank order, row by row; the others' outputs have the shape of
// their slice and unspecified contents. A device that does not implement the gather rejects it.
TEST_P(TPCollectiveContractTest, D4_GatherConcatenatesOnTheRootOrIsRejected) {
    for (const ov::Shape& shape : {ov::Shape{2, 3, 128}, ov::Shape{4, 128}, ov::Shape{1, 512}}) {
        SCOPED_TRACE(shape.to_string());
        const size_t slice = shape.back();
        const size_t rows = ov::shape_size(shape) / slice;
        auto contexts = create_contexts();
        Requests requests;
        try {
            requests = compile(contexts, [&](int rank) {
                auto x = make_input(shape);
                auto gather = std::make_shared<TPGather>(x, rank, ranks(), 0, -1);
                return std::make_shared<ov::Model>(ov::OutputVector{gather}, ov::ParameterVector{x});
            });
        } catch (const ov::Exception&) {
            GTEST_SKIP() << "the device does not implement TPGather";
        }
        std::vector<ov::Tensor> inputs;
        for (size_t rank = 0; rank < m_world_size; ++rank) {
            inputs.push_back(exact_input(rank, shape));
            requests[rank].set_input_tensor(inputs.back());
        }
        run_ranks(requests);

        // Row by row: the slices of ranks 0..N-1 side by side.
        std::vector<float> root;
        for (size_t row = 0; row < rows; ++row)
            for (const auto& input : inputs)
                root.insert(root.end(),
                            input.data<const float>() + row * slice,
                            input.data<const float>() + (row + 1) * slice);
        expect_values(requests[0].get_output_tensor(), root, 0.0f, 0);
        for (size_t rank = 1; rank < m_world_size; ++rank)
            EXPECT_EQ(requests[rank].get_output_tensor().get_shape(), shape) << "rank " << rank;
    }
}

// Two chained collectives with rank-local work between them and a third on a parallel branch, over many
// inferences: every call pairs with the same call on the other ranks.
TEST_P(TPCollectiveContractTest, D4_CollectivesRunInGraphOrderAcrossManyInferences) {
    // Rank r scales by r + 2: the rank models differ in constants only, never by a multiplication by one.
    auto scale_of = [](size_t rank) {
        return static_cast<float>(rank + 2);
    };
    auto contexts = create_contexts();
    auto requests = compile(contexts, [&](int rank) {
        auto x = make_input();
        auto first = std::make_shared<TPAllReduce>(x, rank, ranks(), 0, ReduceOp::SUM);
        auto scale = ov::op::v0::Constant::create(ov::element::f32, {}, {scale_of(static_cast<size_t>(rank))});
        auto scaled = std::make_shared<ov::op::v1::Multiply>(first, scale);
        auto second = std::make_shared<TPAllReduce>(scaled, rank, ranks(), 0, ReduceOp::SUM);
        auto branch = std::make_shared<TPAllReduce>(std::make_shared<ov::op::v1::Multiply>(x, scale),
                                                    rank,
                                                    ranks(),
                                                    0,
                                                    ReduceOp::SUM);
        auto sum = std::make_shared<ov::op::v1::Add>(second, branch);
        return std::make_shared<ov::Model>(ov::OutputVector{sum}, ov::ParameterVector{x});
    });
    // x_r = (r + 1) * v: first = sum(r + 1) * v, second = first * sum(s_r), branch = sum((r + 1) * s_r) * v.
    float ranks_sum = 0, scales_sum = 0, products_sum = 0;
    for (size_t rank = 0; rank < m_world_size; ++rank) {
        ranks_sum += static_cast<float>(rank + 1);
        scales_sum += scale_of(rank);
        products_sum += static_cast<float>(rank + 1) * scale_of(rank);
    }
    for (int iteration = 0; iteration < 32; ++iteration) {
        // v = (i + iteration) % 5: small integers, exact in every precision.
        for (size_t rank = 0; rank < m_world_size; ++rank) {
            ov::Tensor input(ov::element::f32, {1, elements});
            auto* data = input.data<float>();
            for (size_t i = 0; i < elements; ++i)
                data[i] = static_cast<float>(rank + 1) * static_cast<float>((i + iteration) % 5);
            requests[rank].set_input_tensor(input);
        }
        run_ranks(requests);
        std::vector<float> expected(elements);
        for (size_t i = 0; i < elements; ++i)
            expected[i] = (ranks_sum * scales_sum + products_sum) * static_cast<float>((i + iteration) % 5);
        for (size_t rank = 0; rank < m_world_size; ++rank)
            expect_values(requests[rank].get_output_tensor(), expected, 0.0f, rank);
    }
}

// ---- D5: queries and backend names ----

TEST_P(TPCollectiveContractTest, D5_AnswersTheQueriesWithoutThrowing) {
    const auto valid = backends(m_devices);
    ASSERT_FALSE(valid.empty());
    EXPECT_EQ(support(m_devices, ov::tp::auto_backend), "");
    for (const auto& name : valid)
        EXPECT_EQ(support(m_devices, name), "") << name;

    const auto unknown = support(m_devices, "NO_SUCH_BACKEND");
    EXPECT_NE(unknown, "");
    for (const auto& name : valid)
        EXPECT_NE(unknown.find(name), std::string::npos) << "the reason does not list " << name << ": " << unknown;

    auto missing = m_devices;
    missing.back() = m_device_type + ".99";
    EXPECT_TRUE(backends(missing).empty());
    EXPECT_NE(support(missing, ov::tp::auto_backend), "");
}

// Several ranks on one device are the plugin's decision; the queries and the contexts agree on it.
TEST_P(TPCollectiveContractTest, D5_TheQueriesAndTheContextsAgreeOnARepeatedDevice) {
    std::vector<std::string> repeated(m_world_size, m_devices[0]);
    const bool supported = !backends(repeated).empty();
    EXPECT_EQ(support(repeated, ov::tp::auto_backend).empty(), supported);
    if (supported)
        GTEST_SKIP() << "several ranks on one device are supported; covered by the other tests";
    const auto group = make_group();
    auto params = rank_params(group, 0);
    params[ov::tp::internal::device_group.name()] = repeated;
    OV_EXPECT_THROW_HAS_SUBSTRING(m_core.create_context(m_devices[0], params), ov::Exception, "");
}

// ---- D6: channels ----

TEST_P(TPCollectiveContractTest, D6_RejectsAModelWhoseStreamsDoNotFitTheChannels) {
    auto contexts = create_contexts();
    OV_EXPECT_THROW_HAS_SUBSTRING(
        m_core.compile_model(make_allreduce_model(0, ranks()), contexts[0], {ov::num_streams(2)}),
        ov::Exception,
        "");
    OV_EXPECT_THROW_HAS_SUBSTRING(m_core.compile_model(make_allreduce_model(0, ranks()),
                                                       contexts[0],
                                                       {{ov::tp::internal::channel_base.name(), uint32_t{1}}}),
                                  ov::Exception,
                                  "");
}

// ---- D7: timeout ----

TEST_P(TPCollectiveContractTest, D7_TheSetupTimesOutWithoutThePeers) {
    constexpr int short_timeout = 500;
    const auto group = make_group(short_timeout);
    auto context = m_core.create_context(m_devices[0], rank_params(group, 0, ov::tp::auto_backend, short_timeout));
    const auto started = std::chrono::steady_clock::now();
    OV_EXPECT_THROW_HAS_SUBSTRING(m_core.compile_model(make_allreduce_model(0, ranks()), context), ov::Exception, "");
    EXPECT_LT(std::chrono::steady_clock::now() - started, std::chrono::milliseconds{30 * short_timeout});
}

// A call that the peers never join fails after the timeout, and the communicator stays failed.
TEST_P(TPCollectiveContractTest, D7_ACallTimesOutWithoutThePeersAndTheGroupStaysFailed) {
    constexpr int short_timeout = 500;
    auto contexts = create_contexts({}, short_timeout);
    auto requests = compile_allreduce(contexts);
    requests[0].set_input_tensor(exact_input(0));
    const auto started = std::chrono::steady_clock::now();
    OV_EXPECT_THROW_HAS_SUBSTRING(requests[0].infer(), ov::Exception, "");
    EXPECT_LT(std::chrono::steady_clock::now() - started, std::chrono::milliseconds{30 * short_timeout});

    for (size_t rank = 0; rank < m_world_size; ++rank)
        requests[rank].set_input_tensor(exact_input(rank));
    OV_EXPECT_THROW_HAS_SUBSTRING(run_ranks(requests), ov::Exception, "");
}

// ---- D8-D10: memory ----

TEST_P(TPCollectiveContractTest, D8_HostTensorsOfAnyRankAreReadByEveryRank) {
    auto contexts = create_contexts();
    auto requests = compile_allreduce(contexts);
    for (size_t rank = 0; rank < m_world_size; ++rank) {
        auto host = contexts[(rank + 1) % m_world_size].create_host_tensor(ov::element::f32, {1, elements});
        exact_input(rank).copy_to(host);
        requests[rank].set_input_tensor(host);
    }
    run_ranks(requests);
    const auto factor = static_cast<float>(m_world_size * (m_world_size + 1) / 2);
    std::vector<float> expected(elements);
    for (size_t i = 0; i < elements; ++i)
        expected[i] = factor * static_cast<float>(i % 7);
    for (size_t rank = 0; rank < m_world_size; ++rank)
        expect_values(requests[rank].get_output_tensor(), expected, 0.0f, rank);
}

// The rank models' constants are views into one host buffer; what the plugin reads after compilation is
// its own copy, never the buffer in place.
TEST_P(TPCollectiveContractTest, D9_ConstantsAreViewsAndAreNotReadInPlace) {
    ov::Tensor weights(ov::element::f32, {m_world_size, elements});
    for (size_t rank = 0; rank < m_world_size; ++rank) {
        auto* row = weights.data<float>() + rank * elements;
        for (size_t i = 0; i < elements; ++i)
            row[i] = static_cast<float>(rank + 1);
    }
    auto contexts = create_contexts();
    auto requests = compile(contexts, [&](int rank) {
        const auto r = static_cast<size_t>(rank);
        ov::Tensor view(weights, {r, 0}, {r + 1, elements});
        auto x = make_input();
        auto add = std::make_shared<ov::op::v1::Add>(x, std::make_shared<ov::op::v0::Constant>(view));
        auto reduce = std::make_shared<TPAllReduce>(add, rank, ranks(), 0, ReduceOp::SUM);
        return std::make_shared<ov::Model>(ov::OutputVector{reduce}, ov::ParameterVector{x});
    });
    std::fill_n(weights.data<float>(), weights.get_size(), 1000.0f);

    const auto factor = static_cast<float>(m_world_size * (m_world_size + 1) / 2);
    for (size_t rank = 0; rank < m_world_size; ++rank)
        requests[rank].set_input_tensor(exact_input(rank));
    run_ranks(requests);
    std::vector<float> expected(elements);
    for (size_t i = 0; i < elements; ++i)
        expected[i] = factor * static_cast<float>(i % 7) + factor;
    for (size_t rank = 0; rank < m_world_size; ++rank)
        expect_values(requests[rank].get_output_tensor(), expected, 0.0f, rank);
}

TEST_P(TPCollectiveContractTest, D10_RankLocalTensorsAreReadAndWrittenInPlace) {
    auto contexts = create_contexts();
    auto requests = compile_allreduce(contexts);
    std::vector<ov::RemoteTensor> outputs;
    for (size_t rank = 0; rank < m_world_size; ++rank) {
        auto input = contexts[rank].create_tensor(ov::element::f32, {1, elements}, {});
        auto output = contexts[rank].create_tensor(ov::element::f32, {1, elements}, {});
        EXPECT_EQ(input.get_device_name(), m_devices[rank]);
        input.copy_from(exact_input(rank));
        requests[rank].set_input_tensor(input);
        requests[rank].set_output_tensor(output);
        outputs.push_back(output);
    }
    run_ranks(requests);
    const auto factor = static_cast<float>(m_world_size * (m_world_size + 1) / 2);
    std::vector<float> expected(elements);
    for (size_t i = 0; i < elements; ++i)
        expected[i] = factor * static_cast<float>(i % 7);
    for (size_t rank = 0; rank < m_world_size; ++rank) {
        ov::Tensor host(ov::element::f32, {1, elements});
        outputs[rank].copy_to(host);
        expect_values(host, expected, 0.0f, rank);
    }
}

// ---- D11: memory accounting, bounded collective memory ----

// A fixed staging is allocated before the first inference and counted as used device memory while the
// contexts live.
TEST_P(TPCollectiveContractTest, D11_DeviceMemoryCountsTheCollectiveMemoryBeforeTheFirstInference) {
    constexpr size_t staging = size_t{8} << 20;
    for (const auto& device : m_devices)
        m_core.get_default_context(device);
    const auto before = device_memory(m_core, m_devices[0]);
    EXPECT_GT(before.total, 0u);
    EXPECT_GT(before.max_allocation, 0u);
    EXPECT_LE(before.used, before.total);
    {
        auto contexts = create_contexts([&](ov::AnyMap& params) {
            params[ov::tp::internal::comm_buffer_size.name()] = staging;
        });
        auto requests = compile_allreduce(contexts);
        EXPECT_GE(device_memory(m_core, m_devices[0]).used, before.used + staging);
        expect_exact_sum(requests);
    }
    EXPECT_EQ(device_memory(m_core, m_devices[0]).used, before.used);
}

// 8 KiB of staging is smaller than the 16 KiB message, which therefore goes in chunks.
TEST_P(TPCollectiveContractTest, D11_AMessageLargerThanTheStagingIsReducedExactly) {
    auto contexts = create_contexts([](ov::AnyMap& params) {
        params[ov::tp::internal::comm_buffer_size.name()] = size_t{8192};
    });
    auto requests = compile_allreduce(contexts);
    expect_exact_sum(requests);
}

}  // namespace

}  // namespace behavior
}  // namespace test
}  // namespace ov
