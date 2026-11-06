// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"
#include "openvino/runtime/core.hpp"

namespace ov {
namespace tp {
struct IExchange;
}  // namespace tp

namespace test {
namespace behavior {

/// Device type (e.g. "GPU") and number of ranks.
using TPCollectiveContractParams = std::tuple<std::string, size_t>;

/// L3 conformance tests: what a device plugin taking part in tensor parallelism guarantees (the device-plugin
/// obligations D1-D11 and the op contract), checked through the contract only -- TP contexts from the internal
/// TP properties, hand-sharded rank models, no TP plugin. A device plugin instantiates them with its device type.
class TPCollectiveContractTest : public ::testing::TestWithParam<TPCollectiveContractParams> {
public:
    static std::string get_test_case_name(const ::testing::TestParamInfo<TPCollectiveContractParams>& info);

protected:
    using Group = std::vector<std::shared_ptr<ov::tp::IExchange>>;
    using Contexts = std::vector<ov::RemoteContext>;
    using Requests = std::vector<ov::InferRequest>;

    void SetUp() override;

    static constexpr int default_timeout_ms = 5000;

    int ranks() const;
    Group make_group(int timeout_ms = default_timeout_ms) const;
    /// The TP parameters of rank `rank` of the group.
    ov::AnyMap rank_params(const Group& group,
                           int rank,
                           const std::string& backend = "AUTO",
                           int timeout_ms = default_timeout_ms) const;
    /// One context per rank, created one after another on this thread.
    Contexts create_contexts(const std::function<void(ov::AnyMap&)>& adjust = {},
                             int timeout_ms = default_timeout_ms);
    /// Compiles `make_model(rank)` on every rank's context.
    Requests compile(Contexts& contexts,
                     const std::function<std::shared_ptr<ov::Model>(int)>& make_model,
                     const ov::AnyMap& config = {});
    Requests compile_allreduce(Contexts& contexts, const ov::AnyMap& config = {});
    /// Runs every rank on its exact input and checks the exact sum on every rank.
    void expect_exact_sum(Requests& requests);
    std::vector<ov::element::Type> supported_precisions();
    std::vector<std::string> backends(const std::vector<std::string>& devices);
    std::string support(const std::vector<std::string>& devices, const std::string& backend);

    ov::Core m_core;
    std::string m_device_type;
    size_t m_world_size = 0;
    /// One device per rank, "<type>.<id>" in the order the device plugin lists them.
    std::vector<std::string> m_devices;
};

}  // namespace behavior
}  // namespace test
}  // namespace ov
