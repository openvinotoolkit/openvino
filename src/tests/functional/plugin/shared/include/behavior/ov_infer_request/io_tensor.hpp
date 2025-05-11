// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <thread>
#include <future>

#include "base/ov_behavior_test_utils.hpp"

namespace ov {
namespace test {
namespace behavior {

struct OVInferRequestIOTensorTest : public OVInferRequestTests {
    void SetUp() override;
    void TearDown() override;
    ov::InferRequest req;
    ov::Output<const ov::Node> input;
    ov::Output<const ov::Node> output;
};

using OVInferRequestSetPrecisionParams = std::tuple<
        element::Type,                                                     // element type
        std::string,                                                       // Device name
        ov::AnyMap                                              // Config
>;
struct OVInferRequestIOTensorSetPrecisionTest : public testing::WithParamInterface<OVInferRequestSetPrecisionParams>,
                                                public OVInferRequestTestBase {
    static std::string getTestCaseName(const testing::TestParamInfo<OVInferRequestSetPrecisionParams>& obj);
    void SetUp() override;
    void TearDown() override;
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    std::shared_ptr<ov::Model> function;
    ov::CompiledModel execNet;
    ov::InferRequest req;
    ov::AnyMap          config;
    element::Type       element_type;
};

using OVInferRequestCheckTensorPrecisionParams = OVInferRequestSetPrecisionParams;

struct OVInferRequestCheckTensorPrecision : public testing::WithParamInterface<OVInferRequestCheckTensorPrecisionParams>,
                                            public OVInferRequestTestBase {
    static std::string getTestCaseName(const testing::TestParamInfo<OVInferRequestCheckTensorPrecisionParams>& obj);
    void SetUp() override;
    void TearDown() override;
    bool compareTensors(const ov::Tensor& t1, const ov::Tensor& t2);
    void createInferRequest();

    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    std::shared_ptr<ov::Model> model;
    CompiledModel compModel;
    InferRequest request;
    AnyMap  config;
    element::Type  element_type;

    std::vector<ov::element::Type> precisions = {
        ov::element::boolean,
        ov::element::bf16,
        ov::element::f16,
        ov::element::f32,
        ov::element::f64,
        ov::element::i4,
        ov::element::i8,
        ov::element::i16,
        ov::element::i32,
        ov::element::i64,
        ov::element::u1,
        ov::element::u4,
        ov::element::u8,
        ov::element::u16,
        ov::element::u32,
        ov::element::u64,
    };
    std::string exp_error_str_ = "The plugin does not support input precision";
};

} // namespace behavior
}  // namespace test
}  // namespace ov
