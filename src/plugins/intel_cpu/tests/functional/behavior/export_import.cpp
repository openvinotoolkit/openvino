// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-corer: Apache-2.0
//

#include "openvino/core/any.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/properties.hpp"
#include "common_test_utils/test_common.hpp"
#include "ngraph_functions/builders.hpp"


#include <openvino/opsets/opset9.hpp>
#include <ie/ie_core.hpp>

namespace {

class ExportImportTest : public CommonTestUtils::TestsCommon {};

std::shared_ptr<ov::Model> MakeMatMulModel() {
    const ov::Shape input_shape = {1, 4096};
    const ov::element::Type precision = ov::element::f32;

    auto params = ngraph::builder::makeParams(precision, {input_shape});
    auto matmul_const = ngraph::builder::makeConstant(precision, {4096, 1024}, std::vector<float>{}, true);
    auto matmul = ngraph::builder::makeMatMul(params[0], matmul_const);

    auto add_const = ngraph::builder::makeConstant(precision, {1, 1024}, std::vector<float>{}, true);
    auto add = ngraph::builder::makeEltwise(matmul, add_const, ngraph::helpers::EltwiseTypes::ADD);
    auto softmax = std::make_shared<ov::opset9::Softmax>(add);

    ngraph::NodeVector results{softmax};
    return std::make_shared<ov::Model>(results, params, "MatMulModel");
}

TEST(ExportImportTest, ExportOptimalNumStreams) {
    auto original_model = MakeMatMulModel();
    std::string deviceName = "CPU";
    ov::Core core;
    auto tput_mode = ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT);
    auto latency_mode = ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY);

    auto original_tp_network = core.compile_model(original_model, deviceName, tput_mode);
    auto original_latency_network = core.compile_model(original_model, deviceName, latency_mode);

    auto nstreams_tp_original = original_tp_network.get_property(ov::num_streams.name()).as<std::string>();
    auto nstreams_latency_original = original_latency_network.get_property(ov::num_streams.name()).as<std::string>();

    std::stringstream exported_stream;
    original_tp_network.export_model(exported_stream);
    {
        std::stringstream ss(exported_stream.str());
        auto imported_tp_network = core.import_model(ss, deviceName, tput_mode);
        auto nstreams_tp_imported = imported_tp_network.get_property(ov::num_streams.name()).as<std::string>();
        EXPECT_EQ(nstreams_tp_original, nstreams_tp_imported);
    }

    {
        std::stringstream ss(exported_stream.str());
        auto imported_latency_network = core.import_model(ss, deviceName, latency_mode);
        auto nstreams_latency_imported = imported_latency_network.get_property(ov::num_streams.name()).as<std::string>();
        EXPECT_EQ(nstreams_latency_original, nstreams_latency_imported);
    }
}
}  // namespace
