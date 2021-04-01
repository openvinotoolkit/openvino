// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "ngraph_reader_tests.hpp"
#include "common_test_utils/xml_net_builder/ir_net.hpp"

TEST_F(NGraphReaderTests, ReadLrnNetwork) {
    CommonTestUtils::IRBuilder_v10 ir_builder_v10("LRN");

    auto input_layer = ir_builder_v10
            .AddLayer("in1", "Parameter", {{"shape", "1,3,22,22"},
                {"element_type", "f32"}}).AddOutPort(Precision::ePrecision::FP32, {1, 3, 22, 22})
            .getLayer();

    auto data_layer = ir_builder_v10
            .AddLayer("data1", "Const", {{"element_type", "i64"}, {"offset", "0"}, {"size", "16"}, {"shape", "2"}})
            .AddOutPort(Precision::ePrecision::I64, {2})
            .getLayer();

    auto lrn_layer = ir_builder_v10
            .AddLayer("activation", "LRN", {{"alpha", "0"},
                                            {"beta",  "0.75"},
                                            {"size",  "5"},
                                            {"bias",  "1"}})
            .AddInPort(Precision::ePrecision::FP32, {1, 3, 22, 22})
            .AddInPort(Precision::ePrecision::I64, {1})
            .AddOutPort(Precision::ePrecision::FP32, {1, 3, 22, 22})
            .getLayer();

    auto result_layer = ir_builder_v10
            .AddLayer("output", "Result")
            .AddInPort(Precision::ePrecision::FP32, {1, 3, 22, 22})
            .getLayer();

    input_layer.out(0).connect(lrn_layer.in(0));
    data_layer.out(0).connect(lrn_layer.in(1));
    lrn_layer.out(0).connect(result_layer.in(0));

    CommonTestUtils::IRBuilder_v6 ir_builder_v6("LRN");
    auto in_layer = ir_builder_v6
            .AddLayer("in1", "Input", Precision::ePrecision::FP32)
            .AddOutPort({1, 3, 22, 22})
            .getLayer();

    auto activation_layer = ir_builder_v6
            .AddLayer("activation", "Norm", Precision::ePrecision::FP32, {{"alpha",     "0.000000"},
                                                                         {"beta",       "0.750000"},
                                                                         {"local-size", "5"},
                                                                         {"region",     "same"},
                                                                         {"k",          "1"}})
            .AddInPort({1, 3, 22, 22})
            .AddOutPort({1, 3, 22, 22})
            .getLayer();

    in_layer.out(0).connect(activation_layer.in(0));

    std::string model_v10 = ir_builder_v10.serialize();
    std::string model_v5 = ir_builder_v6.serialize();

    compareIRs(model_v10, model_v5, 16, [](Blob::Ptr& weights) {
                auto * w = weights->buffer().as<int64_t*>();
                w[0] = 2;
                w[1] = 3;
            });
}

TEST_F(NGraphReaderTests, ReadLrnNetwork2) {
    CommonTestUtils::IRBuilder_v10 ir_builder_v10("Activation");

    auto input_layer = ir_builder_v10
            .AddLayer("in1", "Parameter", {{"shape", "1,3,22,22"},
                {"element_type", "f32"}}).AddOutPort(Precision::ePrecision::FP32, {1, 3, 22, 22})
            .getLayer();

    auto data_layer = ir_builder_v10
            .AddLayer("data1", "Const", {{"element_type", "i64"}, {"offset", "0"}, {"size", "8"}, {"shape", "1"}})
            .AddOutPort(Precision::ePrecision::I64, {1})
            .getLayer();

    auto lrn_layer = ir_builder_v10
            .AddLayer("activation", "LRN", {{"alpha", "0"},
                                            {"beta",  "0.75"},
                                            {"size",  "5"},
                                            {"bias",  "1"}})
            .AddInPort(Precision::ePrecision::FP32, {1, 3, 22, 22})
            .AddInPort(Precision::ePrecision::I64, {1})
            .AddOutPort(Precision::ePrecision::FP32, {1, 3, 22, 22})
            .getLayer();

    auto result_layer = ir_builder_v10
            .AddLayer("output", "Result")
            .AddInPort(Precision::ePrecision::FP32, {1, 3, 22, 22})
            .getLayer();

    input_layer.out(0).connect(lrn_layer.in(0));
    data_layer.out(0).connect(lrn_layer.in(1));
    lrn_layer.out(0).connect(result_layer.in(0));

    CommonTestUtils::IRBuilder_v6 ir_builder_v6("Activation");
    auto in_layer = ir_builder_v6
            .AddLayer("in1", "Input", Precision::ePrecision::FP32)
            .AddOutPort({1, 3, 22, 22})
            .getLayer();

    auto activation_layer = ir_builder_v6
            .AddLayer("activation", "Norm", Precision::ePrecision::FP32, {{"alpha",      "0.000000"},
                                                                          {"beta",       "0.750000"},
                                                                          {"local-size", "5"},
                                                                          {"region",     "across"},
                                                                          {"k",          "1"}})
            .AddInPort({1, 3, 22, 22})
            .AddOutPort({1, 3, 22, 22})
            .getLayer();

    in_layer.out(0).connect(activation_layer.in(0));

    std::string model_v10 = ir_builder_v10.serialize();
    std::string model_v5 = ir_builder_v6.serialize();

    compareIRs(model_v10, model_v5, 8, [](Blob::Ptr& weights) {
        auto * w = weights->buffer().as<int64_t*>();
        w[0] = 1;
    });
}
