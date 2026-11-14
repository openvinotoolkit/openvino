// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "builder/arch_registry.hpp"
#include "builder/projector_registry.hpp"
#include "graph_regression_utils.hpp"
#include "gtest/gtest.h"
#include "helper_ops/gguf_constant.hpp"
#include "op_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/split.hpp"
#include "pass/materialize_weights.hpp"

namespace {
nlohmann::json regression_catalog() {
    std::ifstream input(std::filesystem::path(ov_gguf_test::test_data_dir()) / "graph_regressions/catalog.json");
    OPENVINO_ASSERT(input, "Required GGUF graph-regression catalog is missing");
    return nlohmann::json::parse(input);
}

std::shared_ptr<ov::Model> projection(bool transpose = false) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2});
    input->output(0).set_names({"input"});
    auto weight = ov::op::v0::Constant::create(ov::element::f32, {2, 2}, {1, 2, 3, 4});
    auto matmul = std::make_shared<ov::op::v0::MatMul>(input, weight, false, transpose);
    matmul->output(0).set_names({"output"});
    return std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{input});
}
}  // namespace

TEST(GGUFGraphRegression, IgnoresDebugNamesButDetectsAttributes) {
    auto model = projection();
    const auto reference = ov_gguf_test::graph_snapshot(model);
    for (const auto& node : model->get_ops())
        node->set_friendly_name("unrelated_debug_name");
    for (const auto& node : model->get_ops()) {
        if (ov::is_type<ov::op::v0::Constant>(node))
            node->output(0).set_names({"unrelated_private_alias"});
    }
    EXPECT_EQ(reference, ov_gguf_test::graph_snapshot(model));
    EXPECT_NE(reference.at("canonical"), ov_gguf_test::graph_snapshot(projection(true)).at("canonical"));
}

TEST(GGUFGraphRegression, CatalogCoversEveryNativeHandler) {
    const auto catalog = regression_catalog();
    std::set<std::string> covered;
    for (const auto& entry : catalog.at("cases")) {
        if (entry.value("kind", "model") == "model" && entry.value("stage", "deferred_plan") == "deferred_plan")
            covered.insert(entry.at("architecture").get<std::string>());
    }
    EXPECT_EQ(covered, ov::frontend::gguf::default_arch_registry().supported_archs())
        << "New native handlers require an original-dimension recipe and reference";
}

TEST(GGUFGraphRegression, CatalogClassifiesEveryProjector) {
    const auto catalog = regression_catalog();
    std::set<std::string> covered;
    // LFM3 generation is a composite mmproj branch, not a registry entry.
    std::set<std::string> registered{"audio_generation.lfm3a_gen"};
    for (const auto& entry : catalog.at("cases")) {
        if (entry.contains("projectors")) {
            for (const auto& projector : entry.at("projectors"))
                covered.insert(projector.get<std::string>());
        }
    }
    for (const auto& definition : ov::frontend::gguf::default_arch_registry().projectors().supported_projectors())
        registered.insert(definition.modality + "." + definition.projector_type);
    EXPECT_EQ(covered, registered) << "New projector contracts must not silently escape the coverage inventory";
}

TEST(GGUFGraphRegression, CapturesDeepGraphsWithoutRecursiveTraversal) {
    auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    ov::Output<ov::Node> output = parameter;
    for (size_t index = 0; index < 8192; ++index)
        output = std::make_shared<ov::op::v0::Relu>(output);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{parameter});
    const auto snapshot = ov_gguf_test::graph_snapshot(model);
    EXPECT_EQ(snapshot.at("totals").at("nodes"), 8194);
}

TEST(GGUFGraphRegression, PreservesSharingWhenOperationHistogramMatches) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2});
    auto a = ov::op::v0::Constant::create(ov::element::f32, {2}, {1, 1});
    auto b = ov::op::v0::Constant::create(ov::element::f32, {2}, {1, 1});
    auto first = std::make_shared<ov::op::v1::Add>(input, a);
    auto second = std::make_shared<ov::op::v1::Add>(input, b);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{first, second, b}, ov::ParameterVector{input});
    const auto reference = ov_gguf_test::graph_snapshot(model);
    second->input(1).replace_source_output(first);
    const auto actual = ov_gguf_test::graph_snapshot(model);
    EXPECT_EQ(reference.at("ops"), actual.at("ops"));
    EXPECT_NE(reference.at("canonical"), actual.at("canonical"));
}

TEST(GGUFGraphRegression, RejectsUnknownOwnedRuntimeKeys) {
    auto model = projection();
    model->get_parameters().front()->get_rt_info()["gguf.unclassified"] = true;
    EXPECT_ANY_THROW(ov_gguf_test::graph_snapshot(model));
    model->get_parameters().front()->get_rt_info().erase("gguf.unclassified");
    model->get_parameters().front()->get_rt_info()["unclassified_marker"] = true;
    EXPECT_ANY_THROW(ov_gguf_test::graph_snapshot(model));
    model->get_parameters().front()->get_rt_info().erase("unclassified_marker");
    model->get_parameters().front()->get_rt_info()["fused_names_0"] = std::string("debug");
    model->get_rt_info()["unrelated_model_key"] = true;
    EXPECT_NO_THROW(ov_gguf_test::graph_snapshot(model));
}

TEST(GGUFGraphRegression, IgnoresFloatValuesButDetectsShapesAndPerformanceMarkers) {
    const auto make_model = [](float scale, size_t size) {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2});
        auto constant = ov::op::v0::Constant::create(ov::element::f32, {size}, std::vector<float>(size, scale));
        return std::make_shared<ov::Model>(ov::OutputVector{std::make_shared<ov::op::v1::Multiply>(input, constant)},
                                           ov::ParameterVector{input});
    };
    const auto reference = ov_gguf_test::graph_snapshot(make_model(0.0f, 2)).at("canonical");
    EXPECT_EQ(reference, ov_gguf_test::graph_snapshot(make_model(0.125f, 2)).at("canonical"));
    EXPECT_NE(reference, ov_gguf_test::graph_snapshot(make_model(0.0f, 1)).at("canonical"));
    auto model = make_model(0.0f, 2);
    model->get_results().front()->get_input_node_shared_ptr(0)->get_rt_info()["decompression_0"] = true;
    EXPECT_NE(reference, ov_gguf_test::graph_snapshot(model).at("canonical"));
}

TEST(GGUFGraphRegression, CapturesEmptyOptionalConstantWithoutReadingValues) {
    auto empty = std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0});
    auto model = std::make_shared<ov::Model>(ov::OutputVector{empty}, ov::ParameterVector{});
    EXPECT_NO_THROW(ov_gguf_test::graph_snapshot(model));
}

TEST(GGUFGraphRegression, DetectsLongControlValuesAndModelContracts) {
    std::vector<int64_t> values(128, 1);
    const auto make_model = [&]() {
        auto constant = ov::op::v0::Constant::create(ov::element::i64, {values.size()}, values);
        return std::make_shared<ov::Model>(ov::OutputVector{constant}, ov::ParameterVector{});
    };
    const auto reference = ov_gguf_test::graph_snapshot(make_model());
    values.back() = 2;
    auto model = make_model();
    EXPECT_NE(reference.at("canonical"), ov_gguf_test::graph_snapshot(model).at("canonical"));
    const auto without_contract = ov_gguf_test::graph_snapshot(model);
    model->get_rt_info()["gguf_model_contract"] =
        ov::AnyMap{{"supports_genai_adaptation", std::string("true")},
                   {"component_groups", ov::AnyMap{{"decoder", ov::AnyMap{{"primary", std::string("native")}}}}}};
    EXPECT_NE(without_contract.at("canonical"), ov_gguf_test::graph_snapshot(model).at("canonical"));
    model->get_rt_info()["gguf_unknown_contract"] = true;
    EXPECT_ANY_THROW(ov_gguf_test::graph_snapshot(model));
}

TEST(GGUFGraphRegression, CapturesLoopBodyAndPortMapping) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2});
    auto body_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2});
    auto increment = ov::op::v0::Constant::create(ov::element::f32, {2}, {1, 1});
    auto sum = std::make_shared<ov::op::v1::Add>(body_input, increment);
    auto condition = ov::op::v0::Constant::create(ov::element::boolean, {}, {true});
    auto body = std::make_shared<ov::Model>(ov::OutputVector{condition, sum}, ov::ParameterVector{body_input});
    auto trip_count = ov::op::v0::Constant::create(ov::element::i64, {}, {2});
    auto loop = std::make_shared<ov::op::v5::Loop>(trip_count, condition);
    loop->set_function(body);
    loop->set_special_body_ports({-1, 0});
    loop->set_merged_input(body_input, input, sum);
    const auto output = loop->get_iter_value(sum, -1);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{input});
    const auto reference = ov_gguf_test::graph_snapshot(model);
    for (const auto& node : body->get_ops())
        node->set_friendly_name("body_debug_name");
    EXPECT_EQ(reference, ov_gguf_test::graph_snapshot(model));
    auto replacement = ov::op::v0::Constant::create(ov::element::f32, {1}, {1});
    sum->input(1).replace_source_output(replacement);
    EXPECT_NE(reference.at("canonical"), ov_gguf_test::graph_snapshot(model).at("canonical"));
    auto product = std::make_shared<ov::op::v1::Multiply>(body_input, increment);
    body->get_results().at(1)->input(0).replace_source_output(product);
    EXPECT_NE(reference.at("ops"), ov_gguf_test::graph_snapshot(model).at("ops"));
}

TEST(GGUFGraphRegression, WeightExtensionIsFrontendInstanceScoped) {
    using namespace ov::frontend::gguf;
    GgufWeight source;
    source.shape = {2, 2};
    source.source_name = "test.weight";
    source.data = ov::Tensor(ov::element::f32, source.shape);
    std::fill_n(source.data.data<float>(), 4, 1.f);
    auto decoder = std::make_shared<ov_gguf_test::SingleOpDecoder>(
        "GGML_OP_NONE",
        std::vector<ov_gguf_test::TensorDesc>{},
        std::vector<ov_gguf_test::TensorDesc>{},
        ov_gguf_test::TensorDesc{"weight", ov::element::f32, source.shape},
        std::map<std::string, ov::Any>{{"gguf_source", source}});
    FrontEnd structural, ordinary;
    structural.add_extension(ov_gguf_test::structural_weight_extension());
    auto planned = structural.convert(structural.load(std::shared_ptr<GgufDecoder>(decoder)));
    auto materialized = ordinary.convert(ordinary.load(std::shared_ptr<GgufDecoder>(decoder)));
    EXPECT_EQ(planned->get_output_shape(0), materialized->get_output_shape(0));
    EXPECT_FALSE(ov::is_type<ov::op::v0::Constant>(planned->get_results().front()->get_input_node_shared_ptr(0)));
    EXPECT_TRUE(ov::is_type<ov::op::v0::Constant>(materialized->get_results().front()->get_input_node_shared_ptr(0)));
    EXPECT_ANY_THROW(structural.add_extension(ov_gguf_test::structural_weight_extension()));
}

TEST(GGUFGraphRegression, MaterializerCallbackPreservesPlanAndDefaultExecution) {
    using namespace ov::frontend::gguf;
    GgufWeight source;
    source.shape = {2, 2};
    source.source_name = "test.weight";
    source.data = ov::Tensor(ov::element::f32, source.shape);
    std::fill_n(source.data.data<float>(), 4, 1.f);
    auto leaf = std::make_shared<GGUFConstant>(source);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{leaf}, ov::ParameterVector{});
    auto normal = model->clone();
    ov::pass::Manager manager;
    ov_gguf_test::structural_weight_extension()->register_passes(manager);
    manager.run_passes(model);
    EXPECT_EQ(model->get_output_shape(0), source.shape);
    const auto snapshot = ov_gguf_test::graph_snapshot(model);
    EXPECT_NE(snapshot.dump().find("test.weight"), std::string::npos);
    pass::MaterializeGGUFWeights().run_on_model(normal);
    EXPECT_TRUE(ov::is_type<ov::op::v0::Constant>(normal->get_results().front()->get_input_node_shared_ptr(0)));
    EXPECT_EQ(normal->get_output_shape(0), model->get_output_shape(0));
}
