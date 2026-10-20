// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <tuple>

#include "builder/api/metadata_store.hpp"
#include "builder/arch/dspark_builder.hpp"
#include "builder/gguf_builder_decoder.hpp"
#include "builder/gguf_graph.hpp"
#include "builder/graph_emitter.hpp"
#include "cnpy.h"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "gtest/gtest.h"
#include "op_test_utils.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/frontend/extension/decoder_transformation.hpp"
#include "openvino/frontend/gguf/adapt_mmproj_to_genai.hpp"
#include "openvino/frontend/gguf/adapt_to_genai.hpp"
#include "openvino/frontend/gguf/extension/genai.hpp"
#include "openvino/frontend/gguf/frontend.hpp"
#include "openvino/frontend/gguf/make_stateful.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/paged_gated_delta_net.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/openvino.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"
#include "quant/gguf.hpp"
#include "utils.hpp"

namespace {
class GGUFArchitectureAccuracy : public ::testing::TestWithParam<const char*> {};

TEST(GGUFDSparkAccuracy, ConfidenceAndMarkovHeadsAreIndependent) {
    using namespace ov::frontend::gguf;
    const auto arrays = cnpy::npz_load(ov_gguf_test::test_data_dir() + "/arch_accuracy/dspark.npz");
    const ov_gguf_test::TemporaryGguf temporary(ov_gguf_test::npz_array(arrays, "model"));
    for (const auto& [rank, confidence, with_markov] :
         {std::tuple{0, false, false}, {0, true, false}, {8, false, false}, {8, true, false}, {8, true, true}}) {
        SCOPED_TRACE(::testing::Message() << rank << ", " << confidence << ", " << with_markov);
        auto loaded = get_gguf_data(temporary.path);
        auto& metadata = std::get<0>(loaded);
        auto& weights = std::get<1>(loaded);
        auto& qtypes = std::get<2>(loaded);
        ov::Tensor rank_value(ov::element::u32, {});
        *rank_value.data<uint32_t>() = static_cast<uint32_t>(rank);
        metadata["dspark.dspark.markov_rank"] = rank_value;
        for (const auto& [key, enabled] :
             {std::pair{"confidence_head", confidence}, {"confidence_head_with_markov", with_markov}}) {
            ov::Tensor flag(ov::element::boolean, {});
            *flag.data<char>() = enabled;
            metadata[std::string("dspark.dspark.") + key] = flag;
        }
        if (rank == 0) {
            weights.erase("dspark.markov_head_a.weight");
            weights.erase("dspark.markov_head_b.weight");
        }
        ov::Tensor projection(ov::element::f32, {1, with_markov ? size_t{72} : size_t{64}});
        std::copy_n(weights.at("dspark.confidence_head.weight").data<const float>(),
                    projection.get_size(),
                    projection.data<float>());
        weights["dspark.confidence_head.weight"] = projection;
        detail::MetadataStore store{metadata};
        detail::WeightStore tensors{weights, qtypes};
        const BuildContext context{GgufMetadata(store), "dspark", &tensors};
        const auto graph = make_dspark_builder(context)->build();
        EXPECT_EQ(graph->model_inputs.count("dspark_prev_tokens"), rank > 0 ? 1u : 0u);
        FrontEnd frontend;
        std::shared_ptr<GgufDecoder> decoder = std::make_shared<GgufBuilderDecoder>(graph);
        const auto model = frontend.convert(frontend.load(decoder));
        std::set<std::string> outputs;
        for (const auto& output : model->outputs())
            outputs.insert(output.get_names().begin(), output.get_names().end());
        EXPECT_EQ(outputs.count("dspark.confidence_score"), confidence ? 1u : 0u);
        EXPECT_EQ(outputs.count("dspark.corrected_logits"), rank > 0 ? 1u : 0u);
        if (!confidence)
            continue;
        ov::Core core;
        auto compiled = core.compile_model(model,
                                           "CPU",
                                           ov::hint::inference_precision(ov::element::f32),
                                           ov::hint::dynamic_quantization_group_size(0),
                                           ov::inference_num_threads(4));
        auto request = compiled.create_infer_request();
        for (const auto& input : compiled.inputs()) {
            const auto name = input.get_any_name();
            auto shape = input.get_partial_shape();
            for (size_t i = 0; i < shape.size(); ++i)
                if (shape[i].is_dynamic())
                    shape[i] = name == "dspark_ctx_feat" ? 2 : 6;
            ov::Tensor value(input.get_element_type(), shape.to_shape());
            std::memset(value.data(), 0, value.get_byte_size());
            if (input.get_element_type() == ov::element::i32)
                for (size_t i = 0; i < value.get_size(); ++i)
                    value.data<int32_t>()[i] = static_cast<int32_t>(i);
            request.set_tensor(name, value);
        }
        request.infer();
        const auto hidden = request.get_tensor("dspark.hidden");
        const auto score = request.get_tensor("dspark.confidence_score");
        ASSERT_EQ(score.get_size(), 4u);
        for (size_t token = 0; token < 4; ++token) {
            double expected = weights.at("dspark.confidence_head.bias").data<const float>()[0];
            for (size_t feature = 0; feature < 64; ++feature)
                expected += hidden.data<const float>()[token * 64 + feature] * projection.data<const float>()[feature];
            if (with_markov)
                for (size_t feature = 0; feature < 8; ++feature)
                    expected += weights.at("dspark.markov_head_a.weight").data<const float>()[token * 8 + feature] *
                                projection.data<const float>()[64 + feature];
            EXPECT_NEAR(score.data<const float>()[token], expected, 1e-6);
        }
    }
}

TEST(GGUFDSparkAccuracy, ConfidenceWithMarkovRequiresPositiveRank) {
    using namespace ov::frontend::gguf;
    const auto arrays = cnpy::npz_load(ov_gguf_test::test_data_dir() + "/arch_accuracy/dspark.npz");
    const ov_gguf_test::TemporaryGguf temporary(ov_gguf_test::npz_array(arrays, "model"));
    auto loaded = get_gguf_data(temporary.path);
    auto& metadata = std::get<0>(loaded);
    ov::Tensor rank(ov::element::u32, {});
    *rank.data<uint32_t>() = 0;
    metadata["dspark.dspark.markov_rank"] = rank;
    detail::MetadataStore store{metadata};
    detail::WeightStore tensors{std::get<1>(loaded), std::get<2>(loaded)};
    const BuildContext context{GgufMetadata(store), "dspark", &tensors};
    OV_EXPECT_THROW(make_dspark_builder(context)->build(),
                    ov::Exception,
                    testing::HasSubstr("confidence_head_with_markov"));
}

TEST(GGUFHadamardMetadata, MissingAndMistypedFieldsNameTheInvalidKey) {
    using namespace ov::frontend::gguf;
    const auto arrays = cnpy::npz_load(ov_gguf_test::test_data_dir() + "/arch_accuracy/bonsai2-hadamard.npz");
    const ov_gguf_test::TemporaryGguf temporary(ov_gguf_test::npz_array(arrays, "model"));
    auto loaded = get_gguf_data(temporary.path);
    const auto config = decoder_config_from_meta(std::get<0>(loaded));
    GraphEmitter valid(std::get<1>(loaded), std::get<2>(loaded), "qwen35");
    EXPECT_NO_THROW(valid.configure_hadamard(config));
    for (const std::string key : {"prism.hadamard.version",
                                  "prism.hadamard.block_size",
                                  "prism.hadamard.transform",
                                  "prism.hadamard.axis",
                                  "prism.hadamard.weight_names",
                                  "prism.hadamard.inverse_weight_names",
                                  "prism.hadamard.sign_mode",
                                  "prism.hadamard.sign_widths",
                                  "prism.hadamard.sign_values",
                                  "prism.hadamard.gdn_v_grouped",
                                  "ssm_time_step_rank",
                                  "ssm_group_count"}) {
        SCOPED_TRACE(key);
        for (const bool missing : {false, true}) {
            if (missing && (key == "prism.hadamard.inverse_weight_names" || key == "prism.hadamard.gdn_v_grouped"))
                continue;
            auto invalid = config;
            if (missing)
                invalid.erase(key);
            else
                invalid[key] = std::monostate{};
            GraphEmitter emitter(std::get<1>(loaded), std::get<2>(loaded), "qwen35");
            OV_EXPECT_THROW(emitter.configure_hadamard(invalid), ov::Exception, testing::HasSubstr(key));
        }
        if (std::holds_alternative<ov::Tensor>(config.at(key))) {
            auto invalid = config;
            invalid[key] = ov::Tensor(ov::element::f32, {1});
            GraphEmitter emitter(std::get<1>(loaded), std::get<2>(loaded), "qwen35");
            OV_EXPECT_THROW(emitter.configure_hadamard(invalid), ov::Exception, testing::HasSubstr(key));
        }
    }
}

TEST(GGUFDSparkAccuracy, RejectsOversizedDraftBlocksBeforeBuildingTheGraph) {
    const auto arrays = cnpy::npz_load(ov_gguf_test::test_data_dir() + "/arch_accuracy/dspark.npz");
    const ov_gguf_test::TemporaryGguf temporary(ov_gguf_test::npz_array(arrays, "model"));
    auto loaded = ov::frontend::gguf::get_gguf_data(temporary.path);
    auto& metadata = std::get<0>(loaded);
    auto& weights = std::get<1>(loaded);
    auto& qtypes = std::get<2>(loaded);
    ov::frontend::gguf::detail::MetadataStore store{metadata};
    ov::frontend::gguf::detail::WeightStore tensors{weights, qtypes};
    const ov::frontend::gguf::BuildContext context{ov::frontend::gguf::GgufMetadata(store), "dspark", &tensors};
    for (int64_t block : {int64_t{0}, int64_t{4097}, int64_t{UINT32_MAX}, std::numeric_limits<int64_t>::max()}) {
        SCOPED_TRACE(block);
        ov::Tensor value(ov::element::i64, {});
        *value.data<int64_t>() = block;
        metadata["dspark.dspark.block_size"] = value;
        auto builder = ov::frontend::gguf::make_dspark_builder(context);
        try {
            builder->build();
            FAIL() << "Invalid DSpark block size was accepted";
        } catch (const ov::Exception& error) {
            EXPECT_NE(std::string(error.what()).find("block_size"), std::string::npos);
        }
    }
    ov::Tensor value(ov::element::i64, {});
    *value.data<int64_t>() = 4096;
    metadata["dspark.dspark.block_size"] = value;
    const auto graph = ov::frontend::gguf::make_dspark_builder(context)->build();
    EXPECT_EQ(graph->model_inputs.at("inp_tokens")->get_output_partial_shape(0), (ov::PartialShape{1, 1, 1, 4096}));
    EXPECT_EQ(weights.at("dspark.log_snr_features").get_byte_size(), 4096 * 128 * sizeof(float));
}

TEST(GGUFDSparkAccuracy, LogSnrUsesNormalizedEndpointsAndRequiresValidBounds) {
    using namespace ov::frontend::gguf;
    const auto arrays = cnpy::npz_load(ov_gguf_test::test_data_dir() + "/arch_accuracy/dspark.npz");
    const ov_gguf_test::TemporaryGguf temporary(ov_gguf_test::npz_array(arrays, "model"));
    const std::string low_key = "dspark.dspark.min_log_snr";
    const std::string high_key = "dspark.dspark.max_log_snr";
    for (const auto& [low, high] : {std::pair{-9.f, 9.f}, {-9.f, 6.f}, {1.f, 5.f}}) {
        auto loaded = get_gguf_data(temporary.path);
        auto& metadata = std::get<0>(loaded);
        for (const auto& [key, value] : {std::pair{low_key, low}, {high_key, high}}) {
            ov::Tensor bound(ov::element::f32, {});
            *bound.data<float>() = value;
            metadata[key] = bound;
        }
        detail::MetadataStore store{metadata};
        detail::WeightStore tensors{std::get<1>(loaded), std::get<2>(loaded)};
        const BuildContext context{GgufMetadata(store), "dspark", &tensors};
        EXPECT_NO_THROW(make_dspark_builder(context)->build());
        const auto& features = tensors.weights.at("dspark.log_snr_features");
        for (size_t pos = 0; pos < 4; ++pos) {
            const float log_snr = pos == 0 ? high : low;
            const float time = (log_snr - low) / (high - low) * 1000.f;
            for (size_t i = 0; i < 64; ++i) {
                const float angle = time * std::exp(-std::log(10000.f) * i / 64.f);
                EXPECT_FLOAT_EQ(features.data<const float>()[pos * 128 + i], std::sin(angle));
                EXPECT_FLOAT_EQ(features.data<const float>()[pos * 128 + 64 + i], std::cos(angle));
            }
        }
    }
    for (const auto& [key, value, missing] : {std::tuple{low_key, 0.f, true},
                                              {high_key, 0.f, true},
                                              {low_key, std::numeric_limits<float>::quiet_NaN(), false},
                                              {high_key, std::numeric_limits<float>::infinity(), false},
                                              {low_key, 9.f, false},
                                              {low_key, 10.f, false}}) {
        SCOPED_TRACE(key);
        auto loaded = get_gguf_data(temporary.path);
        auto& metadata = std::get<0>(loaded);
        if (missing) {
            metadata.erase(key);
        } else {
            ov::Tensor bound(ov::element::f32, {});
            *bound.data<float>() = value;
            metadata[key] = bound;
        }
        detail::MetadataStore store{metadata};
        detail::WeightStore tensors{std::get<1>(loaded), std::get<2>(loaded)};
        const BuildContext context{GgufMetadata(store), "dspark", &tensors};
        OV_EXPECT_THROW(make_dspark_builder(context)->build(), ov::Exception, testing::HasSubstr("log-SNR bounds"));
    }
}

TEST(GGUFHadamardTypes, ExplicitF32SignsFollowF16Activations) {
    using ov::element::f16;
    const ov::PartialShape shape{-1, 8};
    ov::Tensor signs(ov::element::f32, {8});
    for (size_t i = 0; i < 8; ++i)
        signs.data<float>()[i] = i % 3 == 0 ? -1.f : 1.f;
    for (const bool inverse : {false, true}) {
        SCOPED_TRACE(inverse);
        const auto decoder = std::make_shared<ov_gguf_test::SingleOpDecoder>(
            "GGML_OP_MUL_MAT",
            std::vector<ov_gguf_test::TensorDesc>{{"x", f16, shape}},
            std::vector<ov_gguf_test::TensorDesc>{},
            ov_gguf_test::TensorDesc{"y", f16, shape},
            std::map<std::string, ov::Any>{{"hadamard_block", int64_t{4}},
                                           {"hadamard_signs", signs},
                                           {"hadamard_inverse", inverse}});
        const auto x = std::make_shared<ov::op::v0::Parameter>(f16, shape);
        auto tensor_map = std::make_shared<ov::frontend::gguf::TensorMap>();
        (*tensor_map)["x"] = x;
        const ov::frontend::gguf::NodeContext context(decoder, tensor_map);
        const auto y = ov::frontend::gguf::apply_hadamard(context, x);
        const auto model = std::make_shared<ov::Model>(ov::OutputVector{y}, ov::ParameterVector{x});
        ASSERT_EQ(y.get_element_type(), f16);
        ov::Core core;
        auto compiled = core.compile_model(model,
                                           "CPU",
                                           ov::hint::inference_precision(ov::element::f32),
                                           ov::inference_num_threads(4));
        ASSERT_EQ(compiled.input().get_element_type(), f16);
        ASSERT_EQ(compiled.output().get_element_type(), f16);
        auto request = compiled.create_infer_request();
        for (size_t rows : {1, 3}) {
            ov::Tensor input(f16, {rows, 8});
            for (size_t i = 0; i < input.get_size(); ++i)
                input.data<ov::float16>()[i] = ov::float16(static_cast<float>(i % 7) - 3.f);
            request.set_input_tensor(input);
            request.infer();
            const auto output = request.get_output_tensor();
            ASSERT_EQ(output.get_shape(), input.get_shape());
            for (size_t offset = 0; offset < input.get_size(); offset += 4) {
                for (size_t column = 0; column < 4; ++column) {
                    float expected = 0.f;
                    for (size_t row = 0; row < 4; ++row) {
                        const size_t bits = row & column;
                        const float sylvester = bits == 1 || bits == 2 ? -1.f : 1.f;
                        expected += static_cast<float>(input.data<ov::float16>()[offset + row]) * sylvester *
                                    (inverse ? 1.f : signs.data<float>()[(offset + row) % 8]);
                    }
                    expected *= .5f * (inverse ? signs.data<float>()[(offset + column) % 8] : 1.f);
                    EXPECT_FLOAT_EQ(static_cast<float>(output.data<ov::float16>()[offset + column]), expected);
                }
            }
        }
    }
}

class GGUFHadamardAccuracy : public ::testing::TestWithParam<std::tuple<int64_t, bool>> {};

TEST_P(GGUFHadamardAccuracy, ButterflyPreservesSylvesterOrderSignsAndDynamicShapes) {
    const auto block = std::get<0>(GetParam());
    const auto inverse = std::get<1>(GetParam());
    const auto width = static_cast<size_t>(2 * block);
    const ov::PartialShape shape{2, -1, 2 * block};
    ov::Tensor signs(ov::element::f32, {width});
    for (size_t i = 0; i < width; ++i)
        signs.data<float>()[i] = i % 3 == 0 ? -1.f : 1.f;
    const std::vector<int64_t> heads = block == 8 && !inverse ? std::vector<int64_t>{2, 2, 4} : std::vector<int64_t>{};
    const auto decoder = std::make_shared<ov_gguf_test::SingleOpDecoder>(
        "GGML_OP_MUL_MAT",
        std::vector<ov_gguf_test::TensorDesc>{{"x", ov::element::f32, shape}},
        std::vector<ov_gguf_test::TensorDesc>{},
        ov_gguf_test::TensorDesc{"y", ov::element::f32, shape},
        std::map<std::string, ov::Any>{{"hadamard_block", block},
                                       {"hadamard_signs", signs},
                                       {"hadamard_inverse", inverse},
                                       {"hadamard_gdn_heads", heads}});
    const auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto tensor_map = std::make_shared<ov::frontend::gguf::TensorMap>();
    (*tensor_map)["x"] = x;
    const ov::frontend::gguf::NodeContext context(decoder, tensor_map);
    const auto y = ov::frontend::gguf::apply_hadamard(context, x);
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{y}, ov::ParameterVector{x});
    size_t constant_bytes = 0;
    for (const auto& op : model->get_ordered_ops()) {
        EXPECT_NE(op->get_type_name(), std::string("MatMul"));
        if (const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(op))
            constant_bytes += constant->get_byte_size();
    }
    EXPECT_LE(constant_bytes, signs.get_byte_size() + 1024);
    ov::Core core;
    auto compiled =
        core.compile_model(model, "CPU", ov::hint::inference_precision(ov::element::f32), ov::inference_num_threads(4));
    auto request = compiled.create_infer_request();
    const auto sylvester_sign = [](size_t row, size_t column) {
        size_t bits = row & column;
        bool odd = false;
        while (bits) {
            odd = !odd;
            bits &= bits - 1;
        }
        return odd ? -1. : 1.;
    };
    for (size_t rows : {1, 3, 1}) {
        SCOPED_TRACE(rows);
        ov::Tensor input(ov::element::f32, {2, rows, width});
        for (size_t i = 0; i < input.get_size(); ++i)
            input.data<float>()[i] = block < 4096 ? std::sin(static_cast<float>(i) * .37f) : 0.f;
        if (block == 4096)
            for (size_t group = 0; group < input.get_size() / block; ++group)
                input.data<float>()[group * block + (group * 17 + 3) % block] = static_cast<float>(group + 1);
        std::vector<float> ordered(input.data<float>(), input.data<float>() + input.get_size());
        if (!heads.empty())
            for (size_t row = 0; row < 2 * rows; ++row)
                for (size_t a = 0; a < 2; ++a)
                    for (size_t b = 0; b < 2; ++b)
                        for (size_t d = 0; d < 4; ++d)
                            ordered[row * width + (b * 2 + a) * 4 + d] =
                                input.data<float>()[row * width + (a * 2 + b) * 4 + d];
        std::vector<float> expected(input.get_size());
        for (size_t group = 0; group < input.get_size() / block; ++group) {
            for (size_t column = 0; column < static_cast<size_t>(block); ++column) {
                double sum = 0.;
                const auto accumulate = [&](size_t row) {
                    const auto index = group * block + row;
                    const auto sign = inverse ? 1.f : signs.data<float>()[index % width];
                    sum += ordered[index] * sign * sylvester_sign(row, column);
                };
                if (block == 4096)
                    accumulate((group * 17 + 3) % block);
                else
                    for (size_t row = 0; row < static_cast<size_t>(block); ++row)
                        accumulate(row);
                expected[group * block + column] =
                    static_cast<float>(sum / std::sqrt(static_cast<double>(block))) *
                    (inverse ? signs.data<float>()[(group * block + column) % width] : 1.f);
            }
        }
        request.set_input_tensor(input);
        request.infer();
        EXPECT_EQ(request.get_output_tensor().get_shape(), input.get_shape());
        ov_gguf_test::expect_near(request.get_output_tensor(), expected, 2e-5f);
    }
}

INSTANTIATE_TEST_SUITE_P(
    BlockSizes,
    GGUFHadamardAccuracy,
    ::testing::Combine(::testing::Values(int64_t{1}, int64_t{2}, int64_t{8}, int64_t{64}, int64_t{4096}),
                       ::testing::Bool()));

TEST(GGUFDSparkAccuracy, NoncausalDraftBlocksMatchPrismCPU) {
    auto arrays = cnpy::npz_load(ov_gguf_test::test_data_dir() + "/arch_accuracy/dspark.npz");
    const ov_gguf_test::TemporaryGguf temporary(ov_gguf_test::npz_array(arrays, "model"));
    ov::frontend::gguf::FrontEnd frontend;
    auto model = frontend.convert(frontend.load(temporary.path));
    EXPECT_FALSE(ov::frontend::gguf::pass::AdaptToGenAI().run_on_model(model));
    ov::Core core;
    auto compiled = core.compile_model(model,
                                       "CPU",
                                       ov::hint::inference_precision(ov::element::f32),
                                       ov::hint::dynamic_quantization_group_size(0),
                                       ov::inference_num_threads(4));
    auto request = compiled.create_infer_request();
    for (size_t rows : {2, 3, 2}) {
        SCOPED_TRACE(rows);
        const auto count = rows + 4;
        for (const auto& input : compiled.inputs()) {
            const auto name = input.get_any_name();
            auto shape = input.get_partial_shape();
            for (size_t i = 0; i < shape.size(); ++i)
                if (shape[i].is_dynamic())
                    shape[i] = name == "dspark_ctx_feat" ? rows : count;
            ov::Tensor value(input.get_element_type(), shape.to_shape());
            std::memset(value.data(), 0, value.get_byte_size());
            if (name == "dspark_ctx_feat") {
                for (size_t i = 0; i < value.get_size(); ++i)
                    value.data<float>()[i] = std::sin(static_cast<float>(i)) * .1f;
            } else if (name == "inp_tokens" || name == "dspark_prev_tokens" || name == "inp_pos" ||
                       name == "inp_kv_idx") {
                for (size_t i = 0; i < value.get_size(); ++i)
                    value.data<int32_t>()[i] =
                        static_cast<int32_t>(i) + (name == "inp_tokens" || name == "dspark_prev_tokens" ? 1 : 0);
            }
            request.set_tensor(name, value);
        }
        request.infer();
        const auto actual = request.get_tensor("logits");
        const auto& expected = ov_gguf_test::npz_array(arrays, "logits_" + std::to_string(rows));
        ASSERT_EQ(actual.get_size(), expected.num_vals);
        ov_gguf_test::expect_nmse_below(
            ov_gguf_test::nmse(actual.data<const float>(), expected.data<float>(), actual.get_size()),
            1e-5);
        const auto hidden = request.get_tensor("dspark.hidden");
        const auto corrected = request.get_tensor("dspark.corrected_logits");
        const auto confidence = request.get_tensor("dspark.confidence_score");
        const auto* a = ov_gguf_test::npz_array(arrays, "dspark.markov_head_a.weight").data<float>();
        const auto* b = ov_gguf_test::npz_array(arrays, "dspark.markov_head_b.weight").data<float>();
        const auto* c = ov_gguf_test::npz_array(arrays, "dspark.confidence_head.weight").data<float>();
        const auto bias = ov_gguf_test::npz_array(arrays, "dspark.confidence_head.bias").data<float>()[0];
        for (size_t token = 0; token < 4; ++token) {
            double score = bias;
            for (size_t feature = 0; feature < 64; ++feature)
                score += hidden.data<const float>()[token * 64 + feature] * c[feature];
            for (size_t rank = 0; rank < 8; ++rank)
                score += a[(token + 1) * 8 + rank] * c[64 + rank];
            EXPECT_NEAR(confidence.data<const float>()[token], score, 1e-6);
            for (size_t vocab = 0; vocab < 256; ++vocab) {
                double logit = actual.data<const float>()[token * 256 + vocab];
                for (size_t rank = 0; rank < 8; ++rank)
                    logit += a[(token + 1) * 8 + rank] * b[vocab * 8 + rank];
                EXPECT_NEAR(corrected.data<const float>()[token * 256 + vocab], logit, 1e-6);
            }
        }
    }
}

// Check batching and beam reordering against independent stateful requests. The
// same synthetic checkpoints are qualified against llama.cpp below.
void check_batched_decode(const std::shared_ptr<ov::Model>& model) {
    ov::Core core;
    auto compiled = core.compile_model(model,
                                       "CPU",
                                       ov::hint::inference_precision(ov::element::f32),
                                       ov::num_streams(1),
                                       ov::inference_num_threads(4),
                                       ov::hint::dynamic_quantization_group_size(0),
                                       ov::hint::kv_cache_precision(ov::element::f16));
    auto first = compiled.create_infer_request();
    auto second = compiled.create_infer_request();
    auto batch = compiled.create_infer_request();
    const auto infer = [](ov::InferRequest& request,
                          size_t batch_size,
                          const std::vector<int64_t>& ids,
                          const std::vector<int64_t>& mask,
                          const std::vector<int64_t>& positions,
                          const std::vector<int32_t>& beams) {
        const auto set_tensor =
            [&](const char* name, const ov::element::Type& type, const ov::Shape& shape, const auto& values) {
                ov::Tensor tensor(type, shape);
                std::memcpy(tensor.data(), values.data(), tensor.get_byte_size());
                request.set_tensor(name, tensor);
            };
        set_tensor("input_ids", ov::element::i64, {batch_size, ids.size() / batch_size}, ids);
        set_tensor("attention_mask", ov::element::i64, {batch_size, mask.size() / batch_size}, mask);
        set_tensor("position_ids", ov::element::i64, {batch_size, positions.size() / batch_size}, positions);
        set_tensor("beam_idx", ov::element::i32, {batch_size}, beams);
        request.infer();
        auto output = request.get_tensor("logits");
        return std::vector<float>(output.data<const float>(), output.data<const float>() + output.get_size());
    };
    const auto compare =
        [](const std::vector<float>& actual, const std::vector<float>& first, const std::vector<float>& second) {
            auto expected = first;
            expected.insert(expected.end(), second.begin(), second.end());
            ASSERT_EQ(actual.size(), expected.size());
            ov_gguf_test::expect_nmse_below(ov_gguf_test::nmse(actual.data(), expected.data(), actual.size()), 1e-5);
        };
    auto a = infer(first, 1, {1, 2, 3}, {1, 1, 1}, {0, 1, 2}, {0});
    auto b = infer(second, 1, {2, 3}, {1, 1}, {0, 1}, {0});
    compare(infer(batch, 2, {1, 2, 3, 0, 2, 3}, {1, 1, 1, 0, 1, 1}, {0, 1, 2, 0, 0, 1}, {0, 0}), a, b);
    a = infer(first, 1, {4}, {1, 1, 1, 1}, {3}, {0});
    b = infer(second, 1, {5}, {1, 1, 1}, {2}, {0});
    compare(infer(batch, 2, {5, 4}, {0, 1, 1, 1, 1, 1, 1, 1}, {2, 3}, {1, 0}), b, a);
}

TEST(GGUFMultimodalBackboneAdaptation, QwenAndGemmaSupportBatchesAndPagedAttention) {
    for (const auto* family :
         {"qwen35", "bonsai2-hadamard", "qwen35moe", "qwen35moe-fused", "gemma4-mqa", "gemma4-moe", "gemma4-ple"}) {
        SCOPED_TRACE(family);
        auto arrays = cnpy::npz_load(ov_gguf_test::test_data_dir() + "/arch_accuracy/" + family + ".npz");
        const ov_gguf_test::TemporaryGguf temporary(ov_gguf_test::npz_array(arrays, "model"));
        if (std::string(family) == "gemma4-ple") {
            const auto loaded = ov::frontend::gguf::get_gguf_data(temporary.path);
            const auto& weights = std::get<1>(loaded);
            EXPECT_EQ(weights.count("blk.0.attn_k.weight"), 1);
            EXPECT_EQ(weights.count("blk.0.attn_v.weight"), 1);
            EXPECT_EQ(weights.count("blk.1.attn_k.weight"), 1);
            for (const auto* name :
                 {"blk.2.attn_k.weight", "blk.2.attn_v.weight", "blk.3.attn_k.weight", "blk.3.attn_v.weight"}) {
                EXPECT_EQ(weights.count(name), 0) << name;
            }
        }
        ov::frontend::gguf::FrontEnd frontend;
        frontend.add_extension(std::make_shared<ov::frontend::gguf::GenAIExtension>());
        auto model = frontend.convert(frontend.load(temporary.path));
        check_batched_decode(model);
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::SDPAToPagedAttention>();
        ASSERT_NO_THROW(manager.run_passes(model));
        // Multiple tokens expose accidental [tokens,tokens,...] M-RoPE broadcasts.
        model->reshape({{"input_ids", {5}}, {"position_ids", {5}}});
        const bool gemma = std::string(family).find("gemma4") == 0;
        size_t attention = 0, recurrent = 0;
        for (const auto& node : model->get_ops()) {
            if (auto pa = ov::as_type_ptr<ov::op::PagedAttentionExtension>(node)) {
                ++attention;
                const auto& query_shape = pa->get_input_partial_shape(0);
                EXPECT_EQ(query_shape.rank(), 2);
                EXPECT_EQ(query_shape[0], 5);
                EXPECT_TRUE(query_shape[1] == 64 || (gemma && query_shape[1] == 32));
            }
            if (auto gdn = ov::as_type_ptr<ov::op::internal::PagedGatedDeltaNet>(node)) {
                ++recurrent;
                EXPECT_EQ(gdn->get_input_partial_shape(0)[0], 5);
            }
        }
        EXPECT_EQ(attention, gemma ? (std::string(family) == "gemma4-ple" ? 4 : 2) : 1);
        EXPECT_EQ(recurrent, gemma ? 0 : 3);
        EXPECT_TRUE(model->get_sinks().empty());
    }
}

// Gemma3/Gemma4 scale token lookups only, as in llama.cpp, so the scale belongs to the embedding
// model and injected media embeddings reach the decoder unscaled.
TEST(GGUFMultimodalBackboneAdaptation, GemmaEmbeddingModelOwnsTokenScaling) {
    for (const auto* family : {"gemma3", "gemma4-mqa", "gemma4-ple"}) {
        SCOPED_TRACE(family);
        auto arrays = cnpy::npz_load(ov_gguf_test::test_data_dir() + "/arch_accuracy/" + family + ".npz");
        const ov_gguf_test::TemporaryGguf temporary(ov_gguf_test::npz_array(arrays, "model"));
        const auto convert = [&] {
            ov::frontend::gguf::FrontEnd frontend;
            frontend.add_extension(std::make_shared<ov::frontend::DecoderTransformationExtension>(
                ov::frontend::gguf::pass::GGUFMakeStateful()));
            return frontend.convert(frontend.load(temporary.path));
        };
        auto token_model = convert();
        ov::frontend::gguf::pass::AdaptToGenAI().run_on_model(token_model);
        ov::frontend::gguf::FrontEnd frontend;
        auto extension = std::make_shared<ov::frontend::gguf::GenAIExtension>(
            ov::frontend::gguf::GenAIExtension::InputMode::EMBEDS_TO_LOGITS);
        frontend.add_extension(extension);
        const auto input = frontend.load(temporary.path);
        auto embedded_model = frontend.convert(input);
        ASSERT_TRUE(extension->get_embedding_model());
        EXPECT_EQ(bool(extension->get_per_layer_embedding_model()), std::string(family) == "gemma4-ple");
        embedded_model = frontend.convert(input);
        const auto embedding_model = extension->get_embedding_model();
        const auto width = embedding_model->output("inputs_embeds").get_partial_shape()[2].get_length();
        const auto embedding_ops = embedding_model->get_ops();
        const bool scaled =
            std::any_of(embedding_ops.begin(), embedding_ops.end(), [&](const std::shared_ptr<ov::Node>& node) {
                if (!ov::is_type<ov::op::v1::Multiply>(node))
                    return false;
                auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node->get_input_node_shared_ptr(1));
                return constant && ov::shape_size(constant->get_shape()) == 1 &&
                       std::abs(constant->cast_vector<float>()[0] - std::sqrt(float(width))) < 1e-4f;
            });
        EXPECT_TRUE(scaled) << "embedding model does not apply sqrt(n_embd)";

        ov::Core core;
        const auto compile = [&](const std::shared_ptr<ov::Model>& model) {
            return core.compile_model(model, "CPU", ov::hint::inference_precision(ov::element::f32))
                .create_infer_request();
        };
        auto tokens = compile(token_model), values = compile(embedded_model), lookup = compile(embedding_model);
        ov::Tensor ids(ov::element::i64, {1, 3}), mask(ov::element::i64, {1, 3}), positions(ov::element::i64, {1, 3});
        for (int64_t i = 0; i < 3; ++i) {
            ids.data<int64_t>()[i] = i + 1;
            positions.data<int64_t>()[i] = i;
        }
        std::fill_n(mask.data<int64_t>(), 3, 1);
        ov::Tensor beam(ov::element::i32, {1});
        beam.data<int32_t>()[0] = 0;
        lookup.set_tensor("input_ids", ids);
        lookup.infer();
        for (auto* request : {&tokens, &values}) {
            request->set_tensor("attention_mask", mask);
            request->set_tensor("position_ids", positions);
            request->set_tensor("beam_idx", beam);
        }
        tokens.set_tensor("input_ids", ids);
        values.set_tensor("inputs_embeds", lookup.get_tensor("inputs_embeds"));
        for (const auto& input : embedded_model->inputs()) {
            if (input.get_names().count("token_type_ids")) {
                ov::Tensor types(ov::element::i64, {1, 3});
                std::fill_n(types.data<int64_t>(), 3, 0);
                values.set_tensor("token_type_ids", types);
            }
            if (input.get_names().count("per_layer_inputs")) {
                auto per_layer = compile(extension->get_per_layer_embedding_model());
                per_layer.set_tensor("input_ids", ids);
                per_layer.infer();
                values.set_tensor("per_layer_inputs", per_layer.get_tensor("per_layer_inputs"));
            }
        }
        tokens.infer();
        values.infer();
        const auto expected = tokens.get_output_tensor(), actual = values.get_output_tensor();
        ASSERT_EQ(actual.get_shape(), expected.get_shape());
        ov_gguf_test::expect_nmse_below(
            ov_gguf_test::nmse(actual.data<const float>(), expected.data<const float>(), actual.get_size()),
            1e-10);
    }
}

// Adapters registered as transformation extensions run during conversion and must already see
// the architecture and projector metadata they select their changes by.
TEST(GGUFMultimodalBackboneAdaptation, AdaptersRegisteredAsExtensionsSeeModelMetadata) {
    using ov::frontend::DecoderTransformationExtension;
    namespace pass = ov::frontend::gguf::pass;
    const auto convert = [](const std::string& fixture, const std::vector<std::shared_ptr<ov::Extension>>& passes) {
        auto arrays = cnpy::npz_load(ov_gguf_test::test_data_dir() + "/" + fixture);
        const ov_gguf_test::TemporaryGguf temporary(ov_gguf_test::npz_array(arrays, "model"));
        ov::frontend::gguf::FrontEnd frontend;
        for (const auto& extension : passes)
            frontend.add_extension(extension);
        return frontend.convert(frontend.load(temporary.path));
    };
    const auto names = [](const auto& ports) {
        std::set<std::string> result;
        for (const auto& port : ports)
            result.insert(port.get_any_name());
        return result;
    };
    const auto language = convert("arch_accuracy/gemma3.npz",
                                  {std::make_shared<DecoderTransformationExtension>(pass::GGUFMakeStateful()),
                                   std::make_shared<DecoderTransformationExtension>(
                                       pass::AdaptToGenAI(pass::AdaptToGenAI::InputMode::EMBEDS_TO_LOGITS))});
    EXPECT_EQ(names(language->inputs()).count("token_type_ids"), 1);
    const auto vision = convert("mmproj_accuracy/qwen3vl_merger.npz",
                                {std::make_shared<DecoderTransformationExtension>(
                                    pass::AdaptMmprojToGenAI(pass::AdaptMmprojToGenAI::Modality::VISION))});
    EXPECT_EQ(names(vision->outputs()), (std::set<std::string>{"image_features", "deepstack_features.0"}));
}

// The Gated-DeltaNet key-head count divides the value-head count; a missing or zero one must
// fail conversion with a message instead of dividing by zero.
TEST(GGUFMultimodalBackboneAdaptation, GatedDeltaNetRejectsMissingOrZeroGroupCount) {
    auto arrays = cnpy::npz_load(ov_gguf_test::test_data_dir() + "/arch_accuracy/qwen35.npz");
    const auto& model = ov_gguf_test::npz_array(arrays, "model");
    const std::string key = "qwen35.ssm.group_count";
    for (const bool missing : {false, true}) {
        SCOPED_TRACE(missing ? "missing" : "zero");
        std::vector<uint8_t> bytes(model.data<uint8_t>(), model.data<uint8_t>() + model.num_vals);
        const auto at = std::search(bytes.begin(), bytes.end(), key.begin(), key.end());
        ASSERT_NE(at, bytes.end());
        if (missing) {
            at[key.size() - 1] = 'X';  // the loader then defaults the count to zero
        } else {
            const auto value = at + key.size() + sizeof(uint32_t);  // after the u32 type tag
            std::fill(value, value + sizeof(uint32_t), 0);
        }
        cnpy::NpyArray patched({bytes.size()}, sizeof(uint8_t), false);
        std::copy(bytes.begin(), bytes.end(), patched.data<uint8_t>());
        const ov_gguf_test::TemporaryGguf temporary(patched);
        ov::frontend::gguf::FrontEnd frontend;
        OV_EXPECT_THROW(frontend.convert(frontend.load(temporary.path)),
                        ov::Exception,
                        testing::HasSubstr("ssm.group_count"));
    }
}

// The grouped-query KV broadcast may reach MakeStateful as Multiply(ones, Unsqueeze(kv)); the
// cache read behind it must still be found, so the fused cache gets its batch-shaped initializer.
TEST(GGUFMultimodalBackboneAdaptation, MakeStatefulFindsCachesBehindReversedGroupedQueryMultiply) {
    auto arrays = cnpy::npz_load(ov_gguf_test::test_data_dir() + "/arch_accuracy/qwen3.npz");
    const ov_gguf_test::TemporaryGguf temporary(ov_gguf_test::npz_array(arrays, "model"));
    for (const bool reversed : {false, true}) {
        SCOPED_TRACE(reversed ? "ones * kv" : "kv * ones");
        size_t rewritten = 0;
        const auto to_multiply = [&](const std::shared_ptr<ov::Model>& model) {
            for (const auto& node : model->get_ordered_ops()) {
                auto broadcast = ov::as_type_ptr<ov::op::v3::Broadcast>(node);
                auto shape = broadcast ? ov::as_type_ptr<ov::op::v0::Constant>(broadcast->get_input_node_shared_ptr(1))
                                       : nullptr;
                if (!shape || !ov::is_type<ov::op::v0::Unsqueeze>(broadcast->get_input_node_ptr(0)))
                    continue;
                auto ones = ov::op::v0::Constant::create(broadcast->get_output_element_type(0),
                                                         ov::Shape(shape->cast_vector<size_t>()),
                                                         {1.f});
                const auto kv = broadcast->input_value(0);
                ov::replace_node(broadcast,
                                 reversed ? std::make_shared<ov::op::v1::Multiply>(ones, kv)
                                          : std::make_shared<ov::op::v1::Multiply>(kv, ones));
                broadcast->input(0).replace_source_output(ones);  // the replaced node no longer reads kv
                ++rewritten;
            }
            return rewritten > 0;
        };
        ov::frontend::gguf::FrontEnd frontend;
        frontend.add_extension(std::make_shared<ov::frontend::DecoderTransformationExtension>(to_multiply));
        frontend.add_extension(std::make_shared<ov::frontend::DecoderTransformationExtension>(
            ov::frontend::gguf::pass::GGUFMakeStateful()));
        std::shared_ptr<ov::Model> model;
        ASSERT_NO_THROW(model = frontend.convert(frontend.load(temporary.path)));
        ASSERT_GT(rewritten, 0u);
        size_t batch_initialized = 0;
        for (const auto& node : model->get_ordered_ops()) {
            if (ov::is_type<ov::op::v6::ReadValue>(node) && node->get_input_size() == 1 &&
                ov::is_type<ov::op::v3::Broadcast>(node->get_input_node_ptr(0)))
                ++batch_initialized;
        }
        EXPECT_EQ(batch_initialized, rewritten);
    }
}

TEST(GGUFEmbeddingAccuracy, TokenEmbeddingsAndPoolingMatchLlamaCPU) {
    for (const auto* family :
         {"llama-embed", "llama-embed-noncausal", "llama-embed-mean", "llama-embed-cls", "llama-embed-last"}) {
        SCOPED_TRACE(family);
        auto arrays = cnpy::npz_load(ov_gguf_test::test_data_dir() + "/arch_accuracy/" + family + ".npz");
        const ov_gguf_test::TemporaryGguf temporary(ov_gguf_test::npz_array(arrays, "model"));
        ov::frontend::gguf::FrontEnd frontend;
        auto model = frontend.convert(frontend.load(temporary.path));
        ASSERT_EQ(model->get_results().size(), 1);
        EXPECT_NO_THROW(model->output("embeddings"));
        for (const auto mode : {ov::frontend::gguf::pass::AdaptToGenAI::InputMode::IDS_TO_LOGITS,
                                ov::frontend::gguf::pass::AdaptToGenAI::InputMode::EMBEDS_TO_LOGITS}) {
            ov::frontend::gguf::pass::AdaptToGenAI adapter(mode);
            EXPECT_FALSE(adapter.run_on_model(model));
        }
        ov::Core core;
        auto compiled = core.compile_model(model,
                                           "CPU",
                                           ov::hint::inference_precision(ov::element::f32),
                                           ov::hint::dynamic_quantization_group_size(0));
        auto request = compiled.create_infer_request();
        EXPECT_TRUE(request.query_state().empty());
        const auto infer = [&](size_t count, bool compare_reference) {
            for (const auto& input : compiled.inputs()) {
                const auto name = input.get_any_name();
                if (name == "token_len_per_seq") {
                    ov::Tensor lengths(ov::element::i64, {1});
                    lengths.data<int64_t>()[0] = count;
                    request.set_tensor(name, lengths);
                } else if (name == "self_kq_mask") {
                    ov::Tensor mask(ov::element::f32, {1, 1, count, count});
                    for (size_t q = 0; q < count; ++q)
                        for (size_t k = 0; k < count; ++k)
                            mask.data<float>()[q * count + k] = k <= q ? 0.f : -INFINITY;
                    request.set_tensor(name, mask);
                } else {
                    ASSERT_TRUE(name == "inp_tokens" || name == "inp_pos");
                    ov::Tensor values(ov::element::i32, {1, 1, 1, count});
                    for (size_t i = 0; i < count; ++i)
                        values.data<int32_t>()[i] = static_cast<int32_t>(i) + (name == "inp_tokens" ? 1 : 0);
                    request.set_tensor(name, values);
                }
            }
            request.infer();
            auto output = request.get_tensor("embeddings");
            const auto& expected = ov_gguf_test::npz_array(arrays, "embeddings");
            const bool pooled = std::string(family) != "llama-embed" && std::string(family) != "llama-embed-noncausal";
            ASSERT_EQ(output.get_shape(), (ov::Shape{pooled ? 1 : count, expected.shape.back()}));
            if (compare_reference || std::string(family) == "llama-embed") {
                ov_gguf_test::expect_nmse_below(
                    ov_gguf_test::nmse(output.data<const float>(), expected.data<float>(), output.get_size()),
                    1e-5);
            }
        };
        infer(3, true);
        infer(1, false);
        infer(2, false);
        infer(3, true);
    }
}

TEST(GGUFEmbeddingAccuracy, RealCheckpointTokenEmbeddingsMatchLlamaCPU) {
    const auto* directory = std::getenv("OV_GGUF_EMBEDDING_DATA");
    if (!directory)
        GTEST_SKIP() << "Set OV_GGUF_EMBEDDING_DATA for real embedding checkpoint validation";
    const auto base = std::filesystem::path(directory) / "llama-embed";
    std::ifstream file(base.string() + ".bin", std::ios::binary);
    ASSERT_TRUE(file);
    int32_t width = 0, count = 0;
    file.read(reinterpret_cast<char*>(&width), sizeof(width));
    file.read(reinterpret_cast<char*>(&count), sizeof(count));
    ASSERT_GT(width, 0);
    ASSERT_GT(count, 0);
    ASSERT_LE(count, 64);
    std::vector<int32_t> tokens(count);
    std::vector<float> expected(static_cast<size_t>(count) * width);
    file.read(reinterpret_cast<char*>(tokens.data()), tokens.size() * sizeof(int32_t));
    file.read(reinterpret_cast<char*>(expected.data()), expected.size() * sizeof(float));
    ASSERT_TRUE(file);
    ov::frontend::gguf::FrontEnd frontend;
    auto model = frontend.convert(frontend.load(base.string() + ".gguf"));
    ov::Core core;
    auto compiled = core.compile_model(model,
                                       "CPU",
                                       ov::hint::inference_precision(ov::element::f32),
                                       ov::hint::dynamic_quantization_group_size(0),
                                       ov::inference_num_threads(4));
    auto request = compiled.create_infer_request();
    for (const auto& input : compiled.inputs()) {
        const auto name = input.get_any_name();
        if (name == "token_len_per_seq") {
            ov::Tensor lengths(ov::element::i64, {1});
            lengths.data<int64_t>()[0] = count;
            request.set_tensor(name, lengths);
        } else if (name == "self_kq_mask") {
            ov::Tensor mask(ov::element::f32, {1, 1, size_t(count), size_t(count)});
            for (int q = 0; q < count; ++q)
                for (int k = 0; k < count; ++k)
                    mask.data<float>()[q * count + k] = k <= q ? 0.f : -INFINITY;
            request.set_tensor(name, mask);
        } else {
            ASSERT_TRUE(name == "inp_tokens" || name == "inp_pos");
            ov::Tensor values(ov::element::i32, {1, 1, 1, size_t(count)});
            for (int i = 0; i < count; ++i)
                values.data<int32_t>()[i] = name == "inp_tokens" ? tokens[i] : i;
            request.set_tensor(name, values);
        }
    }
    request.infer();
    auto output = request.get_tensor("embeddings");
    ASSERT_EQ(output.get_shape(), (ov::Shape{size_t(count), size_t(width)}));
    const auto metric = ov_gguf_test::nmse(output.data<const float>(), expected.data(), expected.size());
    RecordProperty("nmse", std::to_string(metric.value()));
    ov_gguf_test::expect_nmse_below(metric, 1e-5);
}

TEST_P(GGUFArchitectureAccuracy, PrefillAndCachedDecodeMatchLlamaCPU) {
    const char* override_dir = std::getenv("OV_GGUF_ACCURACY_DATA");
    const bool mamba = std::string(GetParam()).find("mamba2") == 0 || std::string(GetParam()) == "nemotron_h";
    const auto directory = override_dir ? std::filesystem::path(override_dir)
                                        : std::filesystem::path(ov_gguf_test::test_data_dir()) / "arch_accuracy";
    ASSERT_TRUE(std::filesystem::exists(directory)) << "Missing architecture reference data: " << directory;
    const auto base = directory / GetParam();
    const bool real_checkpoint = override_dir && !std::filesystem::exists(base.string() + ".npz");
    std::vector<std::vector<int64_t>> schedule{{1, 2, 3}, {4}, {5, 6}};
    std::vector<float> reference;
    int32_t vocab = 0;
    std::string model_path = base.string() + ".gguf";
    std::unique_ptr<ov_gguf_test::TemporaryGguf> temporary;
    if (real_checkpoint) {
        std::ifstream file(base.string() + ".bin", std::ios::binary);
        ASSERT_TRUE(file) << base;
        file.read(reinterpret_cast<char*>(&vocab), sizeof(vocab));
        ASSERT_GT(vocab, 6);
        ASSERT_LT(vocab, 1000000);
        std::ifstream token_file(base.string() + ".bin.tokens");
        if (token_file) {
            schedule.clear();
            size_t count = 0;
            while (token_file >> count) {
                ASSERT_GT(count, 0);
                ASSERT_LE(count, 64);
                std::vector<int64_t> ids(count);
                for (auto& token : ids)
                    ASSERT_TRUE(token_file >> token);
                schedule.push_back(std::move(ids));
            }
        }
        ASSERT_FALSE(schedule.empty());
        reference.resize(schedule.size() * vocab);
        file.read(reinterpret_cast<char*>(reference.data()), reference.size() * sizeof(float));
        ASSERT_TRUE(file);
    } else {
        const auto arrays = cnpy::npz_load(base.string() + ".npz");
        const auto array = [&](const std::string& name) -> const cnpy::NpyArray& {
            return ov_gguf_test::npz_array(arrays, name);
        };
        const auto& logits = array("logits");
        ASSERT_EQ(logits.shape.size(), 2);
        ASSERT_EQ(logits.shape[0], 3);
        vocab = static_cast<int32_t>(logits.shape[1]);
        const auto* values = logits.data<float>();
        reference.assign(values, values + 3 * vocab);
        temporary = std::make_unique<ov_gguf_test::TemporaryGguf>(array("model"));
        model_path = temporary->path;
    }
    ov::frontend::gguf::FrontEnd fe;
    fe.add_extension(std::make_shared<ov::frontend::gguf::GenAIExtension>());
    auto model = fe.convert(fe.load(model_path));
    if (mamba) {
        EXPECT_EQ(model->input("input_ids").get_partial_shape(), (ov::PartialShape{1, -1}));
        EXPECT_NO_THROW(model->input("beam_idx"));
    }
    ov::Core core;
    auto compiled = core.compile_model(model,
                                       "CPU",
                                       ov::hint::inference_precision(ov::element::f32),
                                       ov::num_streams(1),
                                       ov::inference_num_threads(4),
                                       ov::hint::dynamic_quantization_group_size(0),
                                       ov::hint::kv_cache_precision(ov::element::f16));
    auto request = compiled.create_infer_request();
    const auto infer = [&](const std::vector<int64_t>& tokens, size_t past) {
        const auto count = tokens.size();
        ov::Tensor ids(ov::element::i64, {1, count});
        ov::Tensor positions(ov::element::i64, ids.get_shape());
        ov::Tensor mask(ov::element::i64, {1, past + count});
        ov::Tensor beam(ov::element::i32, {1});
        *beam.data<int32_t>() = 0;
        for (size_t i = 0; i < count; ++i) {
            ids.data<int64_t>()[i] = tokens[i];
            positions.data<int64_t>()[i] = static_cast<int64_t>(past + i);
        }
        std::fill_n(mask.data<int64_t>(), mask.get_size(), 1);
        request.set_tensor("input_ids", ids);
        request.set_tensor("position_ids", positions);
        request.set_tensor("attention_mask", mask);
        if (std::any_of(compiled.inputs().begin(), compiled.inputs().end(), [](const ov::Output<const ov::Node>& p) {
                return p.get_names().count("beam_idx");
            }))
            request.set_tensor("beam_idx", beam);
        request.infer();
        return request.get_output_tensor();
    };
    size_t past = 0;
    size_t step = 0;
    size_t matching_tokens = 0;
    for (const auto& tokens : schedule) {
        SCOPED_TRACE("past=" + std::to_string(past) + ", tokens=" + std::to_string(tokens.size()));
        const auto result = infer(tokens, past);
        ASSERT_EQ(result.get_element_type(), ov::element::f32);
        ASSERT_GE(result.get_size(), static_cast<size_t>(vocab));
        const auto* actual = result.data<const float>() + result.get_size() - vocab;
        const auto* expected = reference.data() + step++ * vocab;
        const auto metric = ov_gguf_test::nmse(actual, expected, static_cast<size_t>(vocab));
        ASSERT_TRUE(metric.all_finite());
        ASSERT_GT(metric.reference_norm(), 1e-12) << "Reference must contain nonzero logits";
        const auto predicted = std::max_element(actual, actual + vocab) - actual;
        const auto wanted = std::max_element(expected, expected + vocab) - expected;
        matching_tokens += predicted == wanted;
        RecordProperty("top1_match_step_" + std::to_string(step), predicted == wanted ? 1 : 0);
        RecordProperty("nmse_step_" + std::to_string(step), std::to_string(metric.value()));
        if (real_checkpoint) {
            // Real checkpoints can use different quantization arithmetic. Check the first prediction
            // and continuation agreement; keep their full-logit metrics in the XML report.
            if (step == 1) {
                EXPECT_EQ(predicted, wanted);
            }
        } else {
            EXPECT_LT(metric.value(), 1e-5) << "Normalized MSE against llama.cpp CPU";
        }
        past += tokens.size();
    }
    if (mamba) {
        const auto states = request.query_state();
        ASSERT_FALSE(states.empty());
        for (auto state : states)
            state.reset();
        const auto logits = infer(schedule.front(), 0);
        const auto* actual = logits.data<const float>() + logits.get_size() - vocab;
        const auto metric = ov_gguf_test::nmse(actual, reference.data(), static_cast<size_t>(vocab));
        ASSERT_TRUE(metric.all_finite());
        ASSERT_GT(metric.reference_norm(), 1e-12);
        if (real_checkpoint)
            EXPECT_EQ(std::max_element(actual, actual + vocab) - actual,
                      std::max_element(reference.begin(), reference.begin() + vocab) - reference.begin());
        else
            EXPECT_LT(metric.value(), 1e-5) << "Fresh prefill after resetting recurrent states";
    }
    if (real_checkpoint) {
        EXPECT_GE(matching_tokens * 10, schedule.size() * 9) << "Fewer than 90% of greedy choices match";
    }
}

INSTANTIATE_TEST_SUITE_P(Architectures,
                         GGUFArchitectureAccuracy,
                         ::testing::Values("qwen35",
                                           "bonsai2-hadamard",
                                           "qwen35-mixed",
                                           "qwen35moe",
                                           "qwen35moe-fused",
                                           "gemma4-mqa",
                                           "gemma4-moe",
                                           "gemma4-ple",
                                           "nemotron_h",
                                           "mamba2",
                                           "mamba2-tied",
                                           "llama",
                                           "qwen2",
                                           "qwen3",
                                           "phi3",
                                           "minicpm",
                                           "olmoe",
                                           "hunyuan-dense",
                                           "hunyuan-moe",
                                           "exaone-moe",
                                           "exaone-moe-nextn",
                                           "glm4moe",
                                           "jais2",
                                           "minimax-m2",
                                           "plamo3",
                                           "qwen3moe",
                                           "gemma",
                                           "gemma2",
                                           "gemma3",
                                           "exaone4",
                                           "ernie4_5-moe",
                                           "bailingmoe2",
                                           "maincoder",
                                           "mistral3",
                                           "smollm3",
                                           "mellum",
                                           "muse-glimmer",
                                           "deepseek2-ocr",
                                           "devstral-small",
                                           "devstral-small2",
                                           "devstral2"),
                         [](const ::testing::TestParamInfo<const char*>& info) {
                             std::string name = info.param;
                             std::replace(name.begin(), name.end(), '-', '_');
                             return name;
                         });
}  // namespace
