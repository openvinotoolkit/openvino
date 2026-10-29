// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// End-to-end tests for weights surfaced as GGML_OP_NONE (leaf) nodes.
//
// A weight is a regular node in the gguf frontend: a ggml leaf (op type "GGML_OP_NONE") that
// the decoder marks as a weight by providing the raw GGUF bytes (get_attribute<ov::Tensor>
// ("data")), the ggml quant type name (get_attribute<std::string>("quant_type")) and the
// logical [rows, cols] shape (get_output_shape()); the frontend's translate_weight does the
// dequant. These tests build a single GGML_OP_NONE model -- exactly like the single-op tests in
// test_ops.cpp -- run it on CPU and compare against the same real-ggml reference data used by
// test_dequant_vs_ggml.cpp.

#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>

#include "builder/gguf_builder_decoder.hpp"
#include "builder/graph_emitter.hpp"
#include "common_test_utils/file_utils.hpp"
#include "op_test_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "ov_ops/fully_connected_compressed.hpp"
#include "quant/gguf.hpp"
#include "quant/weights.hpp"

using namespace ov_gguf_test;

namespace {

// rows/cols and tolerance match the reference generator (see test_dequant_vs_ggml.cpp).
// Q5_K/Q6_K requantize to channel-wise Q8_0_C (matching the llama.cpp ggml-openvino CPU/GPU
// backend), so they diverge from ggml's faithful to_float by the Q8_0_C round-off rather
// than f16 noise.
constexpr size_t kRows = 4;
constexpr size_t kCols = 256;
constexpr float kTolFaithful = 3e-3f;
constexpr float kTolRequant = 1.5e-2f;
// Native Q4_K group-wise requantization improves on the former 5e-2 integer-zp tolerance.
constexpr float kTolU4Requant = 4e-2f;

struct WeightCase {
    const char* stem;        // test_data prefix
    const char* quant_type;  // name passed to the frontend
    float tol;
};

ov::Tensor bytes_to_u8_tensor(const std::vector<uint8_t>& bytes) {
    ov::Tensor t(ov::element::u8, ov::Shape{bytes.size()});
    std::memcpy(t.data(), bytes.data(), bytes.size());
    return t;
}

}  // namespace

class GGUFWeight : public ::testing::TestWithParam<WeightCase> {};

TEST(GGUFRawCompressedFC, PreservesPayloadAndInfersFloatingOutputs) {
    const auto bytes = load_npy<uint8_t>("q4_k_qbytes");
    ov::Tensor storage(ov::element::gguf_q4_k, {kRows, kCols});
    ASSERT_EQ(storage.get_byte_size(), bytes.size());
    std::memcpy(storage.data(), bytes.data(), bytes.size());
    const auto weight = std::make_shared<ov::op::v0::Constant>(storage);
    const auto absent = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{0});
    const std::vector<ov::PartialShape> inputs{{kCols}, {-1, kCols}, {2, -1, kCols}};
    const std::vector<ov::PartialShape> outputs{{kRows}, {-1, kRows}, {2, -1, kRows}};
    for (const auto output_type : {ov::element::f32, ov::element::f16}) {
        for (size_t index = 0; index < inputs.size(); ++index) {
            const auto activation = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputs[index]);
            const auto fc = std::make_shared<ov::op::internal::FullyConnectedCompressed>(activation,
                                                                                        weight,
                                                                                        absent,
                                                                                        absent,
                                                                                        absent,
                                                                                        output_type);
            EXPECT_EQ(fc->get_output_partial_shape(0), outputs[index]);
            EXPECT_EQ(fc->get_output_element_type(0), output_type);
            const auto clone = fc->clone_with_new_inputs({activation, weight, absent, absent, absent});
            EXPECT_EQ(clone->get_output_partial_shape(0), outputs[index]);
            EXPECT_EQ(clone->get_output_element_type(0), output_type);
            EXPECT_EQ(clone->get_input_node_shared_ptr(1), weight);
        }
    }
    const auto activation = ov::op::v0::Constant::create(ov::element::f32, {2, kCols}, {0.f});
    const auto fc = std::make_shared<ov::op::internal::FullyConnectedCompressed>(activation,
                                                                                weight,
                                                                                absent,
                                                                                absent,
                                                                                absent,
                                                                                ov::element::f32);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{});
    ov::pass::ConstantFolding().run_on_model(model);
    EXPECT_EQ(model->get_results().at(0)->get_input_node_shared_ptr(0), fc);
    EXPECT_EQ(fc->get_input_node_shared_ptr(1), weight);
    EXPECT_EQ(weight->get_data_ptr(), storage.data());
    EXPECT_EQ(std::memcmp(weight->get_data_ptr(), bytes.data(), bytes.size()), 0);
}

TEST(GGUFRawCompressedFC, RejectsIncompatibleReductionDimension) {
    const auto activation = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, kCols / 2});
    const auto weight = std::make_shared<ov::op::v0::Constant>(ov::element::gguf_q4_k, ov::Shape{kRows, kCols});
    const auto absent = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{0});
    EXPECT_ANY_THROW(std::make_shared<ov::op::internal::FullyConnectedCompressed>(activation,
                                                                               weight,
                                                                               absent,
                                                                               absent,
                                                                               absent,
                                                                               ov::element::f32));
}

TEST(GGUFRawCompressedFC, SharedEmitterPreservesRawWeightThroughConversion) {
    const auto bytes = load_npy<uint8_t>("q4_k_qbytes");
    for (const auto activation_type : {ov::element::f32, ov::element::f16}) {
        std::unordered_map<std::string, ov::Tensor> weights;
        weights.emplace("projection.weight", ov::Tensor(ov::element::gguf_q4_k, {kRows, kCols}));
        std::memcpy(weights.at("projection.weight").data(), bytes.data(), bytes.size());
        std::unordered_map<std::string, GgufTensorType> qtypes{{"projection.qtype", GGUF_TYPE_Q4_K}};
        GraphEmitter emitter(weights, qtypes, "test");
        emitter.add_input("activation", activation_type, {1, 1, -1, kCols});
        emitter.add_weight("projection.weight");
        emitter.add_op("GGML_OP_MUL_MAT", "first", {"projection.weight", "activation"});
        emitter.add_op("GGML_OP_MUL_MAT", "second", {"projection.weight", "activation"});
        emitter.graph()->model_output_names = {"first", "second"};
        std::shared_ptr<GgufDecoder> decoder = std::make_shared<GgufBuilderDecoder>(emitter.graph());
        FrontEnd frontend;
        const auto model = frontend.convert(frontend.load(decoder));
        std::shared_ptr<ov::Node> shared_weight;
        for (const auto& result : model->get_results()) {
            const auto fc = ov::as_type_ptr<ov::op::internal::FullyConnectedCompressed>(
                result->get_input_node_shared_ptr(0));
            ASSERT_TRUE(fc);
            EXPECT_EQ(fc->get_output_partial_shape(0), (ov::PartialShape{1, 1, -1, kRows}));
            EXPECT_EQ(fc->get_output_element_type(0), ov::element::f32);
            const auto raw = ov::as_type_ptr<ov::op::v0::Constant>(fc->get_input_node_shared_ptr(1));
            ASSERT_TRUE(raw);
            EXPECT_EQ(raw->get_element_type(), ov::element::gguf_q4_k);
            EXPECT_EQ(raw->get_data_ptr(), weights.at("projection.weight").data());
            EXPECT_EQ(raw->get_byte_size(), bytes.size());
            EXPECT_EQ(std::memcmp(raw->get_data_ptr(), bytes.data(), bytes.size()), 0);
            if (shared_weight) {
                EXPECT_EQ(raw, shared_weight);
            }
            shared_weight = raw;
            for (size_t input = 2; input < 5; ++input) {
                EXPECT_EQ(fc->get_input_shape(input), (ov::Shape{0}));
            }
        }
    }
}

TEST(GGUFRawCompressedFC, RejectsUnsupportedConsumersAndMatmulCases) {
    for (const auto qtype : {GGUF_TYPE_Q4_K, GGUF_TYPE_Q5_K, GGUF_TYPE_Q6_K, GGUF_TYPE_Q8_0}) {
        SCOPED_TRACE(qtype);
        std::unordered_map<std::string, ov::Tensor> weights;
        weights.emplace("projection.weight", ov::Tensor(gguf_raw_element_type(qtype), {kRows, kCols}));
        std::unordered_map<std::string, GgufTensorType> qtypes{{"projection.qtype", qtype}};
        GraphEmitter emitter(weights, qtypes, "test");
        emitter.add_input("activation", ov::element::f32, {1, 1, 2, kCols});
        emitter.add_weight("projection.weight");
        for (const auto op_case : {1, 2, 3}) {
            EXPECT_ANY_THROW(emitter.add_op("GGML_OP_MUL_MAT",
                                           "unsupported_case",
                                           {"projection.weight", "activation"},
                                           op_case));
        }
        emitter.add_input("indices", ov::element::i32, {2});
        EXPECT_THROW(emitter.add_op("GGML_OP_GET_ROWS", "embedding", {"projection.weight", "indices"}),
                     ov::frontend::OpConversionFailure);
        emitter.add_input("embedding_indices", ov::element::i64, {1, 1, 1, -1});
        for (const auto op_case : {2, 3, 4, 10}) {
            EXPECT_THROW(emitter.add_op("GGML_OP_GET_ROWS", "unsupported_embedding",
                                       {"projection.weight", "embedding_indices"}, op_case),
                         ov::frontend::OpConversionFailure);
        }
        EXPECT_THROW(emitter.add_op("GGML_OP_ADD", "add", {"projection.weight", "activation"}),
                     ov::frontend::OpConversionFailure);
        EXPECT_THROW(emitter.add_op("GGML_OP_MUL_MAT", "raw_activation", {"activation", "projection.weight"}),
                     ov::frontend::OpConversionFailure);
    }
}

TEST(GGUFRawCompressedFC, RawRowSplitsPreserveBlocks) {
    for (const auto qtype : {GGUF_TYPE_Q4_K, GGUF_TYPE_Q5_K, GGUF_TYPE_Q6_K, GGUF_TYPE_Q8_0}) {
        SCOPED_TRACE(qtype);
        ov::Tensor raw(gguf_raw_element_type(qtype), {8, kCols});
        auto* bytes = static_cast<uint8_t*>(raw.data());
        for (size_t index = 0; index < raw.get_byte_size(); ++index) {
            bytes[index] = static_cast<uint8_t>(index * 13 + index / 97);
        }
        const size_t row_bytes = raw.get_byte_size() / 8;
        const std::unordered_map<std::string, ov::Tensor> weights{{"projection.weight", raw}};
        const std::unordered_map<std::string, GgufTensorType> qtypes{{"projection.qtype", qtype}};
        const auto parts = split_fused_qkv_extracted("projection", weights, qtypes, 2, 2, 4);
        size_t first_row = 0;
        for (const auto& part : parts) {
            EXPECT_EQ(part.qtype, qtype);
            EXPECT_EQ(part.tensors.weight.get_element_type(), raw.get_element_type());
            EXPECT_FALSE(part.tensors.scales);
            EXPECT_FALSE(part.tensors.zero_point);
            EXPECT_EQ(std::memcmp(part.tensors.weight.data(), bytes + first_row * row_bytes,
                                  part.tensors.weight.get_byte_size()), 0);
            first_row += part.tensors.weight.get_shape()[0];
        }
        const auto interleaved = split_interleaved_q_gate("projection", weights, qtypes, 2);
        for (size_t part_index = 0; part_index < interleaved.size(); ++part_index) {
            const auto& part = interleaved[part_index];
            ASSERT_EQ(part.tensors.weight.get_shape(), (ov::Shape{4, kCols}));
            EXPECT_FALSE(part.tensors.scales);
            EXPECT_FALSE(part.tensors.zero_point);
            for (size_t row = 0; row < 4; ++row) {
                const auto source_row = (row / 2) * 4 + part_index * 2 + row % 2;
                EXPECT_EQ(std::memcmp(static_cast<const uint8_t*>(part.tensors.weight.data()) + row * row_bytes,
                                      bytes + source_row * row_bytes, row_bytes), 0);
            }
        }
    }
}

TEST(GGUFRawCompressedFC, RejectsMismatchedRawQuantType) {
    for (const auto qtype : {GGUF_TYPE_Q4_K, GGUF_TYPE_Q5_K, GGUF_TYPE_Q6_K, GGUF_TYPE_Q8_0}) {
        SCOPED_TRACE(qtype);
        std::unordered_map<std::string, ov::Tensor> weights;
        weights.emplace("projection.weight", ov::Tensor(gguf_raw_element_type(qtype), {kRows, kCols}));
        const auto incorrect = qtype == GGUF_TYPE_Q4_K ? GGUF_TYPE_Q5_K : GGUF_TYPE_Q4_K;
        std::unordered_map<std::string, GgufTensorType> qtypes{{"projection.qtype", incorrect}};
        GraphEmitter emitter(weights, qtypes, "test");
        EXPECT_THROW(emitter.add_weight("projection.weight"), ov::frontend::OpConversionFailure);
    }
}

// Convert a GGML_OP_NONE weight node and check the produced (constant-foldable) dequant against
// ggml's to_float reference.
TEST_P(GGUFWeight, MatchesGgmlToFloat) {
    const WeightCase c = GetParam();
    const auto qbytes = load_npy<uint8_t>(std::string(c.stem) + "_qbytes");
    const auto ref = load_npy<float>(std::string(c.stem) + "_deq");
    ASSERT_EQ(ref.size(), kRows * kCols);

    auto model = SingleOpBuilder()
                     .op("GGML_OP_NONE")
                     .output("w", ov::element::f32, {kRows, kCols})
                     .attr<ov::Tensor>("data", bytes_to_u8_tensor(qbytes))
                     .attr<std::string>("quant_type", c.quant_type)
                     .build();

    // The weight node has no graph inputs; it folds to a constant. Run it to read the values.
    auto out = run_on_cpu(model, {});
    ASSERT_EQ(out.get_size(), ref.size());

    const float* a = out.data<float>();
    float max_diff = 0.f;
    for (size_t i = 0; i < ref.size(); ++i)
        max_diff = std::max(max_diff, std::fabs(a[i] - ref[i]));
    if (c.tol == 0.f) {
        for (size_t i = 0; i < ref.size(); ++i)
            EXPECT_EQ(a[i], ref[i]) << "element " << i;
        return;
    }
    EXPECT_LE(max_diff, c.tol) << c.stem << ": frontend weight dequant diverges from ggml to_float";
}

INSTANTIATE_TEST_SUITE_P(AllQuantTypes,
                         GGUFWeight,
                         ::testing::Values(WeightCase{"q4_0", "Q4_0", kTolFaithful},
                                           WeightCase{"q4_1", "Q4_1", kTolFaithful},
                                           WeightCase{"q5_0", "Q5_0", kTolFaithful},
                                           WeightCase{"q5_1", "Q5_1", kTolFaithful},
                                           WeightCase{"q8_0", "Q8_0", kTolFaithful},
                                           WeightCase{"q8_1", "Q8_1", 0.f},
                                           WeightCase{"q2_k", "Q2_K", kTolFaithful},
                                           WeightCase{"q3_k", "Q3_K", kTolFaithful},
                                           WeightCase{"q4_k", "Q4_K", kTolU4Requant},
                                           WeightCase{"q5_k", "Q5_K", kTolRequant},
                                           WeightCase{"q6_k", "Q6_K", kTolRequant},
                                           WeightCase{"q2_0", "Q2_0", kTolFaithful},
                                           WeightCase{"q1_0", "Q1_0", kTolFaithful},
                                           WeightCase{"pq2_0", "PQ2_0", 0.f},
                                           WeightCase{"ptq1_0", "PTQ1_0", 0.f}),
                         [](const ::testing::TestParamInfo<WeightCase>& i) {
                             return std::string(i.param.stem);
                         });

TEST(GGUFWeight, Q8_1KeepsCodesAndScaleBitsForAllWeightNames) {
    const auto bytes = load_npy<uint8_t>("q8_1_qbytes");
    ASSERT_EQ(bytes.size(), kRows * kCols / 32 * 36);
    for (const std::string name : {"blk.0.attn_q.weight", "token_embd.weight", "output.weight"}) {
        SCOPED_TRACE(name);
        const auto node = make_weight_node(bytes_to_u8_tensor(bytes), "Q8_1", {kRows, kCols}, name);
        const auto model = std::make_shared<ov::Model>(ov::OutputVector{node}, ov::ParameterVector{});
        size_t payload_bytes = 0;
        size_t weight_constants = 0;
        size_t scale_constants = 0;
        for (const auto& op : model->get_ordered_ops()) {
            EXPECT_STRNE(op->get_type_name(), "Subtract");
            if (std::string(op->get_type_name()) == "Multiply") {
                EXPECT_EQ(op->get_output_element_type(0), ov::element::f32);
            }
            const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(op);
            if (!constant || constant->get_element_type() == ov::element::i64)
                continue;
            payload_bytes += constant->get_byte_size();
            if (constant->get_element_type() == ov::element::i8) {
                ++weight_constants;
                ASSERT_EQ(ov::shape_size(constant->get_shape()), kRows * kCols);
                for (size_t i = 0; i < ov::shape_size(constant->get_shape()); ++i) {
                    int8_t code;
                    std::memcpy(&code, bytes.data() + (i / 32) * 36 + 4 + i % 32, sizeof(code));
                    EXPECT_EQ(constant->get_data_ptr<int8_t>()[i], code);
                }
            } else {
                ASSERT_EQ(constant->get_element_type(), ov::element::f16);
                ++scale_constants;
                ASSERT_EQ(ov::shape_size(constant->get_shape()), kRows * kCols / 32);
                for (size_t group = 0; group < ov::shape_size(constant->get_shape()); ++group) {
                    uint16_t bits;
                    std::memcpy(&bits, bytes.data() + group * 36, sizeof(bits));
                    EXPECT_EQ(constant->get_data_ptr<ov::float16>()[group].to_bits(), bits);
                }
            }
        }
        EXPECT_EQ(weight_constants, 1);
        EXPECT_EQ(scale_constants, 1);
        EXPECT_EQ(payload_bytes, kRows * kCols / 32 * 34);
    }
}

TEST(GGUFWeight, Q8_1SuppliedDecoderPreservesHigherRankShape) {
    const auto bytes = load_npy<uint8_t>("q8_1_qbytes");
    const auto reference = load_npy<float>("q8_1_deq");
    std::vector<uint8_t> storage(bytes.size() + 1);
    std::memcpy(storage.data() + 1, bytes.data(), bytes.size());
    const ov::Tensor unaligned(ov::element::u8, {bytes.size()}, storage.data() + 1);
    const auto model = SingleOpBuilder()
                           .op("GGML_OP_NONE")
                           .output("output.weight", ov::element::f32, {2, 2, kCols})
                           .attr<ov::Tensor>("data", unaligned)
                           .attr<std::string>("quant_type", "q8_1")
                           .build();
    EXPECT_EQ(model->get_output_shape(0), (ov::Shape{2, 2, kCols}));
    const auto output = run_on_cpu(model, {});
    ASSERT_EQ(output.get_size(), reference.size());
    for (size_t i = 0; i < reference.size(); ++i)
        EXPECT_EQ(output.data<const float>()[i], reference[i]);
}

TEST(GGUFWeight, Q8_1CpuMatmulAndGatherMatchRepresentedWeights) {
    const auto bytes = load_npy<uint8_t>("q8_1_qbytes");
    const auto reference = load_npy<float>("q8_1_deq");
    ASSERT_EQ(reference.size(), kRows * kCols);
    const std::vector<int64_t> selected_rows{3, 0, 2, 1};
    ov::Tensor one_hot(ov::element::f32, {kCols, kCols});
    std::fill_n(one_hot.data<float>(), one_hot.get_size(), 0.f);
    for (size_t i = 0; i < kCols; ++i)
        one_hot.data<float>()[i * kCols + i] = 1.f;
    ov::Core core;
    for (const std::string name : {"blk.0.attn_q.weight", "token_embd.weight", "output.weight"}) {
        SCOPED_TRACE(name);
        const auto weight = make_weight_node(bytes_to_u8_tensor(bytes), "Q8_1", {kRows, kCols}, name);
        const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{kCols, kCols});
        const auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{kRows});
        const auto matmul = std::make_shared<ov::op::v0::MatMul>(input, weight, false, true);
        const auto gather =
            std::make_shared<ov::op::v8::Gather>(weight,
                                                 indices,
                                                 ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
        const auto model =
            std::make_shared<ov::Model>(ov::OutputVector{matmul, gather}, ov::ParameterVector{input, indices});
        auto request = core.compile_model(model,
                                          "CPU",
                                          ov::hint::inference_precision(ov::element::f32),
                                          ov::hint::dynamic_quantization_group_size(0))
                           .create_infer_request();
        request.set_input_tensor(0, one_hot);
        request.set_input_tensor(1, make_i64_tensor({kRows}, selected_rows));
        request.infer();
        const auto products = request.get_output_tensor(0);
        const auto embeddings = request.get_output_tensor(1);
        ASSERT_EQ(products.get_shape(), (ov::Shape{kCols, kRows}));
        ASSERT_EQ(embeddings.get_shape(), (ov::Shape{kRows, kCols}));
        for (size_t row = 0; row < kRows; ++row) {
            for (size_t col = 0; col < kCols; ++col) {
                EXPECT_EQ(products.data<const float>()[col * kRows + row], reference[row * kCols + col]);
                EXPECT_EQ(embeddings.data<const float>()[row * kCols + col],
                          reference[selected_rows[row] * kCols + col]);
            }
        }
    }
}

TEST(GGUFWeight, Q1_0KeepsBinaryCodesForAllWeightNames) {
    const auto bytes = load_npy<uint8_t>("q1_0_qbytes");
    const auto reference = load_npy<float>("q1_0_deq");
    for (const std::string name : {"blk.0.attn_q.weight", "token_embd.weight", "output.weight"}) {
        SCOPED_TRACE(name);
        const auto node = make_weight_node(bytes_to_u8_tensor(bytes), "Q1_0", {kRows, kCols}, name);
        const auto model = std::make_shared<ov::Model>(ov::OutputVector{node}, ov::ParameterVector{});
        size_t binary_bytes = 0;
        for (const auto& op : model->get_ordered_ops()) {
            if (const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(op)) {
                EXPECT_NE(constant->get_element_type(), ov::element::i4);
                EXPECT_NE(constant->get_element_type(), ov::element::i8);
                if (constant->get_element_type() == ov::element::u1)
                    binary_bytes += constant->get_byte_size();
            }
        }
        EXPECT_EQ(binary_bytes, kRows * kCols / 8);
        const auto output = run_on_cpu(model, {});
        ASSERT_EQ(output.get_size(), reference.size());
        for (size_t i = 0; i < reference.size(); ++i)
            EXPECT_FLOAT_EQ(output.data<const float>()[i], reference[i]);
    }
}

TEST(GGUFWeight, Q1_0PreservesFiniteScaleRange) {
    const std::vector<ov::float16> scales{65504.f, -65504.f, 0.f, ov::float16::from_bits(1)};
    std::vector<uint8_t> bytes(scales.size() * 18);
    for (size_t group = 0; group < scales.size(); ++group) {
        const auto bits = scales[group].to_bits();
        std::memcpy(bytes.data() + group * 18, &bits, sizeof(bits));
        std::fill_n(bytes.data() + group * 18 + 2, 16, 0x55);
    }
    const auto node = make_weight_node(bytes_to_u8_tensor(bytes), "Q1_0", {1, scales.size() * 128});
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{node}, ov::ParameterVector{});
    const auto output = run_on_cpu(model, {});
    for (size_t i = 0; i < output.get_size(); ++i) {
        const float scale = static_cast<float>(scales[i / 128]);
        EXPECT_FLOAT_EQ(output.data<const float>()[i], i % 2 == 0 ? scale : -scale);
    }
}

// token_embd / output are requantized to channel-wise Q8_0_C, and that path reads the zero-point
// as f16 -- Q2_0 used to hard-code u8 here, which threw for every ternary model.
TEST(GGUFWeightRequant, Q2_0AsTokenEmbdAndOutput) {
    const auto qbytes = load_npy<uint8_t>("q2_0_qbytes");
    const auto ref = load_npy<float>("q2_0_deq");
    ASSERT_EQ(ref.size(), kRows * kCols);

    for (const std::string name : {"token_embd.weight", "output.weight"}) {
        SCOPED_TRACE(name);
        auto model = SingleOpBuilder()
                         .op("GGML_OP_NONE")
                         .output(name, ov::element::f32, {kRows, kCols})
                         .attr<ov::Tensor>("data", bytes_to_u8_tensor(qbytes))
                         .attr<std::string>("quant_type", "Q2_0")
                         .build();

        auto out = run_on_cpu(model, {});
        ASSERT_EQ(out.get_size(), ref.size());

        const float* a = out.data<float>();
        float max_diff = 0.f;
        for (size_t i = 0; i < ref.size(); ++i)
            max_diff = std::max(max_diff, std::fabs(a[i] - ref[i]));
        EXPECT_LE(max_diff, kTolRequant) << "Q2_0 requant diverges from ggml to_float";
    }
}

// An F16 weight is wrapped directly as a constant (no dequant); round-trips the raw bytes.
TEST(GGUFWeightPlain, F16) {
    std::vector<ov::float16> vals{1.0f, -2.0f, 3.5f, -4.25f, 0.0f, 7.0f};
    ov::Tensor data(ov::element::u8, ov::Shape{vals.size() * sizeof(ov::float16)});
    std::memcpy(data.data(), vals.data(), data.get_byte_size());

    auto model = SingleOpBuilder()
                     .op("GGML_OP_NONE")
                     .output("w", ov::element::f32, {2, 3})
                     .attr<ov::Tensor>("data", data)
                     .attr<std::string>("quant_type", "F16")
                     .build();

    auto out = run_on_cpu(model, {});
    ASSERT_EQ(out.get_size(), vals.size());
    const float* a = out.data<float>();
    for (size_t i = 0; i < vals.size(); ++i)
        EXPECT_NEAR(a[i], static_cast<float>(vals[i]), 1e-6f);
}

// An F32 weight is wrapped directly as a Constant (no dequant, no Convert); round-trips exactly.
TEST(GGUFWeightPlain, F32) {
    std::vector<float> vals{1.0f, -2.0f, 3.5f, -4.25f, 0.0f, 7.0f};
    ov::Tensor data(ov::element::u8, ov::Shape{vals.size() * sizeof(float)});
    std::memcpy(data.data(), vals.data(), data.get_byte_size());

    auto model = SingleOpBuilder()
                     .op("GGML_OP_NONE")
                     .output("w", ov::element::f32, {2, 3})
                     .attr<ov::Tensor>("data", data)
                     .attr<std::string>("quant_type", "F32")
                     .build();

    auto out = run_on_cpu(model, {});
    ASSERT_EQ(out.get_size(), vals.size());
    const float* a = out.data<float>();
    for (size_t i = 0; i < vals.size(); ++i)
        EXPECT_EQ(a[i], vals[i]);
}

// A BF16 weight is wrapped as a bf16 Constant then Convert'ed to f32 for the translators.
TEST(GGUFWeightPlain, BF16) {
    std::vector<ov::bfloat16> vals{1.0f, -2.0f, 3.5f, -4.25f, 0.0f, 7.0f};
    ov::Tensor data(ov::element::u8, ov::Shape{vals.size() * sizeof(ov::bfloat16)});
    std::memcpy(data.data(), vals.data(), data.get_byte_size());

    auto model = SingleOpBuilder()
                     .op("GGML_OP_NONE")
                     .output("w", ov::element::f32, {2, 3})
                     .attr<ov::Tensor>("data", data)
                     .attr<std::string>("quant_type", "BF16")
                     .build();

    auto out = run_on_cpu(model, {});
    ASSERT_EQ(out.get_size(), vals.size());
    const float* a = out.data<float>();
    for (size_t i = 0; i < vals.size(); ++i)
        EXPECT_NEAR(a[i], static_cast<float>(vals[i]), 1e-6f);
}

// Q8_K is a ggml intermediate activation-quantization type: it only ever appears as the
// on-the-fly quantized activation in a dot product, never as a weight tensor stored in a .gguf.
// The frontend therefore does not accept it as a weight -- gguf_type_from_name knows the name
// (so a stray reference is a clear error, not a silent misparse), but make_weight_node has no
// Q8_K path and rejects it. If a future model ever stores Q8_K weights, wire it into the
// make_weight_node switch and turn this into a value test.
TEST(GGUFWeightUnsupported, Q8KIsNotAStoredWeight) {
    // One Q8_K block = 292 bytes covers 256 weights.
    ov::Tensor data(ov::element::u8, ov::Shape{292});
    std::memset(data.data(), 0, data.get_byte_size());
    EXPECT_ANY_THROW({
        SingleOpBuilder()
            .op("GGML_OP_NONE")
            .output("w", ov::element::f32, {1, 256})
            .attr<ov::Tensor>("data", data)
            .attr<std::string>("quant_type", "Q8_K")
            .build();
    });
}

// A stored MoE expert weight is rank>2 ([1,n_expert,m,k]) and stays PACKED for MUL_MAT_ID (see
// GGUFOps.MulMatIdMxfp4Packed). But the llama.cpp cgraph path (e.g. test-backend-ops' GET_ROWS /
// MUL_MAT op tests) also feeds bare 2D MXFP4 tensors that are not stored weights at all, so
// make_weight_node must dequantize those too. f4e2m1 nibble -> value mirrors kF4E2M1 in
// test_ops.cpp; scale byte 127 -> e8m0 exponent 0 -> 2^0 = 1.0.
TEST(GGUFWeight, Mxfp4TwoDim) {
    static const float kF4E2M1[16] =
        {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    // 1 row, 1 block (32 cols): byte0 = e8m0 scale, bytes1..16 = nibble-packed codes (low nibble
    // = element i, high nibble = element i+16, for i in [0,16)).
    ov::Tensor data(ov::element::u8, ov::Shape{17});
    auto* bytes = data.data<uint8_t>();
    bytes[0] = 127;
    for (size_t i = 0; i < 16; ++i)
        bytes[1 + i] = static_cast<uint8_t>((i << 4) | i);

    auto model = SingleOpBuilder()
                     .op("GGML_OP_NONE")
                     .output("w", ov::element::f32, {1, 32})
                     .attr<ov::Tensor>("data", data)
                     .attr<std::string>("quant_type", "MXFP4")
                     .build();

    auto out = run_on_cpu(model, {});
    ASSERT_EQ(out.get_size(), 32u);
    const float* a = out.data<float>();
    for (size_t j = 0; j < 32; ++j)
        EXPECT_FLOAT_EQ(a[j], kF4E2M1[j % 16]) << "element " << j;
}

// Regression for a fused-QKV bias (e.g. phi-3's blk.N.attn_qkv.bias) being silently dropped:
// register_fused_qkv split the fused weight into q/k/v parts but never touched a fused bias, so
// attention()'s add_bias calls (gated on !has_fused_qkv) never ran for these archs, and the
// bias never made it into the graph. split_fused_qkv_bias must slice the fused bias into
// exactly the q/k/v parts the weight split uses (n_q, n_k, n_v rows in that order).
TEST(GGUFFusedQkvBias, SplitsIntoExpectedRanges) {
    constexpr size_t n_q = 4, n_k = 2, n_v = 2;
    ov::Tensor bias(ov::element::f32, ov::Shape{n_q + n_k + n_v});
    auto* data = bias.data<float>();
    for (size_t i = 0; i < n_q + n_k + n_v; ++i) {
        data[i] = static_cast<float>(i);
    }
    std::unordered_map<std::string, ov::Tensor> weights{{"blk.0.attn_qkv.bias", bias}};

    auto parts = ov::frontend::gguf::split_fused_qkv_bias("blk.0.attn_qkv", weights, n_q, n_k, n_v);

    ASSERT_EQ(parts[0].get_shape(), ov::Shape{n_q});
    ASSERT_EQ(parts[1].get_shape(), ov::Shape{n_k});
    ASSERT_EQ(parts[2].get_shape(), ov::Shape{n_v});
    for (size_t i = 0; i < n_q; ++i) {
        EXPECT_FLOAT_EQ(parts[0].data<float>()[i], static_cast<float>(i));
    }
    for (size_t i = 0; i < n_k; ++i) {
        EXPECT_FLOAT_EQ(parts[1].data<float>()[i], static_cast<float>(n_q + i));
    }
    for (size_t i = 0; i < n_v; ++i) {
        EXPECT_FLOAT_EQ(parts[2].data<float>()[i], static_cast<float>(n_q + n_k + i));
    }
}

TEST(GGUFFusedQkvWeight, PreservesScalesForGroupedFormats) {
    const auto check = [](ov::frontend::gguf::GgufTensorType qtype,
                          const ov::element::Type& weight_type,
                          const ov::element::Type& scale_type,
                          size_t groups) {
        constexpr size_t rows = 6;
        constexpr size_t cols = 256;
        std::unordered_map<std::string, ov::Tensor> weights{
            {"fused.weight", ov::Tensor(weight_type, {rows, cols})},
            {"fused.scales", ov::Tensor(scale_type, {rows, groups})},
        };
        std::unordered_map<std::string, ov::frontend::gguf::GgufTensorType> qtypes{{"fused.qtype", qtype}};

        auto qkv = ov::frontend::gguf::split_fused_qkv_extracted("fused", weights, qtypes, 2, 2, 2);
        for (const auto& part : qkv) {
            ASSERT_TRUE(part.tensors.scales);
            EXPECT_EQ(part.tensors.scales.get_shape(), (ov::Shape{2, groups}));
        }

        auto q_gate = ov::frontend::gguf::split_interleaved_q_gate("fused", weights, qtypes, 1);
        for (const auto& part : q_gate) {
            ASSERT_TRUE(part.tensors.scales);
            EXPECT_EQ(part.tensors.scales.get_shape(), (ov::Shape{3, groups}));
        }
    };

    check(ov::frontend::gguf::GGUF_TYPE_MXFP4, ov::element::f4e2m1, ov::element::f8e8m0, 8);
    check(ov::frontend::gguf::GGUF_TYPE_Q1_0, ov::element::u1, ov::element::f16, 2);
    check(ov::frontend::gguf::GGUF_TYPE_Q8_1, ov::element::i8, ov::element::f16, 8);
    check(ov::frontend::gguf::GGUF_TYPE_Q8_K, ov::element::i8, ov::element::f32, 1);
}

TEST(GGUFWeight, RejectsMissingAuxiliaryTensors) {
    using namespace ov::frontend::gguf;

    WeightTensors without_scales{ov::Tensor(ov::element::i4, {1, 32}), {}, {}};
    EXPECT_THROW(make_weight_node(without_scales, GGUF_TYPE_Q4_0), ov::Exception);

    WeightTensors without_zero_point{ov::Tensor(ov::element::u32, {1, 4}), ov::Tensor(ov::element::f16, {1, 1}), {}};
    EXPECT_THROW(make_weight_node(without_zero_point, GGUF_TYPE_Q4_K), ov::Exception);
}

TEST(GGUFWeight, UsesIntegerZeroPointForRequantizedMatmulWeights) {
    using namespace ov::frontend::gguf;

    EXPECT_EQ(gguf_zero_point_type("blk.0.attn_q.weight", GGUF_TYPE_Q4_K), ov::element::u8);
    EXPECT_EQ(gguf_zero_point_type("blk.0.ffn_down.weight", GGUF_TYPE_Q4_1), ov::element::u8);
    EXPECT_EQ(gguf_zero_point_type("blk.0.ssm_out.weight", GGUF_TYPE_Q5_K), ov::element::u8);
    EXPECT_EQ(gguf_zero_point_type("token_embd.weight", GGUF_TYPE_Q4_K), ov::element::f16);
    EXPECT_EQ(gguf_zero_point_type("output.weight", GGUF_TYPE_Q5_K), ov::element::f16);
    EXPECT_EQ(gguf_zero_point_type("blk.0.attn_q.weight", GGUF_TYPE_Q2_0), ov::element::u8);
}

class GGUFQ5KNative : public ::testing::Test {
protected:
    void SetUp() override {
        path = ov::test::utils::generateTestFilePrefix() + ".gguf";
    }
    void TearDown() override {
        ov::test::utils::removeFile(path);
    }
    void write_tensor(const std::string& name, const std::vector<uint8_t>& bytes) {
        std::ofstream out(path, std::ios::binary);
        ASSERT_TRUE(out);
        const auto put = [&out](auto value) {
            out.write(reinterpret_cast<const char*>(&value), sizeof(value));
        };
        put(uint32_t{0x46554747});
        put(uint32_t{3});
        put(uint64_t{1});
        put(uint64_t{0});
        put(uint64_t{name.size()});
        out.write(name.data(), name.size());
        put(uint32_t{3});
        put(uint64_t{kCols});
        put(uint64_t{2});
        put(uint64_t{2});
        put(uint32_t{ov::frontend::gguf::GGUF_TYPE_Q5_K});
        put(uint64_t{0});
        while (static_cast<size_t>(out.tellp()) % 32 != 0)
            out.put('\0');
        out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        ASSERT_TRUE(out);
    }
    std::string path;
};

TEST_F(GGUFQ5KNative, KeepsDiskTypeAndUsesSymmetricStorageForMatrices) {
    using namespace ov::frontend::gguf;
    const auto bytes = load_npy<uint8_t>("q5_k_qbytes");
    for (const std::string name : {"blk.0.ffn_up", "token_embd", "output", "blk.0.attn_q.bias"}) {
        SCOPED_TRACE(name);
        const auto full_name = name == "blk.0.attn_q.bias" ? name : name + ".weight";
        const bool symmetric = name == "blk.0.ffn_up";
        ASSERT_NO_FATAL_FAILURE(write_tensor(full_name, bytes));
        const auto loaded = get_gguf_data(path);
        const auto& arrays = std::get<1>(loaded);
        const auto& prefix = name;
        EXPECT_EQ(std::get<2>(loaded).at(prefix + ".qtype"), GGUF_TYPE_Q5_K);
        WeightTensors tensors{arrays.at(full_name), arrays.at(prefix + ".scales"), {}};
        if (auto zp = arrays.find(prefix + ".zp"); zp != arrays.end())
            tensors.zero_point = zp->second;
        EXPECT_EQ(tensors.weight.get_element_type(), symmetric ? ov::element::i8 : ov::element::u8);
        EXPECT_EQ(tensors.scales.get_shape(), (ov::Shape{2, 2, kCols / 32}));
        EXPECT_EQ(static_cast<bool>(tensors.zero_point), !symmetric);
        EXPECT_NO_THROW(make_weight_node(tensors, GGUF_TYPE_Q5_K, full_name));
        if (!symmetric) {
            EXPECT_EQ(tensors.zero_point.get_element_type(),
                      name == "blk.0.attn_q.bias" ? ov::element::u8 : ov::element::f16);
        }
    }
}

TEST(GGUFQ5KRequant, FusedSplitsKeepSymmetricStorage) {
    using namespace ov::frontend::gguf;
    const auto bytes = load_npy<uint8_t>("q5_k_qbytes");
    GgufTensor source{};
    source.type = GGUF_TYPE_Q5_K;
    source.num_weights = kRows * kCols;
    source.bsize = bytes.size();
    source.weights_data = bytes.data();
    WeightTensors tensors{ov::Tensor(ov::element::i8, {kRows, kCols}),
                          ov::Tensor(ov::element::f16, {kRows, kCols / 32}),
                          {}};
    gguf_fill_q5_k_i8(source, tensors.weight, tensors.scales);
    const std::unordered_map<std::string, ov::Tensor> weights{{"fused.weight", tensors.weight},
                                                              {"fused.scales", tensors.scales}};
    const std::unordered_map<std::string, GgufTensorType> qtypes{{"fused.qtype", GGUF_TYPE_Q5_K}};
    const auto qkv = split_fused_qkv_extracted("fused", weights, qtypes, 2, 1, 1);
    const auto q_gate = split_interleaved_q_gate("fused", weights, qtypes, 1);
    for (const auto& part : qkv) {
        EXPECT_FALSE(part.tensors.zero_point);
        EXPECT_NO_THROW(make_weight_node(part.tensors, part.qtype));
    }
    for (size_t part = 0; part < q_gate.size(); ++part) {
        EXPECT_FALSE(q_gate[part].tensors.zero_point);
        EXPECT_NO_THROW(make_weight_node(q_gate[part].tensors, q_gate[part].qtype));
        for (size_t row = 0; row < 2; ++row) {
            const size_t source_row = row * 2 + part;
            EXPECT_EQ(std::memcmp(q_gate[part].tensors.weight.data<int8_t>() + row * kCols,
                                  tensors.weight.data<int8_t>() + source_row * kCols,
                                  kCols),
                      0);
            EXPECT_EQ(std::memcmp(q_gate[part].tensors.scales.data<ov::float16>() + row * (kCols / 32),
                                  tensors.scales.data<ov::float16>() + source_row * (kCols / 32),
                                  (kCols / 32) * sizeof(ov::float16)),
                      0);
        }
    }
}

TEST(GGUFQ5KRequant, RejectsMalformedSymmetricGraphTensors) {
    using namespace ov::frontend::gguf;
    WeightTensors tensors{ov::Tensor(ov::element::i8, {2, 256}), ov::Tensor(ov::element::f16, {2, 8}), {}};
    for (const ov::Shape& shape : {ov::Shape{1, 8}, ov::Shape{2, 2}, ov::Shape{2, 0}, ov::Shape{2, 8, 1}}) {
        tensors.scales = ov::Tensor(ov::element::f16, shape);
        EXPECT_THROW(make_weight_node(tensors, GGUF_TYPE_Q5_K), ov::Exception);
    }
    tensors.scales = ov::Tensor(ov::element::f32, {2, 8});
    EXPECT_THROW(make_weight_node(tensors, GGUF_TYPE_Q5_K), ov::Exception);
    tensors.weight = ov::Tensor(ov::element::u8, {2, 256});
    tensors.scales = ov::Tensor(ov::element::f16, {2, 8});
    EXPECT_THROW(make_weight_node(tensors, GGUF_TYPE_Q5_K), ov::Exception);
}

TEST(GGUFQ5KRequant, RejectsUnsupportedStorageLayouts) {
    using namespace ov::frontend::gguf;
    const std::unordered_map<std::string, GgufTensorType> qtypes{{"fused.qtype", GGUF_TYPE_Q5_K}};
    for (const auto& type : {ov::element::i8, ov::element::u8, ov::element::f16}) {
        for (const auto& zp_type : {ov::element::dynamic, ov::element::u8, ov::element::f16}) {
            if ((type == ov::element::i8 && zp_type == ov::element::dynamic) ||
                (type == ov::element::u8 && zp_type != ov::element::dynamic))
                continue;
            SCOPED_TRACE(type.get_type_name());
            SCOPED_TRACE(zp_type.get_type_name());
            WeightTensors tensors{ov::Tensor(type, {4, 256}), ov::Tensor(ov::element::f16, {4, 8}), {}};
            std::unordered_map<std::string, ov::Tensor> weights{{"fused.weight", tensors.weight},
                                                                {"fused.scales", tensors.scales}};
            if (zp_type != ov::element::dynamic) {
                tensors.zero_point = ov::Tensor(zp_type, {4, 8});
                weights.emplace("fused.zp", tensors.zero_point);
            }
            EXPECT_THROW(make_weight_node(tensors, GGUF_TYPE_Q5_K), ov::Exception);
            EXPECT_THROW(split_fused_qkv_extracted("fused", weights, qtypes, 2, 1, 1), ov::Exception);
            EXPECT_THROW(split_interleaved_q_gate("fused", weights, qtypes, 1), ov::Exception);
        }
    }
}
