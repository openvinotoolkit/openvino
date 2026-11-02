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

#include "common_test_utils/file_utils.hpp"
#include "node_context.hpp"
#include "op_test_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "quant/gguf.hpp"
#include "quant/weights.hpp"
#include "utils.hpp"

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
    if (c.tol == 0.f) {
        for (size_t i = 0; i < ref.size(); ++i)
            EXPECT_EQ(a[i], ref[i]) << "element " << i;
        return;
    }
    float max_diff = 0.f;
    for (size_t i = 0; i < ref.size(); ++i)
        max_diff = std::max(max_diff, std::fabs(a[i] - ref[i]));
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

TEST(GGUFWeight, TernaryRowsKeepCompressedDictionaryAndTiedProjection) {
    for (const auto& format :
         std::vector<std::pair<const char*, const char*>>{{"q2_0", "Q2_0"}, {"pq2_0", "PQ2_0"}, {"ptq1_0", "PTQ1_0"}}) {
        SCOPED_TRACE(format.second);
        const auto bytes = load_npy<uint8_t>(std::string(format.first) + "_qbytes");
        const auto weight = make_weight_node(bytes_to_u8_tensor(bytes), format.second, {kRows, kCols});
        auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 1, 1, -1});
        indices->set_friendly_name("indices");
        indices->output(0).set_names({"indices"});
        auto activation = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, kCols});
        auto projection = std::make_shared<ov::op::v0::MatMul>(activation, weight, false, true);
        auto decoder = SingleOpBuilder()
                           .op("GGML_OP_GET_ROWS")
                           .input("weight", ov::element::f32, {kRows, kCols})
                           .input("indices", ov::element::i64, {1, 1, 1, -1})
                           .output("rows", ov::element::f32, {1, 1, -1, kCols})
                           .decoder();
        auto tensors = std::make_shared<TensorMap>();
        (*tensors)["weight"] = weight;
        (*tensors)["indices"] = indices;
        NodeContext context(decoder, tensors);
        auto rows = op::translate_get_rows(context)[0];
        auto model = std::make_shared<ov::Model>(ov::OutputVector{rows}, ov::ParameterVector{indices});
        size_t dictionaries = 0;
        const void* dictionary_data = nullptr;
        for (const auto& node : model->get_ops()) {
            const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
            if (!constant || constant->get_element_type() != ov::element::u8 || constant->get_shape().size() != 3)
                continue;
            ++dictionaries;
            dictionary_data = constant->get_data_ptr();
            EXPECT_EQ(constant->get_byte_size(), kRows * kCols / 4);
        }
        EXPECT_EQ(dictionaries, 1);
        EXPECT_EQ(projection->get_input_node_shared_ptr(1), weight);
        auto original = std::make_shared<ov::Model>(ov::OutputVector{weight}, ov::ParameterVector{});
        size_t subtracts = 0;
        for (const auto& op : original->get_ordered_ops()) {
            const auto subtract = ov::as_type_ptr<ov::op::v1::Subtract>(op);
            if (!subtract)
                continue;
            ++subtracts;
            const auto convert = ov::as_type_ptr<ov::op::v0::Convert>(subtract->get_input_node_shared_ptr(1));
            const auto zp = ov::as_type_ptr<ov::op::v0::Constant>(convert ? convert->get_input_node_shared_ptr(0)
                                                                          : subtract->get_input_node_shared_ptr(1));
            ASSERT_TRUE(zp);
            EXPECT_EQ(zp->get_element_type(),
                      std::string(format.second) == "Q2_0" ? ov::element::u8 : ov::element::f16);
            EXPECT_EQ(zp->get_shape(), ov::Shape{});
            EXPECT_EQ(zp->cast_vector<float>(), std::vector<float>{1.f});
        }
        EXPECT_EQ(subtracts, 1);
        size_t original_dictionaries = 0;
        for (const auto& node : original->get_ops()) {
            const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
            if (constant && constant->get_element_type() == ov::element::u2) {
                ++original_dictionaries;
                EXPECT_EQ(dictionary_data, constant->get_data_ptr());
            }
        }
        EXPECT_EQ(original_dictionaries, 1);
        const auto reference = load_npy<float>(std::string(format.first) + "_deq");
        for (const auto& selected : std::vector<std::vector<int64_t>>{{3, 0, 3}, {2}}) {
            auto output = run_on_cpu(model, {{"indices", make_i64_tensor({1, 1, 1, selected.size()}, selected)}});
            ASSERT_EQ(output.get_shape(), (ov::Shape{1, 1, selected.size(), kCols}));
            for (size_t i = 0; i < selected.size(); ++i) {
                for (size_t j = 0; j < kCols; ++j)
                    EXPECT_FLOAT_EQ(output.data<float>()[i * kCols + j], reference[selected[i] * kCols + j]);
            }
        }
    }
}

TEST(GGUFWeight, GatherPreservesOtherWeightFormats) {
    for (const auto& format : std::vector<std::pair<const char*, const char*>>{{"q1_0", "Q1_0"},
                                                                               {"q2_k", "Q2_K"},
                                                                               {"q4_0", "Q4_0"},
                                                                               {"q8_0", "Q8_0"}}) {
        SCOPED_TRACE(format.second);
        const auto bytes = load_npy<uint8_t>(std::string(format.first) + "_qbytes");
        const auto weight = make_weight_node(bytes_to_u8_tensor(bytes), format.second, {kRows, kCols});
        auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
        const auto gather =
            ov::as_type_ptr<ov::op::v8::Gather>(gather_weight_rows(weight, indices).get_node_shared_ptr());
        ASSERT_TRUE(gather);
        EXPECT_EQ(gather->get_input_node_shared_ptr(0), weight);
    }
}

TEST(GGUFWeight, PreservesNonUniformU2ZeroPoints) {
    using namespace ov::frontend::gguf;
    WeightTensors tensors{ov::Tensor(ov::element::u2, {1, 256}),
                          ov::Tensor(ov::element::f16, {1, 2}),
                          ov::Tensor(ov::element::u8, {1, 2})};
    auto* zp_data = tensors.zero_point.data<uint8_t>();
    zp_data[0] = 1;
    zp_data[1] = 2;
    const auto node = make_weight_node(tensors, GGUF_TYPE_PTQ1_0);

    const auto model = std::make_shared<ov::Model>(ov::OutputVector{node}, ov::ParameterVector{});
    size_t subtracts = 0;
    for (const auto& op : model->get_ordered_ops()) {
        const auto subtract = ov::as_type_ptr<ov::op::v1::Subtract>(op);
        if (!subtract)
            continue;
        ++subtracts;
        const auto convert = ov::as_type_ptr<ov::op::v0::Convert>(subtract->get_input_node_shared_ptr(1));
        ASSERT_TRUE(convert);
        const auto zp = ov::as_type_ptr<ov::op::v0::Constant>(convert->get_input_node_shared_ptr(0));
        ASSERT_TRUE(zp);
        EXPECT_EQ(zp->get_shape(), (ov::Shape{1, 2, 1}));
        EXPECT_EQ(zp->cast_vector<uint8_t>(), (std::vector<uint8_t>{1, 2}));
    }
    EXPECT_EQ(subtracts, 1);
}

TEST(GGUFWeight, PreservesPQ2CodeThreeFallback) {
    for (size_t lane = 0; lane < 4; ++lane) {
        SCOPED_TRACE(lane);
        auto bytes = load_npy<uint8_t>("pq2_0_qbytes");
        bytes[2] = (bytes[2] & ~(3 << (lane * 2))) | (3 << (lane * 2));
        const auto node = make_weight_node(bytes_to_u8_tensor(bytes), "PQ2_0", {kRows, kCols});
        const auto model = std::make_shared<ov::Model>(ov::OutputVector{node}, ov::ParameterVector{});
        size_t subtracts = 0;
        for (const auto& op : model->get_ordered_ops()) {
            const auto subtract = ov::as_type_ptr<ov::op::v1::Subtract>(op);
            if (!subtract)
                continue;
            ++subtracts;
            EXPECT_EQ(subtract->get_input_shape(1), (ov::Shape{kRows, kCols / 128, 1}));
        }
        EXPECT_EQ(subtracts, 1);
        const auto output = run_on_cpu(model, {});
        uint16_t scale_bits;
        std::memcpy(&scale_bits, bytes.data(), sizeof(scale_bits));
        EXPECT_FLOAT_EQ(output.data<float>()[lane], 2.f * float(ov::float16::from_bits(scale_bits)));
    }
}

TEST(GGUFWeight, MixedU2ProjectionsKeepGroupedZeroPoints) {
    using namespace ov::frontend::gguf;
    for (const bool mixed_inputs : {false, true}) {
        for (const size_t count : {2, 3}) {
            for (const bool grouped_first : {false, true}) {
                for (const bool singleton : {false, true}) {
                    SCOPED_TRACE(::testing::Message()
                                 << count << ":" << grouped_first << ":" << singleton << ":" << mixed_inputs);
                    const size_t k = singleton ? 128 : 256;
                    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                                         ov::PartialShape{-1, static_cast<int64_t>(k)});
                    input->output(0).get_tensor().set_names({"input"});
                    ov::OutputVector outputs;
                    std::vector<std::shared_ptr<ov::op::v1::Subtract>> subtracts;
                    for (size_t i = 0; i < count; ++i) {
                        const bool fallback = i == (grouped_first ? 0 : count - 1);
                        const size_t n = singleton && fallback ? 1 : 15;
                        WeightTensors tensors{ov::Tensor(ov::element::u2, {n, k}),
                                              ov::Tensor(ov::element::f16, {n, k / 128}),
                                              ov::Tensor(ov::element::u8, {n, k / 128})};
                        std::memset(tensors.weight.data(), fallback ? 0xff : 0x55, tensors.weight.get_byte_size());
                        std::fill_n(tensors.scales.data<ov::float16>(), tensors.scales.get_size(), ov::float16(0.125f));
                        std::memset(tensors.zero_point.data(), 1, tensors.zero_point.get_byte_size());
                        auto weight = make_weight_node(tensors, GGUF_TYPE_PQ2_0);
                        auto weight_model =
                            std::make_shared<ov::Model>(ov::OutputVector{weight}, ov::ParameterVector{});
                        for (const auto& node : weight_model->get_ops()) {
                            if (auto subtract = ov::as_type_ptr<ov::op::v1::Subtract>(node))
                                subtracts.push_back(subtract);
                        }
                        auto activation = std::make_shared<ov::op::v1::Reshape>(
                            input,
                            ov::op::v0::Constant::create(ov::element::i64,
                                                         {2},
                                                         std::vector<int64_t>{-1, static_cast<int64_t>(k)}),
                            false);
                        const ov::Output<ov::Node> projection_input =
                            mixed_inputs && fallback ? input->output(0) : activation->output(0);
                        outputs.push_back(std::make_shared<ov::op::v0::MatMul>(projection_input, weight, false, true));
                    }
                    auto model = std::make_shared<ov::Model>(outputs, ov::ParameterVector{input});
                    auto reference = model->clone();
                    normalize_u2_zero_points(model);
                    for (size_t i = 0; i < count; ++i) {
                        const bool fallback = i == (grouped_first ? 0 : count - 1);
                        const auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(outputs[i].get_node_shared_ptr());
                        EXPECT_EQ(ov::is_type<ov::op::v1::Reshape>(matmul->get_input_node_shared_ptr(0)),
                                  !mixed_inputs || !fallback);
                    }
                    ASSERT_EQ(subtracts.size(), count);
                    for (const auto& subtract : subtracts) {
                        const auto convert =
                            ov::as_type_ptr<ov::op::v0::Convert>(subtract->get_input_node_shared_ptr(1));
                        ASSERT_TRUE(convert);
                        const auto zp = ov::as_type_ptr<ov::op::v0::Constant>(convert->get_input_node_shared_ptr(0));
                        ASSERT_TRUE(zp);
                        ASSERT_EQ(zp->get_shape().size(), 3);
                        EXPECT_EQ(zp->get_shape().back(), 1);
                    }
                    model->validate_nodes_and_infer_types();
                    ov::Tensor data(ov::element::f32, {1, k});
                    std::fill_n(data.data<float>(), k, 1.f);
                    ov::Core core;
                    const auto properties = ov::hint::inference_precision(ov::element::f32);
                    auto actual = core.compile_model(model, "CPU", properties).create_infer_request();
                    auto expected = core.compile_model(reference, "CPU", properties).create_infer_request();
                    actual.set_input_tensor(data);
                    expected.set_input_tensor(data);
                    actual.infer();
                    expected.infer();
                    for (size_t output = 0; output < count; ++output) {
                        const auto a = actual.get_output_tensor(output);
                        const auto b = expected.get_output_tensor(output);
                        ASSERT_EQ(a.get_size(), b.get_size());
                        for (size_t i = 0; i < a.get_size(); ++i)
                            EXPECT_FLOAT_EQ(a.data<const float>()[i], b.data<const float>()[i]);
                    }
                }
            }
        }
    }
}

TEST(GGUFWeight, HadamardProjectionGroupsPreserveDistinctActivations) {
    using namespace ov::frontend::gguf;
    for (const auto& [different_signs, different_output] :
         std::vector<std::pair<bool, bool>>{{false, false}, {true, false}, {false, true}}) {
        SCOPED_TRACE(::testing::Message() << different_signs << ":" << different_output);
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 256});
        auto split =
            std::make_shared<ov::op::v1::Split>(input, ov::op::v0::Constant::create(ov::element::i64, {}, {1}), 2);
        ov::OutputVector outputs;
        std::vector<std::shared_ptr<ov::op::v1::Subtract>> subtracts;
        for (size_t i = 0; i < 2; ++i) {
            const auto x = split->output(different_output ? i : 0);
            ov::Tensor signs(ov::element::f32, {128});
            std::fill_n(signs.data<float>(), 128, 1.f);
            signs.data<float>()[0] = different_signs && i == 1 ? -1.f : 1.f;
            const auto decoder = std::make_shared<SingleOpDecoder>(
                "GGML_OP_MUL_MAT",
                std::vector<TensorDesc>{{"x", ov::element::f32, {-1, 128}}},
                std::vector<TensorDesc>{},
                TensorDesc{"y", ov::element::f32, {-1, 128}},
                std::map<std::string, ov::Any>{{"hadamard_block", int64_t{4}}, {"hadamard_signs", signs}});
            auto tensor_map = std::make_shared<TensorMap>();
            (*tensor_map)["x"] = x;
            const auto activation = apply_hadamard(NodeContext(decoder, tensor_map), x);
            WeightTensors tensors{ov::Tensor(ov::element::u2, {16, 128}),
                                  ov::Tensor(ov::element::f16, {16, 1}),
                                  ov::Tensor(ov::element::u8, {16, 1})};
            std::memset(tensors.weight.data(), i == 0 ? 0xff : 0x99, tensors.weight.get_byte_size());
            std::fill_n(tensors.scales.data<ov::float16>(), tensors.scales.get_size(), ov::float16(0.125f));
            std::memset(tensors.zero_point.data(), 1, tensors.zero_point.get_byte_size());
            const auto weight = make_weight_node(tensors, GGUF_TYPE_PQ2_0);
            const auto weight_model = std::make_shared<ov::Model>(ov::OutputVector{weight}, ov::ParameterVector{});
            for (const auto& node : weight_model->get_ordered_ops()) {
                if (const auto subtract = ov::as_type_ptr<ov::op::v1::Subtract>(node))
                    subtracts.push_back(subtract);
            }
            outputs.push_back(std::make_shared<ov::op::v0::MatMul>(activation, weight, false, true));
        }
        auto model = std::make_shared<ov::Model>(outputs, ov::ParameterVector{input});
        const auto reference = model->clone();
        std::map<std::shared_ptr<ov::Node>, ov::OutputVector> inputs;
        for (const auto& node : model->get_ordered_ops())
            inputs[node] = node->input_values();
        normalize_u2_zero_points(model);
        ASSERT_EQ(subtracts.size(), 2);
        EXPECT_FALSE(subtracts[0]->get_input_shape(1).empty());
        EXPECT_EQ(subtracts[1]->get_input_shape(1).empty(), different_signs || different_output);
        for (const auto& [node, original_inputs] : inputs) {
            if (!ov::is_type<ov::op::v1::Subtract>(node)) {
                EXPECT_EQ(node->input_values(), original_inputs);
            }
        }
        ov::Tensor data(ov::element::f32, {2, 256});
        for (size_t i = 0; i < data.get_size(); ++i)
            data.data<float>()[i] = std::sin(static_cast<float>(i) * 0.1f);
        ov::Core core;
        const auto properties = ov::hint::inference_precision(ov::element::f32);
        auto actual = core.compile_model(model, "CPU", properties).create_infer_request();
        auto expected = core.compile_model(reference, "CPU", properties).create_infer_request();
        actual.set_input_tensor(data);
        expected.set_input_tensor(data);
        actual.infer();
        expected.infer();
        for (size_t i = 0; i < 2; ++i) {
            const auto a = actual.get_output_tensor(i);
            const auto b = expected.get_output_tensor(i);
            ASSERT_EQ(a.get_size(), b.get_size());
            for (size_t j = 0; j < a.get_size(); ++j)
                EXPECT_FLOAT_EQ(a.data<const float>()[j], b.data<const float>()[j]);
        }
    }
}

TEST(GGUFWeight, HadamardCodeThreeFallbackPreservesGroupedProjections) {
    using namespace ov::frontend::gguf;
    for (const auto type : {ov::element::f32, ov::element::f16}) {
        SCOPED_TRACE(type);
        for (const int64_t block : {256, 1024}) {
            for (const int code3_branch : {-1, 0, 1, 2}) {
                SCOPED_TRACE(::testing::Message() << block << ":" << code3_branch);
                const ov::PartialShape shape{2, 1, -1, block};
                const auto x = std::make_shared<ov::op::v0::Parameter>(type, shape);
                x->set_friendly_name("x");
                const auto cache = std::make_shared<ConstantCache>();
                ov::OutputVector outputs;
                std::vector<std::shared_ptr<ov::op::v1::Subtract>> subtracts;
                for (int branch = 0; branch < (code3_branch == 2 ? 1 : 2); ++branch) {
                    WeightTensors tensors{ov::Tensor(ov::element::u2, {16, static_cast<size_t>(block)}),
                                          ov::Tensor(ov::element::f16, {16, static_cast<size_t>(block / 128)}),
                                          ov::Tensor(ov::element::f16, {16, static_cast<size_t>(block / 128)})};
                    std::memset(tensors.weight.data(), 0x99, tensors.weight.get_byte_size());
                    if (code3_branch == branch || code3_branch == 2)
                        static_cast<uint8_t*>(tensors.weight.data())[0] = 0x9b;
                    std::fill_n(tensors.scales.data<ov::float16>(), tensors.scales.get_size(), ov::float16(0.125f));
                    std::fill_n(tensors.zero_point.data<ov::float16>(),
                                tensors.zero_point.get_size(),
                                ov::float16(1.f));
                    const auto weight = make_weight_node(tensors, GGUF_TYPE_PQ2_0);
                    const auto weight_model =
                        std::make_shared<ov::Model>(ov::OutputVector{weight}, ov::ParameterVector{});
                    for (const auto& node : weight_model->get_ops()) {
                        if (const auto subtract = ov::as_type_ptr<ov::op::v1::Subtract>(node))
                            subtracts.push_back(subtract);
                    }
                    const auto decoder = std::make_shared<SingleOpDecoder>(
                        "GGML_OP_MUL_MAT",
                        std::vector<TensorDesc>{{"w", ov::element::f32, {1, 1, 16, block}}, {"x", type, shape}},
                        std::vector<TensorDesc>{},
                        TensorDesc{"y", ov::element::f32, ov::PartialShape::dynamic()},
                        std::map<std::string, ov::Any>{{"hadamard_block", block}, {"activation_rank", 3}});
                    auto tensors_map = std::make_shared<TensorMap>(TensorMap{{"x", x}, {"w", weight}});
                    outputs.push_back(op::translate_mulmat(NodeContext(decoder, tensors_map, cache))[0]);
                }
                const auto model = std::make_shared<ov::Model>(outputs, ov::ParameterVector{x});
                const auto reference = model->clone();
                normalize_u2_zero_points(model);
                const auto nodes = model->get_ops();
                EXPECT_EQ(std::count_if(nodes.begin(),
                                        nodes.end(),
                                        [](const auto& node) {
                                            return node->get_rt_info().count("gguf_hadamard_block") != 0;
                                        }),
                          code3_branch == -1 ? 1 : 0);
                for (const auto& subtract : subtracts)
                    EXPECT_EQ(subtract->get_input_shape(1).empty(), code3_branch == -1);
                const auto normalized = model->get_ops().size();
                normalize_u2_zero_points(model);
                EXPECT_EQ(model->get_ops().size(), normalized);
                for (const size_t tokens : {1, 3}) {
                    ov::Tensor data(type, {2, 1, tokens, static_cast<size_t>(block)});
                    for (size_t i = 0; i < data.get_size(); ++i) {
                        if (type == ov::element::f16)
                            data.data<ov::float16>()[i] = ov::float16(100.f);
                        else
                            data.data<float>()[i] = std::sin(static_cast<float>(i) * 0.1f);
                    }
                    ov::TensorVector expected, actual;
                    for (size_t i = 0; i < outputs.size(); ++i) {
                        expected.emplace_back(ov::element::f32, ov::Shape{2, tokens, 16});
                        actual.emplace_back(ov::element::f32, ov::Shape{2, tokens, 16});
                    }
                    ASSERT_TRUE(reference->evaluate(expected, {data}));
                    ASSERT_TRUE(model->evaluate(actual, {data}));
                    for (size_t i = 0; i < outputs.size(); ++i) {
                        ASSERT_EQ(actual[i].get_shape(), expected[i].get_shape());
                        for (size_t j = 0; j < actual[i].get_size(); ++j)
                            EXPECT_NEAR(actual[i].data<float>()[j], expected[i].data<float>()[j], 1e-5f);
                    }
                }
            }
        }
    }
}

TEST(GGUFWeight, HadamardMixedWeightTypesKeepSeparateForms) {
    using namespace ov::frontend::gguf;
    const auto fixture = load_npy<uint8_t>("q4_0_qbytes");
    std::vector<uint8_t> bytes;
    for (size_t i = 0; i < 16; ++i)
        bytes.insert(bytes.end(), fixture.begin(), fixture.end());
    const auto q4 = make_weight_node(bytes_to_u8_tensor(bytes), "Q4_0", {16, 1024});
    WeightTensors tensors{ov::Tensor(ov::element::u2, {16, 1024}),
                          ov::Tensor(ov::element::f16, {16, 8}),
                          ov::Tensor(ov::element::f16, {16, 8})};
    std::memset(tensors.weight.data(), 0x99, tensors.weight.get_byte_size());
    std::fill_n(tensors.scales.data<ov::float16>(), tensors.scales.get_size(), ov::float16(0.125f));
    std::fill_n(tensors.zero_point.data<ov::float16>(), tensors.zero_point.get_size(), ov::float16(1.f));
    const auto u2 = make_weight_node(tensors, GGUF_TYPE_PQ2_0);
    for (const auto type : {ov::element::f32, ov::element::f16}) {
        for (const bool u2_first : {false, true}) {
            SCOPED_TRACE(::testing::Message() << type << ":" << u2_first);
            const auto x = std::make_shared<ov::op::v0::Parameter>(type, ov::PartialShape{1, -1, 1024});
            const auto cache = std::make_shared<ConstantCache>();
            ov::OutputVector outputs;
            ov::OutputVector references;
            for (const bool ternary : {u2_first, !u2_first}) {
                const auto weight = ternary ? u2 : q4;
                const auto decoder = std::make_shared<SingleOpDecoder>(
                    "GGML_OP_MUL_MAT",
                    std::vector<TensorDesc>{{"w", ov::element::f32, {1, 1, 16, 1024}}, {"x", type, {1, 1, -1, 1024}}},
                    std::vector<TensorDesc>{},
                    TensorDesc{"y", ov::element::f32, ov::PartialShape::dynamic()},
                    std::map<std::string, ov::Any>{{"hadamard_block", int64_t{1024}}});
                auto tensor_map = std::make_shared<TensorMap>(TensorMap{{"x", x}, {"w", weight}});
                const NodeContext context(decoder, tensor_map, cache);
                const auto output = op::translate_mulmat(context)[0];
                const auto branch = std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{x});
                const auto nodes = branch->get_ops();
                EXPECT_EQ(std::count_if(nodes.begin(),
                                        nodes.end(),
                                        [](const auto& node) {
                                            return node->get_rt_info().count("gguf_hadamard_block") != 0;
                                        }),
                          ternary ? 1 : 0);
                EXPECT_EQ(op::translate_mulmat(context)[0].get_node()->input_value(0),
                          output.get_node()->input_value(0));
                outputs.push_back(output);
                const auto rotation = apply_hadamard(context, x, true);
                references.push_back(
                    std::make_shared<ov::op::v0::MatMul>(context.get_f32_activation(rotation), weight, false, true));
            }
            EXPECT_EQ(cache->hadamard_outputs.size(), 2);
            const auto model = std::make_shared<ov::Model>(outputs, ov::ParameterVector{x});
            const auto reference = std::make_shared<ov::Model>(references, ov::ParameterVector{x});
            normalize_u2_zero_points(model);
            for (const size_t tokens : {1, 3}) {
                ov::Tensor data(type, {1, tokens, 1024});
                for (size_t i = 0; i < data.get_size(); ++i) {
                    if (type == ov::element::f16)
                        data.data<ov::float16>()[i] = ov::float16(100.f);
                    else
                        data.data<float>()[i] = std::sin(static_cast<float>(i) * 0.1f);
                }
                ov::TensorVector actual{ov::Tensor(ov::element::f32, {1, tokens, 16}),
                                        ov::Tensor(ov::element::f32, {1, tokens, 16})};
                ov::TensorVector expected{ov::Tensor(ov::element::f32, {1, tokens, 16}),
                                          ov::Tensor(ov::element::f32, {1, tokens, 16})};
                ASSERT_TRUE(model->evaluate(actual, {data}));
                ASSERT_TRUE(reference->evaluate(expected, {data}));
                for (size_t i = 0; i < actual.size(); ++i) {
                    for (size_t j = 0; j < actual[i].get_size(); ++j) {
                        const auto value = expected[i].data<float>()[j];
                        EXPECT_NEAR(actual[i].data<float>()[j], value, 1e-4f + 1e-5f * std::fabs(value));
                    }
                }
            }
        }
    }
}

TEST(GGUFWeight, MixedU2ProjectionsKeepFloatingZeroPointType) {
    using namespace ov::frontend::gguf;
    const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 128});
    ov::OutputVector outputs;
    for (const bool fallback : {false, true}) {
        WeightTensors tensors{ov::Tensor(ov::element::u2, {16, 128}),
                              ov::Tensor(ov::element::f16, {16, 1}),
                              ov::Tensor(ov::element::f16, {16, 1})};
        std::memset(tensors.weight.data(), fallback ? 0xff : 0x55, tensors.weight.get_byte_size());
        std::fill_n(tensors.scales.data<ov::float16>(), tensors.scales.get_size(), ov::float16(0.125f));
        std::fill_n(tensors.zero_point.data<ov::float16>(), tensors.zero_point.get_size(), ov::float16(1.f));
        outputs.push_back(
            std::make_shared<ov::op::v0::MatMul>(input, make_weight_node(tensors, GGUF_TYPE_PTQ1_0), false, true));
    }
    const auto model = std::make_shared<ov::Model>(outputs, ov::ParameterVector{input});
    normalize_u2_zero_points(model);
    size_t subtracts = 0;
    for (const auto& node : model->get_ops()) {
        if (const auto subtract = ov::as_type_ptr<ov::op::v1::Subtract>(node)) {
            ++subtracts;
            EXPECT_EQ(subtract->get_input_shape(1), (ov::Shape{16, 1, 1}));
            EXPECT_EQ(subtract->get_input_element_type(1), ov::element::f16);
        }
    }
    EXPECT_EQ(subtracts, 2);
}

TEST(GGUFWeight, TernaryProjectionsKeepScalarZeroPoints) {
    using namespace ov::frontend::gguf;
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 128});
    ov::OutputVector outputs;
    for (size_t i = 0; i < 3; ++i) {
        WeightTensors tensors{ov::Tensor(ov::element::u2, {16, 128}),
                              ov::Tensor(ov::element::f16, {16, 1}),
                              ov::Tensor(ov::element::u8, {16, 1})};
        std::memset(tensors.weight.data(), 0x55, tensors.weight.get_byte_size());
        std::fill_n(tensors.scales.data<ov::float16>(), tensors.scales.get_size(), ov::float16(0.125f));
        std::memset(tensors.zero_point.data(), 1, tensors.zero_point.get_byte_size());
        outputs.push_back(
            std::make_shared<ov::op::v0::MatMul>(input, make_weight_node(tensors, GGUF_TYPE_PTQ1_0), false, true));
    }
    auto model = std::make_shared<ov::Model>(outputs, ov::ParameterVector{input});
    normalize_u2_zero_points(model);
    size_t subtracts = 0;
    for (const auto& node : model->get_ops()) {
        if (const auto subtract = ov::as_type_ptr<ov::op::v1::Subtract>(node)) {
            ++subtracts;
            EXPECT_EQ(subtract->get_input_shape(1), ov::Shape{});
        }
    }
    EXPECT_EQ(subtracts, 3);
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
    EXPECT_EQ(gguf_zero_point_type("blk.0.attn_q.weight", GGUF_TYPE_PQ2_0), ov::element::f16);
    EXPECT_EQ(gguf_zero_point_type("blk.0.attn_q.weight", GGUF_TYPE_PTQ1_0), ov::element::f16);
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
