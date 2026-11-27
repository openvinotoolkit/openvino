// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <array>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <tuple>
#include <utility>
#include <pugixml.hpp>

#include "builder/arch_registry.hpp"
#include "builder/gguf_builder.hpp"
#include "builder/gguf_builder_decoder.hpp"
#include "builder/graph_emitter.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/graph_comparator.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "gguf_writer.hpp"
#include "input_model.hpp"
#include "op_test_utils.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/frontend/gguf/adapt_to_genai.hpp"
#include "openvino/frontend/gguf/frontend.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gated_delta_net.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/if.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/hash_util.hpp"
#include "ov_ops/fully_connected_compressed.hpp"
#include "ov_ops/gather_compressed.hpp"
#include "ov_ops/gather_matmul.hpp"
#include "quant/weights.hpp"
#include "transformations/utils/gguf_weights.hpp"

using namespace ov_gguf_test;

namespace {

#if defined(GGUF_GPU_NATIVE_TESTS) || defined(GGUF_GPU_RAW_OCL_TESTS)
class GgufScopedEnvironment {
public:
    GgufScopedEnvironment(const char* name, const char* value) : name(name) {
        if (const char* previous = std::getenv(name)) {
            had_previous = true;
            previous_value = previous;
        }
        set(value);
    }

    ~GgufScopedEnvironment() {
        set(had_previous ? previous_value.c_str() : nullptr);
    }

    GgufScopedEnvironment(const GgufScopedEnvironment&) = delete;
    GgufScopedEnvironment& operator=(const GgufScopedEnvironment&) = delete;

private:
    void set(const char* value) {
#ifdef _WIN32
        _putenv_s(name, value ? value : "");
#else
        if (value)
            ::setenv(name, value, 1);
        else
            ::unsetenv(name);
#endif
    }

    const char* name;
    std::string previous_value;
    bool had_previous = false;
};
#endif

struct GgufRawFormat {
    const char* name;
    GgufTensorType type;
};

const std::array<GgufRawFormat, 11> kRawFormats{{
    {"q4_k", GGUF_TYPE_Q4_K}, {"q5_k", GGUF_TYPE_Q5_K},
    {"q6_k", GGUF_TYPE_Q6_K}, {"q8_0", GGUF_TYPE_Q8_0},
    {"q4_0", GGUF_TYPE_Q4_0}, {"q4_1", GGUF_TYPE_Q4_1},
    {"q3_k", GGUF_TYPE_Q3_K}, {"iq2_xs", GGUF_TYPE_IQ2_XS},
    {"iq2_s", GGUF_TYPE_IQ2_S}, {"iq3_xxs", GGUF_TYPE_IQ3_XXS},
    {"iq3_s", GGUF_TYPE_IQ3_S}}};

class GGUFParser : public ::testing::Test {
protected:
    void SetUp() override {
        m_path = ov::test::utils::generateTestFilePrefix() + ".gguf";
    }

    void TearDown() override {
        ov::test::utils::removeFile(m_path);
        ov::test::utils::removeFile(m_path + ".xml");
        ov::test::utils::removeFile(m_path + ".bin");
        std::filesystem::remove(ov::util::make_path(m_path + "_\u6a21\u578b_\u017c\u00f3\u0142\u0107.gguf"));
        ov::test::utils::removeFile(m_path + ".unicode.xml");
        ov::test::utils::removeFile(m_path + ".unicode.bin");
    }

    // One tensor, no metadata. Dimensions use GGUF order (columns first).
    void write_tensor(const std::string& name,
                      GgufTensorType type,
                      uint64_t cols,
                      uint64_t rows,
                      const std::vector<uint8_t>& data = {},
                      const std::string& architecture = {},
                      const std::vector<uint64_t>& extra_dims = {},
                      uint64_t tensor_offset = 0) {
        std::ofstream out(m_path, std::ios::binary);
        ASSERT_TRUE(out);
        const auto put = [&out](auto value) {
            out.write(reinterpret_cast<const char*>(&value), sizeof(value));
        };
        put(uint32_t{0x46554747});
        put(uint32_t{3});
        put(uint64_t{1});                               // tensor count
        put(uint64_t{architecture.empty() ? 0u : 1u});  // metadata count
        if (!architecture.empty()) {
            const std::string key = "general.architecture";
            put(uint64_t{key.size()});
            out.write(key.data(), key.size());
            put(uint32_t(ov_gguf_test::GGUF_TYPE_STRING));
            put(uint64_t{architecture.size()});
            out.write(architecture.data(), architecture.size());
        }
        put(uint64_t{name.size()});
        out.write(name.data(), name.size());
        put(static_cast<uint32_t>(2 + extra_dims.size()));
        put(cols);
        put(rows);
        for (const auto dim : extra_dims)
            put(dim);
        put(static_cast<uint32_t>(type));
        put(tensor_offset);
        while (static_cast<size_t>(out.tellp()) % 32 != 0) {
            out.put('\0');
        }
        if (!data.empty()) {
            out.write(reinterpret_cast<const char*>(data.data()), data.size());
        }
        ASSERT_TRUE(out);
    }

    void serialize_ir(const std::shared_ptr<ov::Model>& model, std::ostream& xml, std::ostream& binary) {
        ov::pass::Serialize(xml, binary).run_on_model(model);
    }

    void serialize_ir(const std::shared_ptr<ov::Model>& model,
                      const std::string& xml_path,
                      const std::string& bin_path) {
        ov::pass::Serialize(xml_path, bin_path).run_on_model(model);
    }

    std::string m_path;
};

TEST_F(GGUFParser, Q8_1PreservesCodesScalesAndIgnoresSumForAllWeightNames) {
    auto bytes = load_npy<uint8_t>("q8_1_qbytes");
    const auto reference = load_npy<float>("q8_1_deq");
    ASSERT_EQ(bytes.size(), 4 * 256 / 32 * 36);
    ASSERT_EQ(reference.size(), 4 * 256);
    for (size_t group = 0; group < bytes.size() / 36; ++group) {
        const uint16_t sum_bits = 0x7c00;
        std::memcpy(bytes.data() + group * 36 + 2, &sum_bits, sizeof(sum_bits));
    }
    for (const std::string name : {"blk.0.attn_q", "token_embd", "output"}) {
        SCOPED_TRACE(name);
        ASSERT_NO_FATAL_FAILURE(write_tensor(name + ".weight", GGUF_TYPE_Q8_1, 256, 4, bytes));
        const auto loaded = get_gguf_data(m_path);
        const auto& source = std::get<1>(loaded).at(name + ".weight");
        EXPECT_EQ(source.qtype, GGUF_TYPE_Q8_1);
        const auto unpacked = unpack_gguf_weight(source);
        const auto& weights = unpacked.weight;
        const auto& scales = unpacked.scales;
        ASSERT_EQ(weights.get_element_type(), ov::element::i8);
        ASSERT_EQ(scales.get_element_type(), ov::element::f16);
        EXPECT_EQ(weights.get_shape(), (ov::Shape{4, 256}));
        EXPECT_EQ(scales.get_shape(), (ov::Shape{4, 8}));
        EXPECT_EQ(weights.get_byte_size() + scales.get_byte_size(), 4 * 256 / 32 * 34);
        EXPECT_FALSE(unpacked.zero_point);
        for (size_t group = 0; group < scales.get_size(); ++group) {
            uint16_t bits;
            std::memcpy(&bits, bytes.data() + group * 36, sizeof(bits));
            EXPECT_EQ(scales.data<const ov::float16>()[group].to_bits(), bits);
            EXPECT_EQ(std::memcmp(weights.data<const int8_t>() + group * 32, bytes.data() + group * 36 + 4, 32), 0);
        }
        const auto node = make_weight_node({weights, scales, {}}, GGUF_TYPE_Q8_1, name + ".weight");
        const auto model = std::make_shared<ov::Model>(ov::OutputVector{node}, ov::ParameterVector{});
        const auto output = run_on_cpu(model, {});
        ASSERT_EQ(output.get_size(), reference.size());
        for (size_t i = 0; i < reference.size(); ++i)
            EXPECT_EQ(output.data<const float>()[i], reference[i]);
    }
}

TEST_F(GGUFParser, QuantizedLoaderKeepsOneRawDescriptorPerTensor) {
    const auto bytes = load_npy<uint8_t>("q8_1_qbytes");
    ASSERT_NO_FATAL_FAILURE(write_tensor("output.weight", GGUF_TYPE_Q8_1, 256, 4, bytes));
    const auto loaded = get_gguf_data(m_path);
    const auto& weights = std::get<1>(loaded);
    ASSERT_EQ(weights.size(), 1);
    const auto& weight = weights.at("output.weight");
    EXPECT_EQ(weight.source_name, "output.weight");
    EXPECT_EQ(weight.qtype, GGUF_TYPE_Q8_1);
    EXPECT_EQ(weight.shape, (ov::Shape{4, 256}));
    EXPECT_TRUE(weight.native);
    ASSERT_EQ(weight.data.get_byte_size(), bytes.size());
    EXPECT_EQ(std::memcmp(weight.data.data<const uint8_t>(), bytes.data(), bytes.size()), 0);
}

TEST_F(GGUFParser, Q8_1KeepsHigherRankGroups) {
    const auto bytes = load_npy<uint8_t>("q8_1_qbytes");
    const auto reference = load_npy<float>("q8_1_deq");
    ASSERT_NO_FATAL_FAILURE(write_tensor("experts.weight", GGUF_TYPE_Q8_1, 256, 2, bytes, {}, {2}));
    const auto unpacked = unpack_gguf_weight(std::get<1>(get_gguf_data(m_path)).at("experts.weight"));
    const auto& weights = unpacked.weight;
    const auto& scales = unpacked.scales;
    EXPECT_EQ(weights.get_shape(), (ov::Shape{2, 2, 256}));
    EXPECT_EQ(scales.get_shape(), (ov::Shape{2, 2, 8}));
    const auto node = make_weight_node({weights, scales, {}}, GGUF_TYPE_Q8_1, "experts.weight");
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{node}, ov::ParameterVector{});
    const auto output = run_on_cpu(model, {});
    EXPECT_EQ(output.get_shape(), (ov::Shape{2, 2, 256}));
    ASSERT_EQ(output.get_size(), reference.size());
    for (size_t i = 0; i < reference.size(); ++i)
        EXPECT_EQ(output.data<const float>()[i], reference[i]);
}

TEST_F(GGUFParser, Q8_1ExtractedBuffersPreserveFiniteScaleRangeAfterParserRelease) {
    const std::array<uint16_t, 6> scale_bits{0, 0x8000, 0xbc00, 1, 0x7bff, 0x3555};
    std::vector<uint8_t> bytes(scale_bits.size() * 36);
    std::vector<float> reference(scale_bits.size() * 32);
    for (size_t group = 0; group < scale_bits.size(); ++group) {
        std::memcpy(bytes.data() + group * 36, &scale_bits[group], sizeof(uint16_t));
        const uint16_t sum_bits = 0x7c00;
        std::memcpy(bytes.data() + group * 36 + 2, &sum_bits, sizeof(sum_bits));
        const float scale = static_cast<float>(ov::float16::from_bits(scale_bits[group]));
        for (size_t k = 0; k < 32; ++k) {
            const int8_t code = k % 2 == 0 ? -128 : 127;
            std::memcpy(bytes.data() + group * 36 + 4 + k, &code, sizeof(code));
            reference[group * 32 + k] = static_cast<float>(code) * scale;
        }
    }
    ASSERT_NO_FATAL_FAILURE(write_tensor("output.weight", GGUF_TYPE_Q8_1, reference.size(), 1, bytes));
    WeightTensors tensors;
    {
        tensors = unpack_gguf_weight(std::get<1>(get_gguf_data(m_path)).at("output.weight"));
        ASSERT_EQ(tensors.scales.get_size(), scale_bits.size());
        for (size_t group = 0; group < scale_bits.size(); ++group)
            EXPECT_EQ(tensors.scales.data<const ov::float16>()[group].to_bits(), scale_bits[group]);
    }
    const auto node = make_weight_node(tensors, GGUF_TYPE_Q8_1, "output.weight");
    tensors = {};
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{node}, ov::ParameterVector{});
    const auto output = run_on_cpu(model, {});
    ASSERT_EQ(output.get_size(), reference.size());
    for (size_t i = 0; i < reference.size(); ++i) {
        EXPECT_TRUE(std::isfinite(output.data<const float>()[i]));
        EXPECT_EQ(output.data<const float>()[i], reference[i]);
    }
    EXPECT_NE(reference.back(), static_cast<float>(ov::float16(reference.back())));
}

TEST_F(GGUFParser, Q8_1RejectsInvalidDimensionsAndTruncatedPayload) {
    for (const uint64_t cols : {31, 33}) {
        SCOPED_TRACE(cols);
        ASSERT_NO_FATAL_FAILURE(write_tensor("invalid.weight", GGUF_TYPE_Q8_1, cols, 1, std::vector<uint8_t>(36)));
        EXPECT_THROW(get_gguf_data(m_path), ov::Exception);
    }
    for (const uint64_t rows : {1, 2}) {
        SCOPED_TRACE(rows);
        ASSERT_NO_FATAL_FAILURE(
            write_tensor("truncated.weight", GGUF_TYPE_Q8_1, 32, rows, std::vector<uint8_t>(rows * 36 - 1)));
        EXPECT_THROW(get_gguf_data(m_path), ov::Exception);
    }
}

TEST_F(GGUFParser, Q1_0PreservesBinaryStorageAndScaleValues) {
    const auto bytes = load_npy<uint8_t>("q1_0_qbytes");
    const auto reference = load_npy<float>("q1_0_deq");
    ASSERT_NO_FATAL_FAILURE(write_tensor("token_embd.weight", GGUF_TYPE_Q1_0, 256, 4, bytes));
    const auto unpacked = unpack_gguf_weight(std::get<1>(get_gguf_data(m_path)).at("token_embd.weight"));
    const auto& weights = unpacked.weight;
    const auto& scales = unpacked.scales;
    ASSERT_EQ(weights.get_element_type(), ov::element::u1);
    EXPECT_EQ(weights.get_shape(), (ov::Shape{4, 256}));
    EXPECT_EQ(weights.get_byte_size(), 128);
    EXPECT_EQ(scales.get_byte_size(), 16);
    EXPECT_FALSE(unpacked.zero_point);
    for (size_t group = 0; group < scales.get_size(); ++group) {
        uint16_t bits;
        std::memcpy(&bits, bytes.data() + group * 18, sizeof(bits));
        EXPECT_EQ(scales.data<ov::float16>()[group].to_bits(), bits);
    }
    const auto node = make_weight_node({weights, scales, {}}, GGUF_TYPE_Q1_0);
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{node}, ov::ParameterVector{});
    const auto output = run_on_cpu(model, {});
    ASSERT_EQ(output.get_size(), reference.size());
    for (size_t i = 0; i < reference.size(); ++i)
        EXPECT_FLOAT_EQ(output.data<const float>()[i], reference[i]);
}

TEST_F(GGUFParser, Q2_0KeepsIntegerZeroPointsForAllWeightNames) {
    const auto bytes = load_npy<uint8_t>("q2_0_qbytes");
    const auto reference = load_npy<float>("q2_0_deq");
    for (const std::string name : {"blk.0.attn_q", "token_embd", "output"}) {
        SCOPED_TRACE(name);
        ASSERT_NO_FATAL_FAILURE(write_tensor(name + ".weight", GGUF_TYPE_Q2_0, 256, 4, bytes));
        const auto unpacked = unpack_gguf_weight(std::get<1>(get_gguf_data(m_path)).at(name + ".weight"));
        const auto& weights = unpacked.weight;
        const auto& scales = unpacked.scales;
        const auto& zp = unpacked.zero_point;
        ASSERT_EQ(zp.get_element_type(), ov::element::u8);
        ASSERT_EQ(zp.get_shape(), (ov::Shape{4, 4}));
        for (size_t i = 0; i < zp.get_size(); ++i) {
            EXPECT_EQ(zp.data<uint8_t>()[i], 1);
        }
        const auto node = make_weight_node({weights, scales, zp}, GGUF_TYPE_Q2_0, name);
        const auto model = std::make_shared<ov::Model>(ov::OutputVector{node}, ov::ParameterVector{});
        const auto output = run_on_cpu(model, {});
        ASSERT_EQ(output.get_size(), reference.size());
        for (size_t i = 0; i < reference.size(); ++i) {
            EXPECT_FLOAT_EQ(output.data<float>()[i], reference[i]);
        }
    }
}

TEST_F(GGUFParser, PrismTernaryFormatsMatchGgmlIncludingLegacyGroup128) {
    for (const auto& [stem, file_type, expected_type] : {std::tuple{"pq2_0", GGUF_TYPE_PQ2_0, GGUF_TYPE_PQ2_0},
                                                         std::tuple{"ptq1_0", GGUF_TYPE_PTQ1_0, GGUF_TYPE_PTQ1_0},
                                                         std::tuple{"pq2_0", GGUF_TYPE_Q2_0, GGUF_TYPE_PQ2_0}}) {
        SCOPED_TRACE(stem);
        const auto bytes = load_npy<uint8_t>(std::string(stem) + "_qbytes");
        const auto reference = load_npy<float>(std::string(stem) + "_deq");
        ASSERT_NO_FATAL_FAILURE(write_tensor("token_embd.weight", file_type, 256, 4, bytes));
        const auto loaded = get_gguf_data(m_path);
        const auto& source = std::get<1>(loaded).at("token_embd.weight");
        EXPECT_EQ(source.qtype, expected_type);
        const auto unpacked = unpack_gguf_weight(source);
        EXPECT_EQ(unpacked.scales.get_shape(), (ov::Shape{4, 2}));
        const auto node = make_weight_node(unpacked, expected_type, "token_embd");
        const auto model = std::make_shared<ov::Model>(ov::OutputVector{node}, ov::ParameterVector{});
        const auto output = run_on_cpu(model, {});
        ASSERT_EQ(output.get_size(), reference.size());
        for (size_t i = 0; i < reference.size(); ++i)
            EXPECT_FLOAT_EQ(output.data<const float>()[i], reference[i]);
    }
}

TEST(GGUFHadamardMetadata, RejectsInvalidBlocksWidthsAndUnusedWeights) {
    GgufWeightTable weights{{"output.weight", ov::Tensor(ov::element::f32, {4, 16})},
                            {"token_embd.weight", ov::Tensor(ov::element::f32, {4, 12})}};
    const auto scalar = [](uint32_t value) {
        ov::Tensor tensor(ov::element::u32, {});
        *tensor.data<uint32_t>() = value;
        return tensor;
    };
    const std::string p = "prism.hadamard.";
    std::map<std::string, GGUFMetaData> config{{p + "version", scalar(1)},
                                               {p + "block_size", scalar(8)},
                                               {p + "transform", std::string("normalized-sylvester-walsh-hadamard")},
                                               {p + "axis", std::string("input-last-dimension")},
                                               {p + "sign_mode", std::string("identity")},
                                               {p + "weight_names", std::vector<std::string>{"output.weight"}}};
    for (uint32_t block : {0, 3, 8192}) {
        auto invalid = config;
        invalid[p + "block_size"] = scalar(block);
        ov::frontend::gguf::GraphEmitter emitter(weights, "qwen35");
        EXPECT_ANY_THROW(emitter.configure_hadamard(invalid));
    }
    auto inverse = config;
    inverse[p + "inverse_weight_names"] = std::vector<std::string>{"token_embd.weight"};
    ov::frontend::gguf::GraphEmitter bad_inverse(weights, "qwen35");
    EXPECT_ANY_THROW(bad_inverse.configure_hadamard(inverse));
    ov::frontend::gguf::GraphEmitter unused(weights, "qwen35");
    ASSERT_NO_THROW(unused.configure_hadamard(config));
    EXPECT_ANY_THROW(unused.validate_hadamard());
}

// Q4_K uses an integer zero-point by default and the faithful f16 one under OV_GGUF_Q4_K_ZP_F16,
// for language and projector models alike.
TEST_F(GGUFParser, Q4KZeroPointModeAppliesToLanguageAndProjectorModels) {
    const auto bytes = load_npy<uint8_t>("q4_k_qbytes");
    const auto reference = load_npy<float>("q4_k_deq");
    const auto expected_zp = gguf_zero_point_type("v.blk.0.attn_q.weight", GGUF_TYPE_Q4_K);
    // The integer zero-point requantizes each group onto a u4 grid: up to half a step of error (~0.04 here).
    const float tolerance = expected_zp == ov::element::f16 ? 3e-3f : 5e-2f;
    for (const std::string architecture : {"clip", "qwen35", "qwen35moe", "gemma4", "muse-glimmer", ""}) {
        SCOPED_TRACE(architecture);
        ASSERT_NO_FATAL_FAILURE(write_tensor("v.blk.0.attn_q.weight", GGUF_TYPE_Q4_K, 256, 4, bytes, architecture));
        const auto unpacked = unpack_gguf_weight(std::get<1>(get_gguf_data(m_path)).at("v.blk.0.attn_q.weight"));
        ASSERT_EQ(unpacked.zero_point.get_element_type(), expected_zp);
        const auto node = make_weight_node(unpacked, GGUF_TYPE_Q4_K, "v.blk.0.attn_q.weight");
        const auto model = std::make_shared<ov::Model>(ov::OutputVector{node}, ov::ParameterVector{});
        const auto output = run_on_cpu(model, {});
        ASSERT_EQ(output.get_size(), reference.size());
        for (size_t i = 0; i < reference.size(); ++i)
            EXPECT_NEAR(output.data<float>()[i], reference[i], tolerance);
    }
}

TEST_F(GGUFParser, RawFormatsPayloadAndOwnerOutliveParserResult) {
    for (const auto& format : kRawFormats) {
        SCOPED_TRACE(format.name);
        const auto bytes = load_npy<uint8_t>(std::string(format.name) + "_qbytes");
        ASSERT_NO_FATAL_FAILURE(write_tensor("blk.0.attn_q.weight", format.type, 256, 4, bytes));
        const GGUFLoadOptions unselected{format.type != GGUF_TYPE_Q4_K, format.type != GGUF_TYPE_Q5_K,
                                         format.type != GGUF_TYPE_Q6_K, format.type != GGUF_TYPE_Q8_0,
                                         format.type != GGUF_TYPE_Q4_0, format.type != GGUF_TYPE_Q4_1,
                                         format.type != GGUF_TYPE_Q3_K, format.type != GGUF_TYPE_IQ2_XS,
                                         format.type != GGUF_TYPE_IQ2_S, format.type != GGUF_TYPE_IQ3_XXS,
                                         format.type != GGUF_TYPE_IQ3_S};
        if (format.type == GGUF_TYPE_IQ2_XS || format.type == GGUF_TYPE_IQ2_S ||
            format.type == GGUF_TYPE_IQ3_XXS || format.type == GGUF_TYPE_IQ3_S) {
            const auto error = "unsupported non-quantized type " + std::to_string(format.type);
            OV_EXPECT_THROW(get_gguf_data(m_path), ov::Exception,
                            testing::HasSubstr(error));
            OV_EXPECT_THROW(get_gguf_data(m_path, GGUFLoadOptions{true}), ov::Exception,
                            testing::HasSubstr(error));
            OV_EXPECT_THROW(get_gguf_data(m_path, unselected), ov::Exception,
                            testing::HasSubstr(error));
        } else {
            const auto legacy = get_gguf_data(m_path);
            EXPECT_FALSE(unpack_gguf_weight(std::get<1>(legacy).at("blk.0.attn_q.weight"))
                             .weight.get_element_type().is_gguf_block());
            const auto original_option = get_gguf_data(m_path, GGUFLoadOptions{true});
            EXPECT_EQ(unpack_gguf_weight(std::get<1>(original_option).at("blk.0.attn_q.weight"))
                          .weight.get_element_type().is_gguf_block(),
                      format.type == GGUF_TYPE_Q4_K);
            const auto decomposed = get_gguf_data(m_path, unselected);
            EXPECT_FALSE(unpack_gguf_weight(std::get<1>(decomposed).at("blk.0.attn_q.weight"))
                             .weight.get_element_type().is_gguf_block());
        }
        const GGUFLoadOptions options{true, true, true, true, true, true, true, true, true, true, true};
        std::shared_ptr<ov::op::v0::Constant> constant;
        std::weak_ptr<ov::MappedMemory> owner;
        {
            const auto loaded = get_gguf_data(m_path, options);
            const auto& arrays = std::get<1>(loaded);
            ASSERT_EQ(arrays.size(), 1);
            const auto raw = unpack_gguf_weight(arrays.at("blk.0.attn_q.weight")).weight;
            ASSERT_EQ(raw.get_element_type(), gguf_raw_element_type(format.type));
            EXPECT_EQ(raw.get_shape(), (ov::Shape{4, 256}));
            ASSERT_EQ(raw.get_byte_size(), bytes.size());
            EXPECT_EQ(std::memcmp(raw.data(), bytes.data(), bytes.size()), 0);
            EXPECT_EQ(arrays.at("blk.0.attn_q.weight").qtype, format.type);
            owner = std::get<2>(loaded).at(0);
            constant = std::make_shared<ov::op::v0::Constant>(raw);
            EXPECT_EQ(constant->get_data_ptr(), raw.data());
        }
        EXPECT_FALSE(owner.expired());
        ASSERT_EQ(constant->get_byte_size(), bytes.size());
        EXPECT_EQ(std::memcmp(constant->get_data_ptr(), bytes.data(), bytes.size()), 0);
        constant.reset();
        EXPECT_TRUE(owner.expired());
    }
}

TEST_F(GGUFParser, Q4KRawIsOptInAndRejectsLegacyWeightGraph) {
    const auto bytes = load_npy<uint8_t>("q4_k_qbytes");
    ASSERT_NO_FATAL_FAILURE(write_tensor("blk.0.attn_q.weight", GGUF_TYPE_Q4_K, 256, 4, bytes));
    const auto legacy = get_gguf_data(m_path);
    const auto unpacked = unpack_gguf_weight(std::get<1>(legacy).at("blk.0.attn_q.weight"));
    EXPECT_EQ(unpacked.weight.get_element_type(), ov::element::u32);
    EXPECT_TRUE(unpacked.scales);
    EXPECT_TRUE(unpacked.zero_point);
    const auto loaded = get_gguf_data(std::vector<std::filesystem::path>{m_path}, GGUFLoadOptions{true});
    const auto raw = unpack_gguf_weight(std::get<1>(loaded).at("blk.0.attn_q.weight")).weight;
    OV_EXPECT_THROW(make_weight_node({raw, {}, {}}, GGUF_TYPE_Q4_K, "blk.0.attn_q"),
                    ov::Exception,
                    testing::HasSubstr("raw GGUF weights require a supported graph boundary"));
}

class GGUFParserGpu : public GGUFParser, public ::testing::WithParamInterface<GgufRawFormat> {};

TEST_F(GGUFParser, RawWeightsPublicIrRoundTrip) {
    EXPECT_EQ(ov::util::fnv1a_64("", 0), 0xcbf29ce484222325ULL);
    EXPECT_EQ(ov::util::fnv1a_64("hello", 5), 0xa430d84680aabd0bULL);
    EXPECT_EQ(ov::util::fnv1a_64("\0\xff", 2), 0x0831c907b4ea2b60ULL);
    ov::Core core;
    for (const auto& format : kRawFormats) {
        SCOPED_TRACE(format.name);
        const auto bytes = load_npy<uint8_t>(std::string(format.name) + "_qbytes");
        ov::Tensor weights(gguf_raw_element_type(format.type), {4, 256});
        ASSERT_EQ(weights.get_byte_size(), bytes.size());
        std::memcpy(weights.data(), bytes.data(), bytes.size());
        const auto constant = std::make_shared<ov::op::v0::Constant>(weights);
        const auto model = std::make_shared<ov::Model>(ov::OutputVector{constant}, ov::ParameterVector{});
        const auto check = [&](const std::shared_ptr<ov::Model>& restored) {
            const auto raw = ov::as_type_ptr<ov::op::v0::Constant>(
                restored->get_results().at(0)->get_input_node_shared_ptr(0));
            ASSERT_TRUE(raw);
            EXPECT_EQ(raw->get_element_type(), weights.get_element_type());
            EXPECT_EQ(raw->get_shape(), weights.get_shape());
            ASSERT_EQ(raw->get_byte_size(), bytes.size());
            EXPECT_EQ(std::memcmp(raw->get_data_ptr(), bytes.data(), bytes.size()), 0);
        };
        std::stringstream xml;
        std::stringstream binary;
        ASSERT_NO_THROW(ov::pass::Serialize(xml, binary).run_on_model(model));
        const auto serialized = binary.str();
        ov::Tensor constants(ov::element::u8, {serialized.size()});
        std::memcpy(constants.data(), serialized.data(), serialized.size());
        ASSERT_NO_FATAL_FAILURE(check(core.read_model(xml.str(), constants)));
        ASSERT_NO_THROW(ov::pass::Serialize(m_path + ".xml", m_path + ".bin").run_on_model(model));
        ASSERT_NO_FATAL_FAILURE(check(core.read_model(m_path + ".xml")));
        for (const bool compress_to_fp16 : {false, true}) {
            SCOPED_TRACE(compress_to_fp16);
            ASSERT_NO_THROW(ov::save_model(model, m_path + ".xml", compress_to_fp16));
            ASSERT_NO_FATAL_FAILURE(check(core.read_model(m_path + ".xml")));
        }
        std::stringstream cache;
        EXPECT_NO_THROW(ov::pass::StreamSerialize(cache, {}).run_on_model(model));
        EXPECT_FALSE(cache.str().empty());
    }
}

TEST_F(GGUFParser, NativeQ4KGatherMatchesLegacyOnCPU) {
    ov::Core core;
    const auto devices = core.get_available_devices();
    if (std::find(devices.begin(), devices.end(), "CPU") == devices.end())
        GTEST_SKIP() << "CPU plugin is unavailable";

    auto bytes = load_npy<uint8_t>("q4_k_qbytes");
    GgufWeight source;
    source.shape = {4, 256};
    source.qtype = GGUF_TYPE_Q4_K;
    source.source_name = "token_embd.weight";
    source.data = ov::Tensor(ov::element::u8, {bytes.size()}, bytes.data());
    const auto legacy = make_weight_node(unpack_gguf_weight(source), source.qtype, source.source_name);
    source.preserve_raw = true;
    const auto weights = ov::as_type_ptr<ov::op::v0::Constant>(make_weight_node(source));
    ASSERT_TRUE(weights);
    ASSERT_EQ(weights->get_byte_size(), bytes.size());

    const auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    const auto axis = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
    const auto scale = ov::op::v0::Constant::create(ov::element::f16, {}, {1});
    const auto gather = std::make_shared<ov::op::internal::GatherCompressed>(weights, indices, axis, 0, scale);
    const auto native = std::make_shared<ov::op::v0::Convert>(gather, ov::element::f32);
    const auto reference = std::make_shared<ov::op::v8::Gather>(legacy, indices, axis);
    const auto compile = [&](const ov::Output<ov::Node>& output) {
        const auto model = std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{indices});
        return core.compile_model(model, "CPU", ov::hint::inference_precision(ov::element::f32));
    };
    auto actual_request = compile(native).create_infer_request();
    auto reference_request = compile(reference).create_infer_request();
    for (std::vector<int32_t> rows : {std::vector<int32_t>{0}, {3, 1, 3, 2}}) {
        ov::Tensor input(ov::element::i32, {rows.size()}, rows.data());
        actual_request.set_input_tensor(input);
        reference_request.set_input_tensor(input);
        actual_request.infer();
        reference_request.infer();
        const auto actual = actual_request.get_output_tensor();
        const auto expected = reference_request.get_output_tensor();
        ASSERT_EQ(actual.get_shape(), (ov::Shape{rows.size(), 256}));
        ASSERT_EQ(actual.get_shape(), expected.get_shape());
        ASSERT_EQ(actual.get_element_type(), expected.get_element_type());
        for (size_t index = 0; index < actual.get_size(); ++index) {
            ASSERT_FLOAT_EQ(actual.data<const float>()[index], expected.data<const float>()[index]) << index;
        }
    }
    EXPECT_EQ(gather->get_input_node_shared_ptr(0), weights);
    EXPECT_EQ(weights->get_element_type(), ov::element::gguf_q4_k);
    EXPECT_EQ(std::memcmp(weights->get_data_ptr(), bytes.data(), bytes.size()), 0);
}

TEST_F(GGUFParser, NativeQ4KGatherExecutesAfterFileIrRoundTrip) {
    ov::Core core;
    std::vector<std::string> devices;
    for (const std::string device : {"CPU", "GPU"}) {
        try {
            RecordProperty(device, core.get_property(device, ov::device::full_name));
            devices.push_back(device);
        } catch (const ov::Exception&) {
            continue;
        }
    }
    if (devices.empty()) {
        GTEST_SKIP() << "CPU and GPU plugins are unavailable";
    }

    const auto bytes = load_npy<uint8_t>("q4_k_qbytes");
    ASSERT_NO_FATAL_FAILURE(write_tensor("token_embd.weight", GGUF_TYPE_Q4_K, 256, 4, bytes));
    const auto loaded = get_gguf_data(std::vector<std::filesystem::path>{m_path}, GGUFLoadOptions{true});
    const auto weights = ov::as_type_ptr<ov::op::v0::Constant>(
        make_weight_node(std::get<1>(loaded).at("token_embd.weight")));
    ASSERT_TRUE(weights);
    const auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    const auto axis = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
    const auto scale = ov::op::v0::Constant::create(ov::element::f16, {}, {1});
    const auto gather = std::make_shared<ov::op::internal::GatherCompressed>(weights, indices, axis, 0, scale);
    const auto output = std::make_shared<ov::op::v0::Convert>(gather, ov::element::f32);
    const auto source = std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{indices});
    serialize_ir(source, m_path + ".xml", m_path + ".bin");
    const auto restored = core.read_model(m_path + ".xml");
    const auto convert = ov::as_type_ptr<ov::op::v0::Convert>(restored->get_results()[0]->get_input_node_shared_ptr(0));
    ASSERT_TRUE(convert);
    const auto restored_gather = ov::as_type_ptr<ov::op::internal::GatherCompressed>(convert->get_input_node_shared_ptr(0));
    ASSERT_TRUE(restored_gather);
    const auto restored_weights = ov::as_type_ptr<ov::op::v0::Constant>(restored_gather->get_input_node_shared_ptr(0));
    ASSERT_TRUE(restored_weights);
    EXPECT_EQ(restored_weights->get_element_type(), ov::element::gguf_q4_k);
    EXPECT_EQ(restored_weights->get_shape(), (ov::Shape{4, 256}));
    ASSERT_EQ(restored_weights->get_byte_size(), bytes.size());
    EXPECT_EQ(std::memcmp(restored_weights->get_data_ptr(), bytes.data(), bytes.size()), 0);

    const auto precision = ov::hint::inference_precision(ov::element::f32);
    for (const auto& device : devices) {
        SCOPED_TRACE(device);
        auto direct = core.compile_model(source, device, precision).create_infer_request();
        auto from_ir = core.compile_model(restored, device, precision).create_infer_request();
        for (auto rows : {std::vector<int32_t>{0}, std::vector<int32_t>{3, 1, 3, 2}}) {
            SCOPED_TRACE(rows.size());
            ov::Tensor input(ov::element::i32, {rows.size()}, rows.data());
            direct.set_input_tensor(input);
            from_ir.set_input_tensor(input);
            direct.infer();
            from_ir.infer();
            const auto expected = direct.get_output_tensor();
            const auto actual = from_ir.get_output_tensor();
            ASSERT_EQ(actual.get_shape(), (ov::Shape{rows.size(), 256}));
            ASSERT_EQ(actual.get_shape(), expected.get_shape());
            ASSERT_EQ(actual.get_element_type(), ov::element::f32);
            ASSERT_EQ(actual.get_element_type(), expected.get_element_type());
            EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.get_byte_size()), 0);
        }
    }
    EXPECT_EQ(gather->get_input_node_shared_ptr(0), weights);
    EXPECT_EQ(restored_gather->get_input_node_shared_ptr(0), restored_weights);
    EXPECT_EQ(std::memcmp(weights->get_data_ptr(), bytes.data(), bytes.size()), 0);
    EXPECT_EQ(std::memcmp(restored_weights->get_data_ptr(), bytes.data(), bytes.size()), 0);
}

TEST_F(GGUFParser, InternalCompressedOpsPublicIrRoundTrip) {
    const auto activation = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 256});
    const auto weights = ov::op::v0::Constant::create(ov::element::u8, {1, 256}, {0});
    const auto absent = ov::op::v0::Constant::create(ov::element::f32, {0}, {});
    const auto scale = ov::op::v0::Constant::create(ov::element::f32, {1}, {1});
    const auto zero_point = ov::op::v0::Constant::create(ov::element::f32, {1}, {0});
    const auto fc = std::make_shared<ov::op::internal::FullyConnectedCompressed>(
        activation, weights, absent, scale, zero_point, ov::element::f32);
    const auto indices = ov::op::v0::Constant::create(ov::element::i32, {1}, {0});
    const auto axis = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
    const auto gather = std::make_shared<ov::op::internal::GatherCompressed>(
        weights, indices, axis, 0, scale, zero_point);
    const ov::OutputVector compressed{fc, gather};
    for (const auto& output : compressed) {
        SCOPED_TRACE(output.get_node()->get_type_name());
        const auto model = std::make_shared<ov::Model>(ov::OutputVector{output},
            output.get_node() == fc.get() ? ov::ParameterVector{activation} : ov::ParameterVector{});
        std::stringstream xml;
        std::stringstream binary;
        ASSERT_NO_THROW(ov::pass::Serialize(xml, binary).run_on_model(model));
        const auto bytes = binary.str();
        ov::Tensor constants(ov::element::u8, {bytes.size()});
        std::memcpy(constants.data(), bytes.data(), bytes.size());
        const auto restored = ov::Core().read_model(xml.str(), constants);
        const auto operation = restored->get_results().at(0)->get_input_node_shared_ptr(0);
        EXPECT_STREQ(operation->get_type_name(), output.get_node()->get_type_name());
        EXPECT_EQ(operation->get_input_size(), output.get_node()->get_input_size());
        EXPECT_EQ(restored->output().get_element_type(), output.get_element_type());
        EXPECT_EQ(restored->output().get_shape(), output.get_shape());
        std::stringstream cache;
        EXPECT_NO_THROW(ov::pass::StreamSerialize(cache).run_on_model(model));
        EXPECT_FALSE(cache.str().empty());
    }
    const auto public_model = std::make_shared<ov::Model>(ov::OutputVector{activation}, ov::ParameterVector{activation});
    std::stringstream xml;
    std::stringstream binary;
    ASSERT_NO_THROW(ov::pass::Serialize(xml, binary).run_on_model(public_model));
    EXPECT_NO_THROW(ov::Core().read_model(xml.str(), ov::Tensor{}));
    const auto public_constant = std::make_shared<ov::Model>(ov::OutputVector{scale}, ov::ParameterVector{});
    std::stringstream constant_xml;
    std::stringstream constant_binary;
    ASSERT_NO_THROW(ov::pass::Serialize(constant_xml, constant_binary).run_on_model(public_constant));
    const auto bytes = constant_binary.str();
    ov::Tensor constants(ov::element::u8, {bytes.size()});
    std::memcpy(constants.data(), bytes.data(), bytes.size());
    const auto restored = ov::Core().read_model(constant_xml.str(), constants);
    EXPECT_EQ(restored->output().get_element_type(), ov::element::f32);
    EXPECT_EQ(restored->output().get_shape(), (ov::Shape{1}));
}

TEST_F(GGUFParser, GatedDeltaNetPublicIrRoundTripWithoutExtension) {
    ov::ParameterVector parameters;
    for (const ov::Shape& shape : {ov::Shape{1, 3, 2, 4}, ov::Shape{1, 3, 2, 4}, ov::Shape{1, 3, 2, 4},
                                  ov::Shape{1, 2, 4, 4}, ov::Shape{1, 3, 2}, ov::Shape{1, 3, 2}}) {
        parameters.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape));
    }
    for (const bool fused : {false, true}) {
        SCOPED_TRACE(fused);
        const auto operation = std::make_shared<ov::op::internal::GatedDeltaNet>(
            parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5],
            fused, 1e-5f, 1e-4f);
        const auto model = std::make_shared<ov::Model>(
            ov::OutputVector{operation->output(0), operation->output(1)}, parameters);
        std::stringstream xml;
        std::stringstream binary;
        ASSERT_NO_THROW(ov::pass::Serialize(xml, binary).run_on_model(model));
        ov::Core core;
        const auto restored = core.read_model(xml.str(), ov::Tensor{});
        const auto restored_operation = ov::as_type_ptr<ov::op::internal::GatedDeltaNet>(
            restored->get_results().at(0)->get_input_node_shared_ptr(0));
        ASSERT_TRUE(restored_operation);
        EXPECT_EQ(restored_operation->get_fuse_qk_l2norm(), fused);
        EXPECT_FLOAT_EQ(restored_operation->get_q_l2_norm_eps(), 1e-5f);
        EXPECT_FLOAT_EQ(restored_operation->get_k_l2_norm_eps(), 1e-4f);
        ASSERT_EQ(restored->get_output_size(), 2);
        EXPECT_EQ(restored->get_results().at(1)->get_input_node_shared_ptr(0), restored_operation);
        for (size_t output = 0; output < model->get_output_size(); ++output) {
            EXPECT_EQ(restored->output(output).get_element_type(), model->output(output).get_element_type());
            EXPECT_EQ(restored->output(output).get_shape(), model->output(output).get_shape());
        }
        pugi::xml_document invalid;
        ASSERT_TRUE(invalid.load_string(xml.str().c_str()));
        for (auto layer : invalid.child("net").child("layers").children("layer")) {
            if (std::string(layer.attribute("type").value()) == "GatedDeltaNet")
                layer.attribute("version").set_value("ie_internal_opset");
        }
        std::stringstream invalid_xml;
        invalid.save(invalid_xml);
        EXPECT_ANY_THROW(core.read_model(invalid_xml.str(), ov::Tensor{}));
    }
}

TEST_F(GGUFParser, NestedRawWeightsPublicIrRoundTrip) {
    ov::Tensor weights(ov::element::gguf_q4_k, {1, 256});
    std::memset(weights.data(), 0, weights.get_byte_size());
    const auto raw = std::make_shared<ov::op::v0::Constant>(weights);
    const auto converted = std::make_shared<ov::op::v0::Convert>(raw, ov::element::f32);
    const auto then_body = std::make_shared<ov::Model>(ov::OutputVector{converted}, ov::ParameterVector{});
    const auto ordinary = ov::op::v0::Constant::create(ov::element::f32, {1, 256}, {0});
    const auto else_body = std::make_shared<ov::Model>(ov::OutputVector{ordinary}, ov::ParameterVector{});
    const auto condition = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, ov::Shape{});
    const auto branch = std::make_shared<ov::op::v8::If>(condition);
    branch->set_then_body(then_body);
    branch->set_else_body(else_body);
    branch->set_output(then_body->get_results().at(0), else_body->get_results().at(0));
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{branch}, ov::ParameterVector{condition});
    ASSERT_EQ(model->output().get_element_type(), ov::element::f32);
    std::stringstream xml;
    std::stringstream binary;
    ASSERT_NO_THROW(ov::pass::Serialize(xml, binary).run_on_model(model));
    const auto bytes = binary.str();
    ov::Tensor constants(ov::element::u8, {bytes.size()});
    std::memcpy(constants.data(), bytes.data(), bytes.size());
    const auto restored = ov::Core().read_model(xml.str(), constants);
    const auto restored_branch = ov::as_type_ptr<ov::op::v8::If>(
        restored->get_results().at(0)->get_input_node_shared_ptr(0));
    ASSERT_TRUE(restored_branch);
    const auto restored_convert = ov::as_type_ptr<ov::op::v0::Convert>(
        restored_branch->get_then_body()->get_results().at(0)->get_input_node_shared_ptr(0));
    ASSERT_TRUE(restored_convert);
    const auto restored_raw = ov::as_type_ptr<ov::op::v0::Constant>(
        restored_convert->get_input_node_shared_ptr(0));
    ASSERT_TRUE(restored_raw);
    EXPECT_EQ(restored_raw->get_element_type(), weights.get_element_type());
    EXPECT_EQ(restored_raw->get_shape(), weights.get_shape());
    ASSERT_EQ(restored_raw->get_byte_size(), weights.get_byte_size());
    EXPECT_EQ(std::memcmp(restored_raw->get_data_ptr(), weights.data(), weights.get_byte_size()), 0);
}

TEST_F(GGUFParser, ActivationDqPreservesFp32Input) {
#ifndef GGUF_GPU_NATIVE_TESTS
    GTEST_SKIP() << "Requires the OCL GPU plugin with oneDNN support";
#else
    ov::Core core;
    const auto devices = core.get_available_devices();
    if (std::none_of(devices.begin(), devices.end(), [](const std::string& device) {
            return device == "GPU" || device.rfind("GPU.", 0) == 0;
        })) {
        GTEST_SKIP() << "GPU device unavailable";
    }
    GgufScopedEnvironment cm("OV_GPU_GGUF_GEMM_CM", "0");
    GgufScopedEnvironment shuffle("OV_GPU_GGUF_SHUFFLE", "0");
    GgufScopedEnvironment threshold("OV_GPU_GGUF_PREFILL_THRESHOLD", "32");
    GgufScopedEnvironment prefill_f16("OV_GPU_GGUF_PREFILL_F16", "0");
    GgufScopedEnvironment dq("OV_GPU_GGUF_PREFILL_DQ_ACT", "1");
    GgufScopedEnvironment group("OV_GPU_GGUF_PREFILL_DQ_GROUP_SIZE", "0");
    GgufScopedEnvironment asymmetric("OV_GPU_GGUF_PREFILL_DQ_ASYMMETRIC", "0");
    constexpr size_t rows = 33;
    constexpr size_t columns = 512;
    constexpr size_t channels = 64;
    std::array<uint8_t, 176> block{};
    const ov::float16 scale(0.03125f);
    std::memcpy(block.data(), &scale, sizeof(scale));
    std::fill(block.begin() + 4, block.begin() + 16, 1);
    std::fill(block.begin() + 48, block.end(), 0x11);
    std::vector<uint8_t> payload(channels * columns / 256 * block.size());
    for (size_t offset = 0; offset < payload.size(); offset += block.size())
        std::memcpy(payload.data() + offset, block.data(), block.size());
    ASSERT_NO_FATAL_FAILURE(write_tensor("projection.weight", GGUF_TYPE_Q5_K, columns, channels, payload));
    auto loaded = get_gguf_data(m_path, GGUFLoadOptions{false, true});
    for (const auto& precision : {ov::element::f32, ov::element::f16}) {
        SCOPED_TRACE(precision.get_type_name());
        GraphEmitter emitter(std::get<1>(loaded), "test");
        emitter.add_input("activation", precision, {1, 1, rows, columns});
        emitter.add_weight("projection.weight");
        emitter.add_op("GGML_OP_MUL_MAT", "output", {"projection.weight", "activation"});
        emitter.graph()->model_output_names = {"output"};
        std::shared_ptr<GgufDecoder> decoder = std::make_shared<GgufBuilderDecoder>(emitter.graph());
        FrontEnd frontend;
        const auto model = frontend.convert(frontend.load(decoder));
        auto compiled = core.compile_model(model, "GPU", ov::hint::inference_precision(precision),
                           ov::num_streams(1));
        auto request = compiled.create_infer_request();
        ov::Tensor input(precision, {1, 1, rows, columns});
        if (precision == ov::element::f32)
            std::fill_n(input.data<float>(), input.get_size(), 0.03125f);
        else
            std::fill_n(input.data<ov::float16>(), input.get_size(), scale);
        request.set_input_tensor(input);
        request.infer();
        const auto output = request.get_output_tensor();
        ASSERT_EQ(output.get_size(), rows * channels);
        for (size_t index = 0; index < output.get_size(); ++index) {
            const float actual = output.get_element_type() == ov::element::f32
                                     ? output.data<const float>()[index]
                                     : static_cast<float>(output.data<const ov::float16>()[index]);
            ASSERT_NEAR(actual, 0.5f, 0.001f) << index;
        }
        if (precision == ov::element::f16) {
            for (const bool compile_asymmetric : {true, false}) {
                SCOPED_TRACE(compile_asymmetric);
                GgufScopedEnvironment compile_group("OV_GPU_GGUF_PREFILL_DQ_GROUP_SIZE", "128");
                GgufScopedEnvironment compile_asym("OV_GPU_GGUF_PREFILL_DQ_ASYMMETRIC",
                                                  compile_asymmetric ? "1" : "0");
                GgufScopedEnvironment compile_requant("OV_GPU_GGUF_REQUANT_GROUP", "64");
                auto cached = core.compile_model(model, "GPU", ov::hint::inference_precision(precision),
                                                 ov::num_streams(1));
                std::stringstream blob;
                cached.export_model(blob);
                GgufScopedEnvironment import_group("OV_GPU_GGUF_PREFILL_DQ_GROUP_SIZE", "512");
                GgufScopedEnvironment import_asym("OV_GPU_GGUF_PREFILL_DQ_ASYMMETRIC",
                                                 compile_asymmetric ? "0" : "1");
                GgufScopedEnvironment import_requant("OV_GPU_GGUF_REQUANT_GROUP", "32");
                ov::Core import_core;
                auto imported = import_core.import_model(blob, "GPU", ov::hint::inference_precision(precision),
                                                          ov::num_streams(1));
                for (auto* compiled_model : {&cached, &imported}) {
                    auto restored_request = compiled_model->create_infer_request();
                    for (size_t index = 0; index < input.get_size(); ++index)
                        input.data<ov::float16>()[index] = ov::float16(0.015625f *
                            static_cast<float>(1 << ((index % columns) / 128)));
                    restored_request.set_input_tensor(input);
                    restored_request.infer();
                    const auto restored_output = restored_request.get_output_tensor();
                    ASSERT_EQ(restored_output.get_size(), rows * channels);
                    for (size_t index = 0; index < restored_output.get_size(); ++index) {
                        const float actual = restored_output.get_element_type() == ov::element::f32
                            ? restored_output.data<const float>()[index]
                            : static_cast<float>(restored_output.data<const ov::float16>()[index]);
                        ASSERT_NEAR(actual, 0.9375f, 0.001f) << index;
                    }
                }
            }
        }
    }
#endif
}

TEST_F(GGUFParser, AsymmetricWeightTranscodePreservesSingleSignGroups) {
#ifndef GGUF_GPU_NATIVE_TESTS
    GTEST_SKIP() << "Requires the OCL GPU plugin with oneDNN support";
#else
    ov::Core core;
    const auto devices = core.get_available_devices();
    if (std::none_of(devices.begin(), devices.end(), [](const std::string& device) {
            return device == "GPU" || device.rfind("GPU.", 0) == 0;
        })) {
        GTEST_SKIP() << "GPU device unavailable";
    }
    GgufScopedEnvironment cm("OV_GPU_GGUF_GEMM_CM", "0");
    GgufScopedEnvironment shuffle("OV_GPU_GGUF_SHUFFLE", "0");
    GgufScopedEnvironment threshold("OV_GPU_GGUF_PREFILL_THRESHOLD", "32");
    GgufScopedEnvironment prefill_f16("OV_GPU_GGUF_PREFILL_F16", "0");
    GgufScopedEnvironment group("OV_GPU_GGUF_PREFILL_DQ_GROUP_SIZE", "0");
    GgufScopedEnvironment asymmetric("OV_GPU_GGUF_PREFILL_DQ_ASYMMETRIC", "0");
    GgufScopedEnvironment requant_group("OV_GPU_GGUF_REQUANT_GROUP", "32");
    constexpr size_t rows = 33;
    constexpr size_t columns = 512;
    constexpr size_t channels = 64;
    const std::array<uint8_t, 6> codes{0x99, 0x77, 0xb9, 0x75, 0x88, 0xa7};
    const std::array<float, 6> expected{0.5f, -0.5f, 1.f, -1.f, 0.f, 0.25f};
    const ov::float16 scale(0.03125f);
    std::vector<uint8_t> payload(channels * columns / 32 * 18);
    for (size_t channel = 0; channel < channels; ++channel) {
        for (size_t block = 0; block < columns / 32; ++block) {
            auto* destination = payload.data() + (channel * columns / 32 + block) * 18;
            std::memcpy(destination, &scale, sizeof(scale));
            std::fill_n(destination + 2, 16, codes[channel % codes.size()]);
        }
    }
    ASSERT_NO_FATAL_FAILURE(write_tensor("projection.weight", GGUF_TYPE_Q4_0, columns, channels, payload));
    auto loaded = get_gguf_data(m_path, GGUFLoadOptions{false, false, false, false, true});
    GraphEmitter emitter(std::get<1>(loaded), "test");
    emitter.add_input("activation", ov::element::f16, {1, 1, rows, columns});
    emitter.add_weight("projection.weight");
    emitter.add_op("GGML_OP_MUL_MAT", "output", {"projection.weight", "activation"});
    emitter.graph()->model_output_names = {"output"};
    std::shared_ptr<GgufDecoder> decoder = std::make_shared<GgufBuilderDecoder>(emitter.graph());
    FrontEnd frontend;
    const auto model = frontend.convert(frontend.load(decoder));
    for (const char* dq_value : {"0", "1"}) {
        SCOPED_TRACE(dq_value);
        GgufScopedEnvironment dq("OV_GPU_GGUF_PREFILL_DQ_ACT", dq_value);
        auto compiled = core.compile_model(model, "GPU", ov::hint::inference_precision(ov::element::f16),
                                           ov::num_streams(1));
        auto request = compiled.create_infer_request();
        ov::Tensor input(ov::element::f16, {1, 1, rows, columns});
        std::fill_n(input.data<ov::float16>(), input.get_size(), scale);
        request.set_input_tensor(input);
        request.infer();
        const auto output = request.get_output_tensor();
        ASSERT_TRUE(output.get_element_type() == ov::element::f16 ||
                output.get_element_type() == ov::element::f32);
        ASSERT_EQ(output.get_size(), rows * channels);
        std::array<float, 6> max_errors{};
        for (size_t index = 0; index < output.get_size(); ++index) {
            const auto pattern = (index % channels) % codes.size();
            const float actual = output.get_element_type() == ov::element::f32
                         ? output.data<const float>()[index]
                         : static_cast<float>(output.data<const ov::float16>()[index]);
            ASSERT_TRUE(std::isfinite(actual)) << index;
            max_errors[pattern] = std::max(max_errors[pattern], std::abs(actual - expected[pattern]));
        }
        for (size_t pattern = 0; pattern < codes.size(); ++pattern)
            EXPECT_LE(max_errors[pattern], 0.001f) << "pattern=" << pattern;
    }
#endif
}

TEST_F(GGUFParser, CmFusedSwishPreservesBeta) {
#ifndef GGUF_GPU_NATIVE_TESTS
    GTEST_SKIP() << "Requires the OCL GPU plugin with oneDNN support";
#else
    ov::Core core;
    const auto devices = core.get_available_devices();
    if (std::none_of(devices.begin(), devices.end(), [](const std::string& device) {
            return device == "GPU" || device.rfind("GPU.", 0) == 0;
        })) {
        GTEST_SKIP() << "GPU device unavailable";
    }
    GgufScopedEnvironment cm("OV_GPU_GGUF_PREFILL_GEMM", "cm");
    GgufScopedEnvironment shuffle("OV_GPU_GGUF_SHUFFLE", "1");
    GgufScopedEnvironment threshold("OV_GPU_GGUF_PREFILL_THRESHOLD", "32");
    constexpr size_t rows = 33;
    constexpr size_t columns = 512;
    constexpr size_t channels = 64;
    const ov::float16 scale(0.03125f);
    ov::Tensor weights(ov::element::gguf_q4_0, {channels, columns});
    for (size_t offset = 0; offset < weights.get_byte_size(); offset += 18) {
        auto* block = static_cast<uint8_t*>(weights.data()) + offset;
        std::memcpy(block, &scale, sizeof(scale));
        std::fill_n(block + 2, 16, 0x99);
    }
    const auto weight = std::make_shared<ov::op::v0::Constant>(weights);
    const auto absent = ov::op::v0::Constant::create(ov::element::f32, {0}, {});
    for (const float beta : {0.f, 1.f, 2.f, -1.f}) {
        SCOPED_TRACE(beta);
        const auto activation = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                                      ov::Shape{rows, columns});
        const auto fc = std::make_shared<ov::op::internal::FullyConnectedCompressed>(
            activation, weight, absent, absent, absent, ov::element::f16);
        const auto swish = std::make_shared<ov::op::v4::Swish>(
            fc, ov::op::v0::Constant::create(ov::element::f16, {}, {beta}));
        const auto model = std::make_shared<ov::Model>(ov::OutputVector{swish}, ov::ParameterVector{activation});
        auto compiled = core.compile_model(model, "GPU", ov::hint::inference_precision(ov::element::f16),
                                           ov::num_streams(1), ov::enable_profiling(true));
        auto request = compiled.create_infer_request();
        ov::Tensor input(ov::element::f16, {rows, columns});
        std::fill_n(input.data<ov::float16>(), input.get_size(), scale);
        request.set_input_tensor(input);
        request.infer();
        const auto runtime_operations = compiled.get_runtime_model()->get_ordered_ops();
        const bool used_cm = std::any_of(runtime_operations.begin(), runtime_operations.end(), [](const auto& operation) {
            const auto& info = operation->get_rt_info();
            const auto kernels = info.find("gguf_last_executed_kernels");
            return kernels != info.end() && kernels->second.template as<std::string>().find("gemm_q40_full") != std::string::npos;
        });
        ASSERT_TRUE(used_cm) << "Requires CM GGUF FC execution";
        const auto output = request.get_output_tensor();
        ASSERT_EQ(output.get_element_type(), ov::element::f16);
        ASSERT_EQ(output.get_size(), rows * channels);
        const float expected = 0.5f / (1.f + std::exp(-beta * 0.5f));
        for (size_t index = 0; index < output.get_size(); ++index)
            EXPECT_NEAR(static_cast<float>(output.data<const ov::float16>()[index]), expected, 0.001f) << index;
    }
#endif
}

TEST_P(GGUFParserGpu, RawFileToGpuPreservesOwnerAndDynamicResults) {
#ifndef GGUF_GPU_NATIVE_TESTS
    GTEST_SKIP() << "Requires the OCL GPU plugin with oneDNN support";
#else
    ov::Core core;
    const auto devices = core.get_available_devices();
    if (std::none_of(devices.begin(), devices.end(), [](const std::string& device) {
            return device == "GPU" || device.rfind("GPU.", 0) == 0;
        })) {
        GTEST_SKIP() << "GPU device unavailable";
    }
    const auto format = GetParam();
    const bool has_gather = format.type != GGUF_TYPE_Q3_K && format.type != GGUF_TYPE_IQ2_XS &&
                            format.type != GGUF_TYPE_IQ2_S && format.type != GGUF_TYPE_IQ3_XXS &&
                            format.type != GGUF_TYPE_IQ3_S;
    const auto cache_directory_value = std::getenv("OV_GPU_GGUF_TEST_CACHE_DIR");
    const std::string cache_directory = cache_directory_value ? cache_directory_value : "";
    const auto cache_hit_value = std::getenv("OV_GPU_GGUF_TEST_CACHE_HIT");
    const std::string expected_cache_hit = cache_hit_value ? cache_hit_value : "";
    const auto weight_delta_value = std::getenv("OV_GPU_GGUF_TEST_WEIGHT_DELTA");
    const std::string weight_delta = weight_delta_value ? weight_delta_value : "";
    ASSERT_EQ(cache_directory.empty(), expected_cache_hit.empty());
    ASSERT_TRUE(weight_delta.empty() || (!cache_directory.empty() && (weight_delta == "0" || weight_delta == "1")));
    if (!cache_directory.empty()) {
        ASSERT_TRUE(std::filesystem::is_directory(cache_directory));
        ASSERT_TRUE(expected_cache_hit == "0" || expected_cache_hit == "1");
        const auto source_directory = std::filesystem::path(cache_directory) / "sources";
        std::filesystem::create_directories(source_directory);
        m_path = (source_directory / (std::string(format.name) + ".gguf")).string();
    }
    const auto qbytes = load_npy<uint8_t>(std::string(format.name) + "_qbytes");
    const auto decoded = load_npy<float>(std::string(format.name) + "_deq");
    ASSERT_EQ(decoded.size(), 4 * 256);
    constexpr size_t channels = 64;
    constexpr size_t columns = 512;
    const auto row_bytes = qbytes.size() / 4;
    const size_t source_row_offset = weight_delta == "1" ? 1 : 0;
    RecordProperty("file_source_row_offset", std::to_string(source_row_offset));
    if (source_row_offset)
        ASSERT_NE(std::memcmp(qbytes.data(), qbytes.data() + row_bytes, row_bytes), 0);
    std::vector<uint8_t> payload(channels * 2 * row_bytes);
    for (size_t channel = 0; channel < channels; ++channel) {
        for (size_t block = 0; block < 2; ++block) {
            std::memcpy(payload.data() + (channel * 2 + block) * row_bytes,
                        qbytes.data() + ((channel + source_row_offset) % 4) * row_bytes,
                        row_bytes);
        }
    }
    ASSERT_NO_FATAL_FAILURE(write_tensor("projection.weight", format.type, columns, channels, payload));
    std::shared_ptr<ov::Model> model;
    std::weak_ptr<ov::MappedMemory> owner;
    {
        const GGUFLoadOptions options{format.type == GGUF_TYPE_Q4_K, format.type == GGUF_TYPE_Q5_K,
                          format.type == GGUF_TYPE_Q6_K, format.type == GGUF_TYPE_Q8_0,
                          format.type == GGUF_TYPE_Q4_0, format.type == GGUF_TYPE_Q4_1,
                          format.type == GGUF_TYPE_Q3_K, format.type == GGUF_TYPE_IQ2_XS,
                          format.type == GGUF_TYPE_IQ2_S, format.type == GGUF_TYPE_IQ3_XXS,
                          format.type == GGUF_TYPE_IQ3_S};
        if (format.type == GGUF_TYPE_Q4_K) {
            const auto unicode_path = ov::util::make_path(m_path + "_\u6a21\u578b_\u017c\u00f3\u0142\u0107.gguf");
            ASSERT_TRUE(std::filesystem::copy_file(m_path, unicode_path));
            auto unicode_data = get_gguf_data(unicode_path, options);
            const auto unicode_raw = make_weight_node(std::get<1>(unicode_data).at("projection.weight"));
            EXPECT_EQ(unicode_raw->get_rt_info().at("gguf_ext_source").as<std::string>(),
                      ov::util::path_to_string(unicode_path.filename()));
            auto unicode_model = std::make_shared<ov::Model>(
                ov::OutputVector{unicode_raw, ov::op::v0::Constant::create(ov::element::f32, {1}, {1.f})},
                ov::ParameterVector{});
            serialize_ir(unicode_model, m_path + ".unicode.xml", m_path + ".unicode.bin");
            const auto restored = core.read_model(m_path + ".unicode.xml");
            const auto restored_raw = ov::as_type_ptr<ov::op::v0::Constant>(
                restored->get_results().at(0)->get_input_node_shared_ptr(0));
            ASSERT_TRUE(restored_raw);
            EXPECT_EQ(std::memcmp(restored_raw->get_data_ptr(), payload.data(), payload.size()), 0);
            std::stringstream unicode_xml, unicode_binary;
            serialize_ir(restored, unicode_xml, unicode_binary);
            EXPECT_LT(unicode_binary.str().size(), payload.size());
            RecordProperty("unicode_public_ir_reference_roundtrip", "1");
        }
        auto loaded = get_gguf_data(m_path, options);
        auto& weights = std::get<1>(loaded);
        const auto raw = unpack_gguf_weight(weights.at("projection.weight")).weight;
        ASSERT_EQ(raw.get_element_type(), gguf_raw_element_type(format.type));
        ASSERT_EQ(raw.get_shape(), (ov::Shape{channels, columns}));
        ASSERT_EQ(raw.get_byte_size(), payload.size());
        ASSERT_EQ(std::memcmp(raw.data(), payload.data(), payload.size()), 0);
        owner = std::get<2>(loaded).at(0);
        GraphEmitter emitter(weights, "test");
        emitter.add_input("activation", ov::element::f32, {1, 1, -1, columns});
        emitter.add_weight("projection.weight");
        emitter.add_op("GGML_OP_MUL_MAT", "output", {"projection.weight", "activation"});
        emitter.graph()->model_output_names = {"output"};
        if (has_gather) {
            emitter.add_input("indices", ov::element::i64, {1, 1, 1, -1});
            emitter.add_op("GGML_OP_GET_ROWS", "embedding", {"projection.weight", "indices"});
            emitter.graph()->model_output_names.push_back("embedding");
        }
        std::shared_ptr<GgufDecoder> decoder = std::make_shared<GgufBuilderDecoder>(emitter.graph());
        FrontEnd frontend;
        model = frontend.convert(frontend.load(decoder));
        const auto fc = ov::as_type_ptr<ov::op::internal::FullyConnectedCompressed>(
            model->get_results().at(0)->get_input_node_shared_ptr(0));
        ASSERT_TRUE(fc);
        const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(fc->get_input_node_shared_ptr(1));
        ASSERT_TRUE(constant);
        EXPECT_EQ(constant->get_data_ptr(), raw.data());
        EXPECT_EQ(std::memcmp(constant->get_data_ptr(), payload.data(), payload.size()), 0);
        ASSERT_TRUE(constant->get_rt_info().count("gguf_ext_source"));
        EXPECT_EQ(constant->get_rt_info().at("gguf_ext_source").as<std::string>(),
                  std::filesystem::path(m_path).filename().string());
        emitter.add_weight_from("alias.weight", "projection");
        emitter.add_op("GGML_OP_MUL_MAT", "alias_output", {"alias.weight", "activation"});
        emitter.graph()->model_output_names = {"alias_output"};
        std::shared_ptr<GgufDecoder> alias_decoder = std::make_shared<GgufBuilderDecoder>(emitter.graph());
        const auto alias_model = frontend.convert(frontend.load(alias_decoder));
        const auto alias_fc = alias_model->get_results().at(0)->get_input_node_shared_ptr(0);
        const auto alias = ov::as_type_ptr<ov::op::v0::Constant>(alias_fc->get_input_node_shared_ptr(1));
        ASSERT_TRUE(alias);
        EXPECT_EQ(alias->get_rt_info().at("gguf_ext_offset").as<uint64_t>(),
              constant->get_rt_info().at("gguf_ext_offset").as<uint64_t>());
        ov::Tensor copied(raw.get_element_type(), raw.get_shape());
        std::memcpy(copied.data(), raw.data(), raw.get_byte_size());
        const auto local_weight = [&](const ov::Tensor& data) {
            GgufWeight source;
            source.data = data;
            source.shape = data.get_shape();
            source.qtype = format.type;
            source.source_name = "projection.weight";
            source.preserve_raw = true;
            return make_weight_node(source);
        };
        const auto copy = local_weight(copied);
        EXPECT_EQ(copy->get_rt_info().count("gguf_ext_source"), 0);
        ov::Tensor sliced(raw.get_element_type(), {channels / 2, columns});
        std::memcpy(sliced.data(), raw.data(), sliced.get_byte_size());
        const auto slice = local_weight(sliced);
        EXPECT_EQ(slice->get_rt_info().count("gguf_ext_source"), 0);
        RecordProperty("automatic_external_source", "1");
        RecordProperty("copied_sliced_sources_absent", "1");
    }
    EXPECT_FALSE(owner.expired());
    std::set<std::string> executed_stages;
    bool cache_roundtrip = false;
    bool external_reference = false;
    bool compiled_import = false;
    int stream_count = 1;
    const auto run = [&](const char* threshold_value, const char* shuffle_value,
                         ov::element::Type inference_type = ov::element::f32, const char* binary_mode = nullptr,
                         size_t static_rows = 0, bool rebind_broadcast = false) {
        SCOPED_TRACE(threshold_value ? "native" : "default_f16_transcode");
        SCOPED_TRACE(shuffle_value);
        GgufScopedEnvironment threshold("OV_GPU_GGUF_PREFILL_THRESHOLD", threshold_value);
        GgufScopedEnvironment prefill_f16("OV_GPU_GGUF_PREFILL_F16", nullptr);
        GgufScopedEnvironment shuffle("OV_GPU_GGUF_SHUFFLE", shuffle_value);
        SCOPED_TRACE(inference_type.get_type_name());
        auto run_model = model->clone();
        if (binary_mode) {
            const auto fc_output = run_model->get_results().at(0)->input_value(0);
            const auto peer = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, fc_output.get_partial_shape());
            ov::Output<ov::Node> binary;
            if (std::string(binary_mode) == "sum")
                binary = std::make_shared<ov::op::v1::Add>(fc_output, peer);
            else
                binary = std::make_shared<ov::op::v1::Multiply>(fc_output, peer);
            binary.get_node_shared_ptr()->set_friendly_name("file_binary_post");
            run_model->get_results().at(0)->input(0).replace_source_output(std::make_shared<ov::op::v0::Relu>(binary));
            run_model->add_parameters({peer});
            run_model->validate_nodes_and_infer_types();
        }
        if (static_rows) {
            run_model->get_parameters().at(0)->set_partial_shape(ov::Shape{1, 1, static_rows, columns});
            run_model->get_parameters().at(1)->set_partial_shape(ov::Shape{1, 1, 1, static_rows});
            run_model->get_parameters().at(2)->set_partial_shape(ov::Shape{1, 1, static_rows, channels});
            run_model->validate_nodes_and_infer_types();
            ASSERT_FALSE(run_model->is_dynamic());
        }
        const bool cm_requested = std::getenv("OV_GPU_GGUF_PREFILL_GEMM") &&
                                  std::string(std::getenv("OV_GPU_GGUF_PREFILL_GEMM")) == "cm";
        GgufScopedEnvironment cm("OV_GPU_GGUF_PREFILL_GEMM", cm_requested ? "cm" : "ocl");
        const bool shuffled = std::string(shuffle_value) == "1";
        const std::string scenario = std::string(binary_mode ? binary_mode : "plain") + "_" +
            inference_type.get_type_name() + "_s" + shuffle_value + (threshold_value ? "_native" : "_default") +
            (static_rows ? "_static" : "_dynamic") + (rebind_broadcast ? "_broadcast" : "") +
            (cache_roundtrip ? "_cache" : "") + (external_reference ? "_external" : "") +
            (compiled_import ? "_import" : "") + (stream_count == 2 ? "_streams2" : "");
        SCOPED_TRACE(scenario);
        const bool check_cache = !cache_directory.empty() && !binary_mode && !cache_roundtrip;
        std::string model_cache_directory;
        if (check_cache) {
            model_cache_directory = (std::filesystem::path(cache_directory) / format.name / scenario).string();
            std::filesystem::create_directories(model_cache_directory);
            run_model->set_friendly_name(std::string("gguf_file_cache_") + format.name);
        }
        auto compiled = core.compile_model(run_model, "GPU", ov::hint::inference_precision(inference_type),
                           ov::enable_profiling(true), ov::num_streams(stream_count),
                           ov::cache_dir(model_cache_directory));
        if (check_cache) {
            const bool cache_hit = compiled.get_property(ov::loaded_from_cache);
            EXPECT_EQ(cache_hit, expected_cache_hit == "1");
            RecordProperty(scenario + "_cache_hit", cache_hit ? "1" : "0");
        }
        const int compiled_streams = compiled.get_property(ov::num_streams);
        EXPECT_EQ(compiled_streams, stream_count);
        RecordProperty(scenario + "_compiled_streams", std::to_string(compiled_streams));
        if (compiled_import) {
            std::stringstream blob;
            compiled.export_model(blob);
            RecordProperty(scenario + "_compiled_blob_bytes", std::to_string(blob.str().size()));
            compiled = ov::CompiledModel{};
            ov::Core import_core;
            GgufScopedEnvironment import_threshold("OV_GPU_GGUF_PREFILL_THRESHOLD", "0");
            GgufScopedEnvironment import_f16("OV_GPU_GGUF_PREFILL_F16", "0");
            GgufScopedEnvironment import_dq("OV_GPU_GGUF_PREFILL_DQ_ACT", "1");
            compiled = import_core.import_model(blob, "GPU", ov::hint::inference_precision(inference_type),
                                                ov::enable_profiling(true), ov::num_streams(stream_count));
            const int imported_streams = compiled.get_property(ov::num_streams);
            EXPECT_EQ(imported_streams, stream_count);
            RecordProperty(scenario + "_imported_streams", std::to_string(imported_streams));
        }
        std::vector<ov::InferRequest> requests{compiled.create_infer_request()};
        if (stream_count == 2)
            requests.push_back(compiled.create_infer_request());
        RecordProperty(scenario + "_requests", std::to_string(requests.size()));
        const std::vector<size_t> row_counts = static_rows ? std::vector<size_t>{static_rows, static_rows}
                                                        : std::vector<size_t>{1, 2, 33, 1, 1, 33, 2};
        for (size_t iteration = 0; iteration < row_counts.size(); ++iteration) {
            const auto rows = row_counts[iteration];
            const bool broadcast_peer = rebind_broadcast && rows > 1 && iteration % 2 == 0;
            for (size_t request_index = 0; request_index < requests.size(); ++request_index) {
                auto& request = requests[request_index];
                ov::Tensor input(ov::element::f32, ov::Shape{1, 1, rows, columns});
                for (size_t row = 0; row < rows; ++row) {
                    const float factor = static_cast<float>((row % 3 + 1) * (iteration % 2 + 1) *
                                                            (request_index * 2 + 1));
                    for (size_t column = 0; column < columns; ++column) {
                        input.data<float>()[row * columns + column] = (binary_mode ? 1.f / 64.f : 1.f) * factor *
                            static_cast<float>(column / 256 + 1) *
                            static_cast<float>(static_cast<int>(((column % 256) * 13) % 31) - 15);
                    }
                }
                request.set_input_tensor(0, input);
                ov::Tensor indices;
                if (has_gather) {
                    indices = ov::Tensor(ov::element::i64, ov::Shape{1, 1, 1, rows});
                    for (size_t row = 0; row < rows; ++row) {
                        indices.data<int64_t>()[row] = static_cast<int64_t>((row * 7 + iteration + request_index) % channels);
                    }
                    request.set_input_tensor(1, indices);
                }
                ov::Tensor peer;
                if (binary_mode) {
                    peer = ov::Tensor(ov::element::f32, ov::Shape{1, 1, broadcast_peer ? 1 : rows, channels});
                    for (size_t index = 0; index < peer.get_size(); ++index) {
                        peer.data<float>()[index] = static_cast<float>(static_cast<int>((index + iteration * 3) % 7) - 3) / 4.f;
                        if (std::string(binary_mode) == "sum") {
                            float fc_reference = 0.f;
                            for (size_t column = 0; column < columns; ++column)
                                fc_reference += decoded[((index % channels + source_row_offset) % 4) * 256 + column % 256] *
                                                input.data<const float>()[index / channels * columns + column];
                            peer.data<float>()[index] += std::ceil(std::abs(fc_reference)) + 2.f;
                        }
                    }
                    request.set_input_tensor(2, peer);
                }
                request.start_async();
            }
            for (auto& request : requests)
                request.wait();
            for (size_t request_index = 0; request_index < requests.size(); ++request_index) {
                auto& request = requests[request_index];
                SCOPED_TRACE(request_index);
                const auto request_scenario = scenario + (stream_count == 2 ? "_request" + std::to_string(request_index) : "");
                const auto input = request.get_input_tensor(0);
                const auto indices = has_gather ? request.get_input_tensor(1) : ov::Tensor{};
                const auto peer = binary_mode ? request.get_input_tensor(2) : ov::Tensor{};
                std::string iteration_stages;
                std::string iteration_fusions;
                for (const auto& operation : compiled.get_runtime_model()->get_ops()) {
                    const auto& info = operation->get_rt_info();
                    const auto stages = info.find("gguf_last_executed_kernels");
                    if (stages != info.end()) {
                        const auto& names = stages->second.as<std::vector<std::string>>();
                        executed_stages.insert(names.begin(), names.end());
                        for (const auto& name : names)
                            iteration_stages += name + ";";
                    }
                    const auto fusions = info.find("gguf_runtime_fused_ids");
                    if (fusions != info.end()) {
                        iteration_fusions += info.at("gguf_runtime_fused_count").as<std::string>() + ":" +
                                             fusions->second.as<std::string>() + ";";
                    }
                }
                RecordProperty(request_scenario + "_stage_" + std::to_string(iteration), iteration_stages);
                if (!has_gather) {
                    const auto weight_name = gguf_raw_element_type(format.type).to_string();
                    const bool prefill_stage = !threshold_value && rows > 1;
                    const std::string expected_stage = prefill_stage
                        ? "fc_gguf_transcode_" + weight_name : "fc_gguf_opt_" + weight_name;
                    // A shuffled prefill node may run the OCL GEMM when the device supports it.
                    const bool ocl_gemm_executed = prefill_stage && shuffled && !cm_requested &&
                                                   iteration_stages.find("fc_gguf_gemm_") != std::string::npos;
                    EXPECT_TRUE(ocl_gemm_executed || iteration_stages.find(expected_stage) != std::string::npos)
                        << iteration_stages;
                    EXPECT_EQ(iteration_stages.find("_sg"), std::string::npos) << iteration_stages;
                    EXPECT_EQ(iteration_stages.find("_full"), std::string::npos) << iteration_stages;
                }
                EXPECT_EQ(iteration_stages.find("fc_gguf_act_dq"), std::string::npos) << iteration_stages;
                if (binary_mode) {
                    const bool require_fused = !static_rows || (shuffled && (cm_requested || static_rows > 1));
                    const bool binary_fused = iteration_fusions.find("file_binary_post") != std::string::npos;
                    if (require_fused)
                        EXPECT_TRUE(binary_fused) << iteration_fusions;
                    EXPECT_EQ(iteration_fusions.rfind("0:", 0) == 0u, !binary_fused) << iteration_fusions;
                    RecordProperty(request_scenario + "_fusions_" + std::to_string(iteration), iteration_fusions);
                }
                if (binary_mode || stream_count == 2) {
                    const bool prefill = !threshold_value && rows > 1;
                    const std::string sg_stage = format.type == GGUF_TYPE_Q4_0 ? "fc_gguf_q4_0_sg"
                        : format.type == GGUF_TYPE_Q4_1 ? "fc_gguf_q4_1_sg"
                        : format.type == GGUF_TYPE_Q4_K ? "fc_gguf_q4k_sg"
                        : format.type == GGUF_TYPE_Q5_K ? "fc_gguf_q5k_sg"
                        : format.type == GGUF_TYPE_Q6_K ? "fc_gguf_q6k_sg" : "fc_gguf_q8_0_sg";
                    const std::string cm_stage = format.type == GGUF_TYPE_Q4_0 ? "gemm_q40_full"
                        : format.type == GGUF_TYPE_Q4_1 ? "gemm_q41_full"
                        : format.type == GGUF_TYPE_Q4_K ? "gemm_q4k_full"
                        : format.type == GGUF_TYPE_Q5_K ? "gemm_q5k_full"
                        : format.type == GGUF_TYPE_Q6_K ? "gemm_q6k_full" : "gemm_q80_full";
                    const auto weight_name = gguf_raw_element_type(format.type).to_string();
                    const std::string expected_stage = broadcast_peer ? sg_stage
                        : prefill ? (cm_requested && shuffled ? cm_stage : "fc_gguf_transcode_" + weight_name)
                        : shuffled ? sg_stage : "fc_gguf_opt_" + weight_name;
                    const bool ocl_gemm_executed = prefill && shuffled && !cm_requested && !broadcast_peer &&
                                                   iteration_stages.find("fc_gguf_gemm_") != std::string::npos;
                    EXPECT_TRUE(ocl_gemm_executed || iteration_stages.find(expected_stage) != std::string::npos)
                        << iteration_stages;
                }
                const auto output = request.get_output_tensor(0);
                ASSERT_EQ(output.get_shape(), (ov::Shape{1, 1, rows, channels}));
                bool detects_missing_fc = false;
                bool detects_missing_binary = false;
                float max_fc_error = 0.f;
                for (size_t row = 0; row < rows; ++row) {
                    for (size_t channel = 0; channel < channels; ++channel) {
                        float expected = 0.f;
                        for (size_t column = 0; column < columns; ++column) {
                            const float weight = decoded[((channel + source_row_offset) % 4) * 256 + column % 256];
                            const float reference = !threshold_value && rows > 1
                                                        ? static_cast<float>(ov::float16(weight)) : weight;
                            expected += reference * input.data<const float>()[row * columns + column];
                        }
                        if (inference_type == ov::element::f16) {
                            expected = ov::float16(expected);
                        }
                        if (binary_mode) {
                            const float fc_reference = expected;
                            const auto operand = peer.data<const float>()[(broadcast_peer ? 0 : row) * channels + channel];
                            expected = std::string(binary_mode) == "sum" ? expected + operand : expected * operand;
                            if (inference_type == ov::element::f16)
                                expected = ov::float16(expected);
                            expected = std::max(expected, 0.f);
                            const float without_fc = std::string(binary_mode) == "sum" ? std::max(operand, 0.f) : 0.f;
                            detects_missing_fc |= std::abs(expected - without_fc) > 0.2f;
                            detects_missing_binary |= std::abs(expected - std::max(fc_reference, 0.f)) > 0.2f;
                        }
                        max_fc_error = std::max(max_fc_error,
                            std::abs(output.data<const float>()[row * channels + channel] - expected));
                        EXPECT_NEAR(output.data<const float>()[row * channels + channel], expected, 0.1f)
                            << "iteration=" << iteration << " row=" << row << " channel=" << channel;
                    }
                }
                RecordProperty(request_scenario + "_max_error_" + std::to_string(iteration), max_fc_error);
                if (binary_mode) {
                    EXPECT_TRUE(detects_missing_fc) << iteration;
                    EXPECT_TRUE(detects_missing_binary) << iteration;
                }
                if (has_gather) {
                    const auto embedding = request.get_output_tensor(1);
                    ASSERT_EQ(embedding.get_element_type(), ov::element::f32);
                    ASSERT_EQ(embedding.get_shape(), (ov::Shape{1, 1, rows, columns}));
                    for (size_t row = 0; row < rows; ++row) {
                        const auto channel = static_cast<size_t>(indices.data<const int64_t>()[row]);
                        for (size_t column = 0; column < columns; ++column) {
                            const float expected = ov::float16(decoded[((channel + source_row_offset) % 4) * 256 + column % 256]);
                            EXPECT_EQ(embedding.data<const float>()[row * columns + column], expected)
                                << "iteration=" << iteration << " row=" << row << " column=" << column;
                        }
                    }
                }
                EXPECT_EQ(owner.expired(), cache_roundtrip);
            }
        }
        RecordProperty(scenario + "_completed_requests", std::to_string(row_counts.size() * requests.size()));
        if (check_cache) {
            std::cout << "GGUF_FILE_CACHE_CHECK format=" << format.name << " scenario=" << scenario
                      << " hit=" << compiled.get_property(ov::loaded_from_cache) << std::endl;
        }
        if (stream_count == 2) {
            std::cout << "GGUF_FILE_STREAM_CHECK format=" << format.name << " scenario=" << scenario
                      << " requests=" << row_counts.size() * requests.size() << std::endl;
        }
    };
    for (const int streams : {1, 2}) {
        if (streams == 2 && !has_gather)
            continue;
        stream_count = streams;
        run("64", "1");
        run("64", "0");
        run(nullptr, "1");
        run(nullptr, "0");
        run("64", "1", ov::element::f16);
        run("64", "0", ov::element::f16);
        run(nullptr, "1", ov::element::f16);
        run(nullptr, "0", ov::element::f16);
    }
    stream_count = 1;
    if (has_gather) {
        for (const auto& precision : {ov::element::f32, ov::element::f16}) {
            for (const char* binary : {"sum", "prod"}) {
                for (const char* shuffle : {"0", "1"}) {
                    for (const char* threshold : {"64", static_cast<const char*>(nullptr)}) {
                        run(threshold, shuffle, precision, binary);
                        run(threshold, shuffle, precision, binary, 33);
                    }
                }
                run(nullptr, "1", precision, binary, 0, true);
            }
        }
    }
    {
        const auto mapped = owner.lock();
        ASSERT_TRUE(mapped);
        const auto fc = ov::as_type_ptr<ov::op::internal::FullyConnectedCompressed>(
            model->get_results().at(0)->get_input_node_shared_ptr(0));
        ASSERT_TRUE(fc);
        const auto raw = ov::as_type_ptr<ov::op::v0::Constant>(fc->get_input_node_shared_ptr(1));
        ASSERT_TRUE(raw);
        auto& info = raw->get_rt_info();
        EXPECT_EQ(info.at("gguf_ext_offset").as<uint64_t>(), static_cast<uint64_t>(
            static_cast<const std::byte*>(raw->get_data_ptr()) - mapped->data()));
        EXPECT_EQ(info.at("gguf_ext_size").as<uint64_t>(), raw->get_byte_size());
        std::stringstream external_xml, external_binary;
        serialize_ir(model, external_xml, external_binary);
        EXPECT_LT(external_binary.str().size(), payload.size());
        RecordProperty("external_reference_bin_bytes", std::to_string(external_binary.str().size()));
        RecordProperty("external_reference_raw_bytes", std::to_string(payload.size()));
        serialize_ir(model, m_path + ".xml", m_path + ".bin");
        {
            const auto external_model = core.read_model(m_path + ".xml");
            const auto external_fc = ov::as_type_ptr<ov::op::internal::FullyConnectedCompressed>(
                external_model->get_results().at(0)->get_input_node_shared_ptr(0));
            ASSERT_TRUE(external_fc);
            const auto external_raw = ov::as_type_ptr<ov::op::v0::Constant>(external_fc->get_input_node_shared_ptr(1));
            ASSERT_TRUE(external_raw);
            ASSERT_EQ(external_raw->get_byte_size(), payload.size());
            EXPECT_EQ(std::memcmp(external_raw->get_data_ptr(), payload.data(), payload.size()), 0);
            std::stringstream chained_xml, chained_binary;
            serialize_ir(external_model, chained_xml, chained_binary);
            RecordProperty("external_reference_chained_bin_bytes", std::to_string(chained_binary.str().size()));
            EXPECT_EQ(chained_binary.str().size(), external_binary.str().size());
            pugi::xml_document chained_document;
            ASSERT_TRUE(chained_document.load_string(chained_xml.str().c_str()));
            size_t chained_sources = 0;
            for (const auto layer : chained_document.child("net").child("layers").children("layer")) {
                if (const auto source = layer.child("data").attribute("source")) {
                    ++chained_sources;
                    EXPECT_EQ(std::string(source.value()), std::filesystem::path(m_path).filename().string());
                    EXPECT_EQ(std::string(layer.child("data").attribute("checksum").value()),
                              "fnv1a64:" + std::to_string(ov::util::fnv1a_64(payload.data(), payload.size())));
                }
            }
            EXPECT_EQ(chained_sources, 1);
            {
                std::ofstream file(m_path + ".chain.xml");
                file << chained_xml.str();
                ASSERT_TRUE(file);
            }
            {
                std::ofstream file(m_path + ".chain.bin", std::ios::binary);
                const auto bytes = chained_binary.str();
                file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
                ASSERT_TRUE(file);
            }
            {
                const auto chained_model = core.read_model(m_path + ".chain.xml");
                std::stringstream next_xml, next_binary;
                serialize_ir(chained_model, next_xml, next_binary);
                EXPECT_EQ(next_binary.str().size(), external_binary.str().size());
                RecordProperty("external_reference_next_bin_bytes", std::to_string(next_binary.str().size()));
            }
            std::filesystem::remove(m_path + ".chain.xml");
            std::filesystem::remove(m_path + ".chain.bin");
            const auto second_model = core.read_model(m_path + ".xml");
            const auto second_fc = ov::as_type_ptr<ov::op::internal::FullyConnectedCompressed>(
                second_model->get_results().at(0)->get_input_node_shared_ptr(0));
            ASSERT_TRUE(second_fc);
            const auto second_raw = ov::as_type_ptr<ov::op::v0::Constant>(second_fc->get_input_node_shared_ptr(1));
            ASSERT_TRUE(second_raw);
            ASSERT_NE(second_raw->get_data_ptr(), external_raw->get_data_ptr());
            for (const auto& operation : second_model->get_ordered_ops()) {
                operation->set_friendly_name("sharing_second_" + operation->get_friendly_name());
                for (auto output : operation->outputs()) {
                    output.set_names({operation->get_friendly_name() + "_" + std::to_string(output.get_index())});
                }
            }
            auto results = external_model->get_results();
            auto parameters = external_model->get_parameters();
            results.insert(results.end(), second_model->get_results().begin(), second_model->get_results().end());
            parameters.insert(parameters.end(), second_model->get_parameters().begin(), second_model->get_parameters().end());
            const auto shared_model = std::make_shared<ov::Model>(results, parameters);
            GgufScopedEnvironment sharing_shuffle("OV_GPU_GGUF_SHUFFLE", "0");
            GgufScopedEnvironment sharing_cm("OV_GPU_GGUF_PREFILL_GEMM", "ocl");
            auto compiled = core.compile_model(shared_model, "GPU");
            std::stringstream exported_binary;
            compiled.export_model(exported_binary);
            const std::string raw_bytes(reinterpret_cast<const char*>(payload.data()), payload.size());
            const auto exported = exported_binary.str();
            size_t copies = 0;
            size_t position = 0;
            while ((position = exported.find(raw_bytes, position)) != std::string::npos) {
                ++copies;
                position += raw_bytes.size();
            }
            RecordProperty("external_source_payload_copies", std::to_string(copies));
            EXPECT_EQ(copies, 1);
        }
        ov::Tensor external_weights(ov::element::u8, {external_binary.str().size()});
        const auto bin_bytes = external_binary.str();
        std::memcpy(external_weights.data(), bin_bytes.data(), bin_bytes.size());
        EXPECT_ANY_THROW(core.read_model(external_xml.str(), external_weights));
        const auto invalid_reference = [&](const char* attribute, const std::string& value) {
            pugi::xml_document document;
            ASSERT_TRUE(document.load_string(external_xml.str().c_str()));
            pugi::xml_node data;
            for (const auto layer : document.child("net").child("layers").children("layer")) {
                if (layer.child("data").attribute("source")) {
                    data = layer.child("data");
                    break;
                }
            }
            ASSERT_TRUE(data);
            data.attribute(attribute).set_value(value.c_str());
            {
                std::ofstream file(m_path + ".xml");
                ASSERT_TRUE(file);
                document.save(file);
            }
            EXPECT_ANY_THROW(core.read_model(m_path + ".xml"));
        };
        for (const char* source : {"", ".", "..", "../missing.gguf", "..\\missing.gguf", "C:missing.gguf"}) {
            invalid_reference("source", source);
        }
        invalid_reference("source", std::filesystem::path(m_path).filename().string() + ".missing");
        invalid_reference("offset", "18446744073709551615");
        invalid_reference("size", std::to_string(payload.size() - 1));
        invalid_reference("checksum", "");
        invalid_reference("checksum", "fnv1a64:0");
        invalid_reference("checksum", "unknown:0");
        {
            pugi::xml_document document;
            ASSERT_TRUE(document.load_string(external_xml.str().c_str()));
            for (const auto layer : document.child("net").child("layers").children("layer")) {
                layer.child("data").remove_attribute("checksum");
            }
            {
                std::ofstream file(m_path + ".xml");
                ASSERT_TRUE(file);
                document.save(file);
            }
            EXPECT_NO_THROW(core.read_model(m_path + ".xml"));
        }
        {
            const auto raw_path = m_path + ".raw";
            std::filesystem::copy_file(m_path, raw_path);
            {
                std::fstream file(raw_path, std::ios::binary | std::ios::in | std::ios::out);
                file.write("RAW!", 4);
                ASSERT_TRUE(file);
            }
            pugi::xml_document document;
            ASSERT_TRUE(document.load_string(external_xml.str().c_str()));
            for (const auto layer : document.child("net").child("layers").children("layer")) {
                if (auto source = layer.child("data").attribute("source")) {
                    source.set_value(std::filesystem::path(raw_path).filename().string().c_str());
                }
            }
            {
                std::ofstream file(m_path + ".xml");
                ASSERT_TRUE(file);
                document.save(file);
            }
            EXPECT_NO_THROW({
                const auto raw_model = core.read_model(m_path + ".xml");
                std::stringstream raw_xml;
                std::stringstream raw_binary;
                serialize_ir(raw_model, raw_xml, raw_binary);
            });
            std::filesystem::remove(raw_path);
        }
        info["gguf_ext_size"] = static_cast<uint64_t>(payload.size() - 1);
        std::stringstream invalid_xml, invalid_binary;
        EXPECT_ANY_THROW(serialize_ir(model, invalid_xml, invalid_binary));
        info["gguf_ext_size"] = static_cast<uint64_t>(payload.size());
        serialize_ir(model, m_path + ".xml", m_path + ".bin");
        info.erase("gguf_ext_source");
        info.erase("gguf_ext_offset");
        info.erase("gguf_ext_size");
        info.erase("gguf_ext_source_path");
    }
    std::stringstream xml, binary;
    serialize_ir(model, xml, binary);
    const auto serialized = binary.str();
    ov::Tensor constants(ov::element::u8, {serialized.size()});
    std::memcpy(constants.data(), serialized.data(), serialized.size());
    model = core.read_model(xml.str(), constants);
    cache_roundtrip = true;
    EXPECT_TRUE(owner.expired());
    size_t raw_constants = 0;
    for (const auto& node : model->get_ordered_ops()) {
        const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
        if (constant && constant->get_element_type().is_gguf_block()) {
            ++raw_constants;
            EXPECT_EQ(constant->get_element_type(), gguf_raw_element_type(format.type));
            EXPECT_EQ(constant->get_shape(), (ov::Shape{channels, columns}));
            ASSERT_EQ(constant->get_byte_size(), payload.size());
            EXPECT_EQ(std::memcmp(constant->get_data_ptr(), payload.data(), payload.size()), 0);
        }
    }
    EXPECT_EQ(raw_constants, 1);
    RecordProperty("public_ir_roundtrip_raw_constants", std::to_string(raw_constants));
    run("64", "0");
    run("64", "1", ov::element::f16);
    run(nullptr, "0");
    run(nullptr, "1", ov::element::f16);
    model = core.read_model(m_path + ".xml");
    external_reference = true;
    EXPECT_TRUE(owner.expired());
    auto replacement_payload = payload;
    std::rotate(replacement_payload.begin(), replacement_payload.begin() + 2 * row_bytes, replacement_payload.end());
    ASSERT_NO_FATAL_FAILURE(write_tensor("projection.weight", format.type, columns, channels, replacement_payload));
    {
        EXPECT_ANY_THROW(core.read_model(m_path + ".xml"));
        const auto original_fc = ov::as_type_ptr<ov::op::internal::FullyConnectedCompressed>(
            model->get_results().at(0)->get_input_node_shared_ptr(0));
        ASSERT_TRUE(original_fc);
        const auto original_raw = ov::as_type_ptr<ov::op::v0::Constant>(original_fc->get_input_node_shared_ptr(1));
        ASSERT_TRUE(original_raw);
        EXPECT_EQ(std::memcmp(original_raw->get_data_ptr(), payload.data(), payload.size()), 0);
        std::stringstream stale_xml, stale_binary;
        EXPECT_ANY_THROW(serialize_ir(model, stale_xml, stale_binary));
        RecordProperty("external_reference_stale_snapshot_rejected", "1");
        const auto materialized_model = model->clone();
        for (const auto& operation : materialized_model->get_ordered_ops()) {
            auto& metadata = operation->get_rt_info();
            for (const auto* key : {"gguf_ext_source", "gguf_ext_offset", "gguf_ext_size", "gguf_ext_source_path"}) {
                metadata.erase(key);
            }
        }
        std::stringstream materialized_xml, materialized_binary;
        serialize_ir(materialized_model, materialized_xml, materialized_binary);
        const auto materialized_bytes = materialized_binary.str();
        ASSERT_GE(materialized_bytes.size(), payload.size());
        ov::Tensor materialized_weights(ov::element::u8, {materialized_bytes.size()});
        std::memcpy(materialized_weights.data(), materialized_bytes.data(), materialized_bytes.size());
        const auto materialized = core.read_model(materialized_xml.str(), materialized_weights);
        const auto materialized_fc = ov::as_type_ptr<ov::op::internal::FullyConnectedCompressed>(
            materialized->get_results().at(0)->get_input_node_shared_ptr(0));
        ASSERT_TRUE(materialized_fc);
        const auto materialized_raw = ov::as_type_ptr<ov::op::v0::Constant>(materialized_fc->get_input_node_shared_ptr(1));
        ASSERT_TRUE(materialized_raw);
        EXPECT_EQ(std::memcmp(materialized_raw->get_data_ptr(), payload.data(), payload.size()), 0);
    }
    run("64", "0");
    run("64", "1", ov::element::f16);
    run(nullptr, "0");
    run(nullptr, "1", ov::element::f16);
    compiled_import = true;
    for (const int streams : {1, 2}) {
        if (streams == 2 && !has_gather)
            continue;
        stream_count = streams;
        run("64", "0");
        run("64", "1", ov::element::f16);
        run(nullptr, "0");
        run(nullptr, "1", ov::element::f16);
    }
    EXPECT_FALSE(executed_stages.empty());
    std::string stage_summary;
    for (const auto& stage : executed_stages) {
        stage_summary += stage + ";";
    }
    RecordProperty("gguf_executed_kernels", stage_summary);
    model.reset();
    EXPECT_TRUE(owner.expired());
#endif
}

INSTANTIATE_TEST_SUITE_P(RawFormats,
                        GGUFParserGpu,
                        ::testing::ValuesIn(kRawFormats),
                        [](const ::testing::TestParamInfo<GgufRawFormat>& info) { return info.param.name; });

TEST_F(GGUFParser, Q4KRawRejectsInvalidRowsAndTruncatedPayload) {
    const GGUFLoadOptions options{true};
    for (const uint64_t cols : {128, 255, 257}) {
        SCOPED_TRACE(cols);
        ASSERT_NO_FATAL_FAILURE(write_tensor("invalid.weight", GGUF_TYPE_Q4_K, cols, 2, std::vector<uint8_t>(288)));
        OV_EXPECT_THROW(get_gguf_data(m_path, options), ov::Exception, testing::HasSubstr("block size"));
    }
    for (const uint64_t rows : {1, 2}) {
        SCOPED_TRACE(rows);
        ASSERT_NO_FATAL_FAILURE(
            write_tensor("truncated.weight", GGUF_TYPE_Q4_K, 256, rows, std::vector<uint8_t>(rows * 144 - 1)));
        OV_EXPECT_THROW(get_gguf_data(m_path, options), ov::Exception, testing::HasSubstr("past EOF"));
    }
}

TEST_F(GGUFParser, RawFormatsRejectInvalidRowsAndTruncatedPayload) {
    const GGUFLoadOptions options{true, true, true, true, true, true, true, true, true, true, true};
    for (const auto& format : kRawFormats) {
        SCOPED_TRACE(format.name);
        const auto block = gguf_raw_element_type(format.type).block_elem_count();
        for (const auto columns : {block - 1, block + 1}) {
            ASSERT_NO_FATAL_FAILURE(write_tensor("invalid.weight", format.type, columns, 2));
            OV_EXPECT_THROW(get_gguf_data(m_path, options), ov::Exception, testing::HasSubstr("block size"));
        }
        const auto bytes = load_npy<uint8_t>(std::string(format.name) + "_qbytes");
        ASSERT_NO_FATAL_FAILURE(write_tensor("truncated.weight", format.type, 256, 4,
                                             std::vector<uint8_t>(bytes.size() - 1)));
        OV_EXPECT_THROW(get_gguf_data(m_path, options), ov::Exception, testing::HasSubstr("past EOF"));
    }
}

TEST_F(GGUFParser, Q4KRawRejectsOverflowAndInvalidOffsets) {
    const GGUFLoadOptions options{true};
    ASSERT_NO_FATAL_FAILURE(write_tensor("overflow.weight",
                                         GGUF_TYPE_Q4_K,
                                         256,
                                         std::numeric_limits<uint64_t>::max() / 256 + 1));
    OV_EXPECT_THROW(get_gguf_data(m_path, options), ov::Exception, testing::HasSubstr("element count overflows"));
    ASSERT_NO_FATAL_FAILURE(write_tensor("offset.weight",
                                         GGUF_TYPE_Q4_K,
                                         256,
                                         1,
                                         std::vector<uint8_t>(144),
                                         {},
                                         {},
                                         std::numeric_limits<uint64_t>::max()));
    OV_EXPECT_THROW(get_gguf_data(m_path, options), ov::Exception, testing::HasSubstr("offset"));
}

TEST_F(GGUFParser, Q4KRawAcceptsEmptyTensorAndKeepsOtherFormatsDecomposed) {
    ASSERT_NO_FATAL_FAILURE(write_tensor("empty.weight", GGUF_TYPE_Q4_K, 256, 0));
    {
        const auto empty = get_gguf_data(m_path, GGUFLoadOptions{true});
        const auto raw = unpack_gguf_weight(std::get<1>(empty).at("empty.weight")).weight;
        EXPECT_EQ(raw.get_element_type(), ov::element::gguf_q4_k);
        EXPECT_EQ(raw.get_shape(), (ov::Shape{0, 256}));
        EXPECT_EQ(raw.get_byte_size(), 0);
    }
    ASSERT_NO_FATAL_FAILURE(write_tensor("other.weight", GGUF_TYPE_Q8_0, 32, 1, std::vector<uint8_t>(34)));
    const auto other = get_gguf_data(m_path, GGUFLoadOptions{true});
    const auto unpacked = unpack_gguf_weight(std::get<1>(other).at("other.weight"));
    EXPECT_EQ(unpacked.weight.get_element_type(), ov::element::i8);
    EXPECT_TRUE(unpacked.scales);
}

TEST_F(GGUFParser, QuantizedBuffersOutliveParserResult) {
    const auto bytes = load_npy<uint8_t>("q2_k_qbytes");
    const auto reference = load_npy<float>("q2_k_deq");
    ASSERT_NO_FATAL_FAILURE(write_tensor("blk.0.attn_q.weight", GGUF_TYPE_Q2_K, 256, 4, bytes));
    WeightTensors tensors;
    {
        const auto loaded = get_gguf_data(m_path);
        const auto& arrays = std::get<1>(loaded);
        tensors = unpack_gguf_weight(arrays.at("blk.0.attn_q.weight"));
    }
    ov::test::utils::removeFile(m_path);
    const auto node = make_weight_node(tensors, GGUF_TYPE_Q2_K, "blk.0.attn_q");
    tensors = {};
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{node}, ov::ParameterVector{});
    const auto output = run_on_cpu(model, {});
    ASSERT_EQ(output.get_size(), reference.size());
    for (size_t i = 0; i < reference.size(); ++i)
        EXPECT_NEAR(output.data<const float>()[i], reference[i], 3e-3f);
}

TEST_F(GGUFParser, AcceptsEmptyQuantizedTensors) {
    constexpr std::array types{
        std::pair{"Q4_0", GGUF_TYPE_Q4_0},
        std::pair{"Q4_1", GGUF_TYPE_Q4_1},
        std::pair{"Q5_0", GGUF_TYPE_Q5_0},
        std::pair{"Q5_1", GGUF_TYPE_Q5_1},
        std::pair{"Q8_0", GGUF_TYPE_Q8_0},
        std::pair{"Q8_1", GGUF_TYPE_Q8_1},
        std::pair{"Q2_K", GGUF_TYPE_Q2_K},
        std::pair{"Q2_0", GGUF_TYPE_Q2_0},
        std::pair{"Q3_K", GGUF_TYPE_Q3_K},
        std::pair{"Q4_K", GGUF_TYPE_Q4_K},
        std::pair{"Q5_K", GGUF_TYPE_Q5_K},
        std::pair{"Q6_K", GGUF_TYPE_Q6_K},
        std::pair{"Q8_K", GGUF_TYPE_Q8_K},
    };
    for (const auto& [name, type] : types) {
        SCOPED_TRACE(name);
        ASSERT_NO_FATAL_FAILURE(write_tensor("empty.weight", type, 0, 2));
        const auto unpacked = unpack_gguf_weight(std::get<1>(get_gguf_data(m_path)).at("empty.weight"));
        EXPECT_EQ(unpacked.weight.get_shape(), (ov::Shape{2, 0}));
        EXPECT_EQ(unpacked.scales.get_shape(), (ov::Shape{2, 0}));
        for (const auto& tensor : {unpacked.weight, unpacked.scales, unpacked.zero_point}) {
            if (tensor) {
                EXPECT_EQ(tensor.get_size(), 0);
            }
        }
    }
}

TEST_F(GGUFParser, RejectsMXFP4WithZeroColumns) {
    ASSERT_NO_FATAL_FAILURE(write_tensor("empty.weight", GGUF_TYPE_MXFP4, 0, 2));
    try {
        get_gguf_data(m_path);
        FAIL() << "MXFP4 with zero columns must be rejected";
    } catch (const ov::Exception& e) {
        EXPECT_NE(std::string(e.what()).find("zero-sized dimension"), std::string::npos);
    }
}

class GGUFShards : public ::testing::Test {
protected:
    void SetUp() override {
        m_base = ov::test::utils::generateTestFilePrefix();
    }

    void TearDown() override {
        for (const auto& path : m_paths) {
            if (std::filesystem::exists(path)) {
                EXPECT_TRUE(std::filesystem::is_regular_file(path)) << path;
                std::filesystem::remove(path);
            }
        }
    }

    std::string path(size_t index) const {
        return m_base + "-0000" + std::to_string(index + 1) + "-of-00002.gguf";
    }

    void write_shard(size_t index,
                     uint16_t number,
                     uint16_t count = 2,
                     int32_t total = 2,
                     const std::string& name = {},
                     const std::string& architecture = {},
                     bool raw = false) {
        GgufWriter writer;
        writer.kv_u16("split.no", number);
        writer.kv_u16("split.count", count);
        writer.kv_i32("split.tensors.count", total);
        if (index == 0 || !architecture.empty()) {
            writer.kv_str("general.architecture", architecture.empty() ? "qwen3" : architecture);
        }
        const auto tensor_name = name.empty() ? "weight" + std::to_string(index) + ".weight" : name;
        if (raw) {
            writer.raw_tensor(tensor_name, {256, 4}, GGUF_TYPE_Q4_K, load_npy<uint8_t>("q4_k_qbytes"));
        } else {
            writer.tensor(tensor_name, {2}, {float(index + 1), float(index + 2)});
        }
        const auto file = path(index);
        m_paths.push_back(file);
        ASSERT_TRUE(writer.write(file));
    }

    std::string m_base;
    std::vector<std::filesystem::path> m_paths;
};

TEST_F(GGUFShards, DiscoversSiblingsFromEitherShardAndKeepsMappingsAlive) {
    ASSERT_NO_FATAL_FAILURE(write_shard(0, 0));
    ASSERT_NO_FATAL_FAILURE(write_shard(1, 1));
    ov::frontend::gguf::GgufWeight retained;
    for (size_t index : {0, 1}) {
        const auto loaded = get_gguf_data(path(index));
        EXPECT_EQ(std::get<2>(loaded).size(), 2);
        EXPECT_EQ(std::get<std::string>(std::get<0>(loaded).at("general.architecture")), "qwen3");
        const auto& arrays = std::get<1>(loaded);
        ASSERT_EQ(arrays.size(), 2);
        EXPECT_FLOAT_EQ(arrays.at("weight0.weight").data.data<const float>()[0], 1.f);
        EXPECT_FLOAT_EQ(arrays.at("weight1.weight").data.data<const float>()[0], 2.f);
        retained = arrays.at("weight1.weight");
    }
    EXPECT_FLOAT_EQ(retained.data.data<const float>()[1], 3.f);
}

TEST_F(GGUFShards, ExplicitListAllowsRenamedFilesInDifferentDirectoriesAndReversedOrder) {
    for (const bool raw : {false, true}) {
        ASSERT_NO_FATAL_FAILURE(write_shard(0, 0, 2, 2, {}, {}, raw));
        ASSERT_NO_FATAL_FAILURE(write_shard(1, 1, 2, 2, {}, {}, raw));
        const auto filename = raw ? std::filesystem::path(path(0)).filename().string()
                                  : std::filesystem::path(m_base).filename().string() + "_renamed.gguf";
        const auto renamed = (std::filesystem::temp_directory_path() / filename).string();
        m_paths.push_back(renamed);
        ASSERT_TRUE(std::filesystem::copy_file(path(1), renamed));
        const auto loaded = get_gguf_data(std::vector<std::filesystem::path>{renamed, path(0)}, GGUFLoadOptions{raw});
        EXPECT_EQ(std::get<1>(loaded).size(), 2);
        EXPECT_EQ(std::get<std::string>(std::get<0>(loaded).at("general.architecture")), "qwen3");
        if (raw) {
            EXPECT_TRUE(std::get<1>(loaded).at("weight0.weight").external_source.empty());
            EXPECT_TRUE(std::get<1>(loaded).at("weight1.weight").external_source.empty());
        }
        EXPECT_EQ(unpack_gguf_weight(std::get<1>(loaded).at("weight0.weight")).weight.get_element_type(),
                  raw ? ov::element::gguf_q4_k : ov::element::f32);
        OV_EXPECT_THROW(get_gguf_data(renamed), ov::Exception, testing::HasSubstr("explicit list"));
    }
}

TEST_F(GGUFShards, LegacyGroup128PackingMatchesMergedFileAcrossShards) {
    const auto oracle_bytes = load_npy<uint8_t>("pq2_0_qbytes");
    const auto oracle_values = load_npy<float>("pq2_0_deq");
    ASSERT_GE(oracle_bytes.size(), 34);
    ASSERT_GE(oracle_values.size(), 128);
    const std::vector<uint8_t> block(oracle_bytes.begin(), oracle_bytes.begin() + 34);
    std::vector<uint8_t> large;
    for (size_t i = 0; i < 16; ++i) {
        large.insert(large.end(), block.begin(), block.end());
    }
    const auto merged_path = m_base + ".gguf";
    m_paths.push_back(merged_path);
    for (bool ambiguous_first : {false, true}) {
        SCOPED_TRACE(ambiguous_first);
        GgufWriter merged;
        for (size_t index : {0, 1}) {
            GgufWriter shard;
            shard.kv_u16("split.no", static_cast<uint16_t>(index));
            shard.kv_u16("split.count", 2);
            shard.kv_i32("split.tensors.count", 2);
            const bool ambiguous = (index == 0) == ambiguous_first;
            const auto name = ambiguous ? "small.weight" : "large.weight";
            const std::vector<uint64_t> dims{128, ambiguous ? 1u : 16u};
            const auto& data = ambiguous ? block : large;
            shard.raw_tensor(name, dims, GGUF_TYPE_Q2_0, data);
            merged.raw_tensor(name, dims, GGUF_TYPE_Q2_0, data);
            m_paths.push_back(path(index));
            ASSERT_TRUE(shard.write(path(index)));
        }
        ASSERT_TRUE(merged.write(merged_path));
        const auto reference = get_gguf_data(merged_path);
        for (const auto& loaded : {get_gguf_data(path(0)),
                                   get_gguf_data(path(1)),
                                   get_gguf_data(std::vector<std::filesystem::path>{path(1), path(0)})}) {
            for (const std::string stem : {"small", "large"}) {
                const auto& source = std::get<1>(loaded).at(stem + ".weight");
                EXPECT_EQ(source.qtype, GGUF_TYPE_PQ2_0);
                auto actual = unpack_gguf_weight(source);
                auto expected = unpack_gguf_weight(std::get<1>(reference).at(stem + ".weight"));
                for (const auto& [lhs, rhs] : {std::pair{actual.weight, expected.weight},
                                               std::pair{actual.scales, expected.scales},
                                               std::pair{actual.zero_point, expected.zero_point}}) {
                    ASSERT_EQ(lhs.get_element_type(), rhs.get_element_type());
                    ASSERT_EQ(lhs.get_shape(), rhs.get_shape());
                    EXPECT_EQ(std::memcmp(lhs.data(), rhs.data(), lhs.get_byte_size()), 0);
                }
                const auto node = make_weight_node(actual, GGUF_TYPE_PQ2_0);
                const auto model = std::make_shared<ov::Model>(ov::OutputVector{node}, ov::ParameterVector{});
                const auto output = run_on_cpu(model, {});
                for (size_t i = 0; i < output.get_size(); ++i) {
                    EXPECT_FLOAT_EQ(output.data<float>()[i], oracle_values[i % 128]);
                }
            }
        }
    }
}

TEST_F(GGUFShards, ReportsMissingSibling) {
    ASSERT_NO_FATAL_FAILURE(write_shard(0, 0));
    OV_EXPECT_THROW(get_gguf_data(path(0)), ov::Exception, testing::HasSubstr("missing GGUF shard"));
}

TEST_F(GGUFShards, RejectsIncompleteExplicitList) {
    ASSERT_NO_FATAL_FAILURE(write_shard(0, 0));
    OV_EXPECT_THROW(get_gguf_data(std::vector<std::filesystem::path>{path(0)}),
                    ov::Exception,
                    testing::HasSubstr("expected 2 GGUF shards"));
    ov::frontend::gguf::FrontEnd frontend;
    OV_EXPECT_THROW(frontend.load(std::vector<std::string>{path(0)}),
                    ov::Exception,
                    testing::HasSubstr("expected 2 GGUF shards"));
}

TEST_F(GGUFShards, RejectsDuplicateShardIndices) {
    ASSERT_NO_FATAL_FAILURE(write_shard(0, 0));
    ASSERT_NO_FATAL_FAILURE(write_shard(1, 0));
    OV_EXPECT_THROW(get_gguf_data(std::vector<std::filesystem::path>{path(0), path(1)}),
                    ov::Exception,
                    testing::HasSubstr("duplicate GGUF shard index"));
    OV_EXPECT_THROW(get_gguf_data(path(0)), ov::Exception, testing::HasSubstr("split.no does not match"));
}

TEST_F(GGUFShards, RejectsInconsistentShardCounts) {
    ASSERT_NO_FATAL_FAILURE(write_shard(0, 0));
    ASSERT_NO_FATAL_FAILURE(write_shard(1, 1, 3));
    OV_EXPECT_THROW(get_gguf_data(path(0)), ov::Exception, testing::HasSubstr("inconsistent split.count"));
}

TEST_F(GGUFShards, RejectsInconsistentTensorCounts) {
    ASSERT_NO_FATAL_FAILURE(write_shard(0, 0));
    ASSERT_NO_FATAL_FAILURE(write_shard(1, 1, 2, 3));
    OV_EXPECT_THROW(get_gguf_data(path(0)), ov::Exception, testing::HasSubstr("inconsistent split.tensors.count"));
}

TEST_F(GGUFShards, RejectsMissingTensors) {
    ASSERT_NO_FATAL_FAILURE(write_shard(0, 0, 2, 3));
    ASSERT_NO_FATAL_FAILURE(write_shard(1, 1, 2, 3));
    OV_EXPECT_THROW(get_gguf_data(path(0)), ov::Exception, testing::HasSubstr("declares 3 tensors, found 2"));
}

TEST_F(GGUFShards, RejectsDuplicateTensorNames) {
    ASSERT_NO_FATAL_FAILURE(write_shard(0, 0));
    ASSERT_NO_FATAL_FAILURE(write_shard(1, 1, 2, 2, "weight0.weight"));
    OV_EXPECT_THROW(get_gguf_data(path(0)), ov::Exception, testing::HasSubstr("duplicate tensor name"));
}

TEST_F(GGUFShards, RejectsConflictingModelMetadata) {
    ASSERT_NO_FATAL_FAILURE(write_shard(0, 0));
    ASSERT_NO_FATAL_FAILURE(write_shard(1, 1, 2, 2, {}, "llama"));
    OV_EXPECT_THROW(get_gguf_data(path(0)), ov::Exception, testing::HasSubstr("inconsistent metadata"));
}

TEST_F(GGUFShards, ComparesRepeatedNumericAndStringArrayMetadata) {
    for (size_t index : {0, 1}) {
        GgufWriter writer;
        writer.kv_u16("split.no", static_cast<uint16_t>(index));
        writer.kv_u16("split.count", 2);
        writer.kv_i32("split.tensors.count", 2);
        writer.kv_str("general.architecture", "qwen3");
        writer.kv_f32("test.scalar", 1.5f);
        writer.kv_u32_array("test.array", {1, 2});
        writer.kv_u32_array("test.empty", {});
        writer.kv_str_array("test.strings", {"one", "two"});
        writer.tensor("weight" + std::to_string(index), {2});
        m_paths.push_back(path(index));
        ASSERT_TRUE(writer.write(path(index)));
    }
    EXPECT_EQ(std::get<1>(get_gguf_data(path(0))).size(), 2);
}

TEST_F(GGUFShards, RejectsConflictingNumericMetadata) {
    enum class Difference { ScalarValue, ScalarType, ArrayValue, ArrayType, ArrayShape };
    for (auto difference : {Difference::ScalarValue,
                            Difference::ScalarType,
                            Difference::ArrayValue,
                            Difference::ArrayType,
                            Difference::ArrayShape}) {
        SCOPED_TRACE(static_cast<int>(difference));
        for (size_t index : {0, 1}) {
            GgufWriter writer;
            writer.kv_u16("split.no", static_cast<uint16_t>(index));
            writer.kv_u16("split.count", 2);
            writer.kv_i32("split.tensors.count", 2);
            writer.kv_str("general.architecture", "qwen3");
            if (index == 1 && difference == Difference::ScalarType) {
                writer.kv_u32("test.scalar", 1);
            } else {
                writer.kv_f32("test.scalar", index == 1 && difference == Difference::ScalarValue ? 2.f : 1.f);
            }
            if (index == 1 && difference == Difference::ArrayType) {
                writer.kv_u16_array("test.array", {1, 2});
            } else {
                std::vector<uint32_t> values{1, 2};
                if (index == 1 && difference == Difference::ArrayValue) {
                    values[1] = 3;
                } else if (index == 1 && difference == Difference::ArrayShape) {
                    values.resize(1);
                }
                writer.kv_u32_array("test.array", values);
            }
            writer.tensor("weight" + std::to_string(index), {2});
            m_paths.push_back(path(index));
            ASSERT_TRUE(writer.write(path(index)));
        }
        OV_EXPECT_THROW(get_gguf_data(path(0)), ov::Exception, testing::HasSubstr("inconsistent metadata"));
    }
}

TEST_F(GGUFShards, RejectsOutOfRangeShardNumber) {
    ASSERT_NO_FATAL_FAILURE(write_shard(0, 2));
    OV_EXPECT_THROW(get_gguf_data(path(0)), ov::Exception, testing::HasSubstr("split.no is out of range"));
}

TEST_F(GGUFShards, RejectsInvalidSplitMetadataType) {
    GgufWriter writer;
    writer.kv_u32("split.count", 2);
    m_paths.push_back(path(0));
    ASSERT_TRUE(writer.write(path(0)));
    OV_EXPECT_THROW(get_gguf_data(path(0)), ov::Exception, testing::HasSubstr("split.count"));
}

TEST_F(GGUFShards, RejectsNegativeTensorCountAndArraySplitMetadata) {
    ASSERT_NO_FATAL_FAILURE(write_shard(0, 0, 2, -1));
    ASSERT_NO_FATAL_FAILURE(write_shard(1, 1, 2, -1));
    OV_EXPECT_THROW(get_gguf_data(path(0)), ov::Exception, testing::HasSubstr("must be nonnegative"));
    GgufWriter invalid;
    invalid.kv_u16("split.count", 2);
    invalid.kv_u16_array("split.no", {0});
    ASSERT_TRUE(invalid.write(path(0)));
    OV_EXPECT_THROW(get_gguf_data(path(0)), ov::Exception, testing::HasSubstr("must be a scalar"));
}

TEST_F(GGUFShards, AcceptsMetadataOnlyFirstShard) {
    GgufWriter writer;
    writer.kv_str("general.architecture", "qwen3");
    writer.kv_u16("split.no", 0);
    writer.kv_u16("split.count", 2);
    writer.kv_i32("split.tensors.count", 1);
    m_paths.push_back(path(0));
    ASSERT_TRUE(writer.write(path(0)));
    ASSERT_NO_FATAL_FAILURE(write_shard(1, 1, 2, 1));
    const auto loaded = get_gguf_data(path(1));
    EXPECT_EQ(std::get<1>(loaded).size(), 1);
    EXPECT_EQ(std::get<std::string>(std::get<0>(loaded).at("general.architecture")), "qwen3");
}

TEST_F(GGUFShards, AcceptsSingleShardAndMergedSplitMetadata) {
    ASSERT_NO_FATAL_FAILURE(write_shard(0, 0, 1, 1));
    EXPECT_EQ(std::get<1>(get_gguf_data(path(0))).size(), 1);
    ASSERT_NO_FATAL_FAILURE(write_shard(0, 0, 0, 1));
    EXPECT_EQ(std::get<1>(get_gguf_data(path(0))).size(), 1);
}

TEST_F(GGUFShards, RejectsUnrelatedModelsAndEmptyLists) {
    GgufWriter writer;
    writer.tensor("weight", {1}, {1.f});
    for (size_t index : {0, 1}) {
        m_paths.push_back(path(index));
        ASSERT_TRUE(writer.write(path(index)));
    }
    OV_EXPECT_THROW(get_gguf_data(std::vector<std::filesystem::path>{path(0), path(1)}),
                    ov::Exception,
                    testing::HasSubstr("one split GGUF model"));
    OV_EXPECT_THROW(get_gguf_data(std::vector<std::filesystem::path>{}), ov::Exception, testing::HasSubstr("empty"));
}

static std::shared_ptr<ov::Model> load_legacy_weight_reference(const std::string& path) {
    FrontEnd frontend;
    auto builder = load_gguf_builder(std::filesystem::path(path), default_arch_registry(), {});
    auto input = std::make_shared<ov::frontend::gguf::InputModel>(std::move(builder),
                                                                 std::vector<ov::Extension::Ptr>{});
    return frontend.convert(input);
}

class GGUFProjection : public GGUFShards {
protected:
    static constexpr uint32_t vocabulary_size = 128;
    std::string write_projection_model(uint32_t width,
                                       GgufTensorType type,
                                       const std::vector<uint8_t>& payload,
                                       uint32_t ffn_channels = 0,
                                       bool output_projection = false,
                                       bool tied_output = false,
                                       bool soft_cap = false);
};

std::string GGUFProjection::write_projection_model(uint32_t width,
                                                   GgufTensorType type,
                                                   const std::vector<uint8_t>& payload,
                                                   uint32_t ffn_channels,
                                                   bool output_projection,
                                                   bool tied_output,
                                                   bool soft_cap) {
    GgufWriter writer;
    writer.kv_str("general.architecture", "qwen3");
    writer.kv_u32("qwen3.block_count", 1);
    writer.kv_u32("qwen3.embedding_length", width);
    writer.kv_u32("qwen3.attention.head_count", 2);
    writer.kv_u32("qwen3.attention.head_count_kv", 1);
    writer.kv_u32("qwen3.rope.dimension_count", width / 2);
    writer.kv_u32("qwen3.context_length", 32);
    writer.kv_f32("qwen3.attention.layer_norm_rms_epsilon", 1e-5f);
    if (soft_cap)
        writer.kv_f32("qwen3.final_logit_softcapping", 2.0f);
    const auto intermediate = ffn_channels ? ffn_channels : 12u;
    const auto embedding_rows = output_projection ? vocabulary_size : 16u;
    const std::vector<std::pair<std::string, std::vector<uint64_t>>> tensors{
        {"token_embd.weight", {width, embedding_rows}}, {"output_norm.weight", {width}},
        {"blk.0.attn_norm.weight", {width}}, {"blk.0.attn_k.weight", {width, width / 2}},
        {"blk.0.attn_v.weight", {width, width / 2}}, {"blk.0.attn_output.weight", {width, width}},
        {"blk.0.attn_q_norm.weight", {width / 2}}, {"blk.0.attn_k_norm.weight", {width / 2}},
        {"blk.0.ffn_norm.weight", {width}}, {"blk.0.ffn_up.weight", {width, intermediate}},
        {"blk.0.ffn_down.weight", {intermediate, width}},
    };
    for (const auto& [name, shape] : tensors) {
        if (tied_output && name == "token_embd.weight") {
            writer.raw_tensor(name, shape, type, payload);
        } else {
            writer.filled_tensor(name, shape, name.find("norm") != std::string::npos);
        }
    }
    if (output_projection) {
        writer.filled_tensor("blk.0.attn_q.weight", {width, width}, false);
        writer.filled_tensor("blk.0.ffn_gate.weight", {width, intermediate}, false);
        if (!tied_output)
            writer.raw_tensor("output.weight", {width, vocabulary_size}, type, payload);
    } else if (ffn_channels) {
        writer.filled_tensor("blk.0.attn_q.weight", {width, width}, false);
        writer.raw_tensor("blk.0.ffn_gate.weight", {width, ffn_channels}, type, payload);
    } else {
        writer.filled_tensor("blk.0.ffn_gate.weight", {width, intermediate}, false);
        writer.raw_tensor("blk.0.attn_q.weight", {width, width}, type, payload);
    }
    const auto model_path = m_base + (tied_output ? "-tied.gguf" : soft_cap ? "-softcap.gguf" : ".gguf");
    m_paths.push_back(model_path);
    return writer.write(model_path) ? model_path : std::string{};
}

TEST_F(GGUFProjection, QuantizedOutputProjectionFromFile) {
    ov::Core core;
    const auto devices = core.get_available_devices();
    const auto has_vocab_projection = [](const ov::Model& model) {
        auto output = model.get_results().front()->input_value(0);
        if (const auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(output.get_node_shared_ptr()))
            output = reshape->input_value(0);
        const auto head = ov::as_type_ptr<ov::op::v0::MatMul>(output.get_node_shared_ptr());
        if (!head || !head->get_transpose_b() || !head->get_input_partial_shape(1).is_static())
            return false;
        const auto head_shape = head->get_input_shape(1);
        for (const auto& node : model.get_ordered_ops()) {
            const auto gather = ov::is_type<ov::op::v8::Gather>(node) ||
                                ov::is_type<ov::op::internal::GatherCompressed>(node);
            if (gather && node->get_input_partial_shape(0).is_static() && node->get_input_shape(0) == head_shape)
                return true;
        }
        return false;
    };
    const std::pair<const char*, GgufTensorType> formats[] = {{"q4_k", GGUF_TYPE_Q4_K}, {"q5_k", GGUF_TYPE_Q5_K}};
    for (const auto& [format, type] : formats) {
        SCOPED_TRACE(format);
        const auto blocks = load_npy<uint8_t>(std::string(format) + "_qbytes");
        std::vector<uint8_t> payload;
        for (size_t group = 0; group < vocabulary_size / 4; ++group)
            payload.insert(payload.end(), blocks.begin(), blocks.end());
        const auto path = write_projection_model(256, type, payload, 0, true);
        ASSERT_FALSE(path.empty());
        ov::frontend::FrontEndManager manager;
        const auto frontend = manager.load_by_framework("gguf");
        ASSERT_TRUE(frontend);
        const auto load_legacy = [&](const std::string& file) {
            return ov::frontend::gguf::adapt_to_genai(load_legacy_weight_reference(file)).at("language");
        };
        const auto legacy = load_legacy(path);
        const auto native = ov::frontend::gguf::adapt_to_genai(frontend->convert(frontend->load(path))).at("language");
        const auto raw_type = gguf_raw_element_type(type);
        ov::Tensor codes(type == GGUF_TYPE_Q4_K ? ov::element::u4 : ov::element::u8, {vocabulary_size, 8, 32});
        ov::Tensor scales(ov::element::f16, {vocabulary_size, 8, 1});
        ov::Tensor zero_points(ov::element::f16, scales.get_shape());
        const auto unpack = type == GGUF_TYPE_Q4_K ? ov::util::unpack_gguf_q4_k : ov::util::unpack_gguf_q5_k_f16_zp;
        unpack(payload.data(), payload.size(), codes, scales, zero_points);
        auto output = legacy->get_results().front()->input_value(0);
        if (const auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(output.get_node_shared_ptr()))
            output = reshape->input_value(0);
        ASSERT_TRUE(ov::is_type<ov::op::v0::MatMul>(output.get_node_shared_ptr()));
        const auto expanded = output.get_node()->get_input_node_shared_ptr(1);
        const auto weight_graph = std::make_shared<ov::Model>(ov::OutputVector{expanded}, ov::ParameterVector{});
        const auto weight_ops = weight_graph->get_ordered_ops();
        for (const auto& tensor : {codes, scales, zero_points}) {
            EXPECT_TRUE(std::any_of(weight_ops.begin(), weight_ops.end(), [&](const auto& op) {
                const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(op);
                return constant && constant->get_element_type() == tensor.get_element_type() &&
                       constant->get_byte_size() == tensor.get_byte_size() &&
                       std::memcmp(constant->get_data_ptr(), tensor.data(), tensor.get_byte_size()) == 0;
            })) << tensor.get_element_type() << ' ' << tensor.get_shape();
        }
        const auto has_raw_head = [&](const std::shared_ptr<ov::Model>& model) {
            for (const auto& node : model->get_ordered_ops()) {
                const auto fc = ov::as_type_ptr<ov::op::internal::FullyConnectedCompressed>(node);
                if (!fc || fc->get_input_element_type(1) != raw_type)
                    continue;
                                return true;
            }
            return false;
        };
        ASSERT_FALSE(has_raw_head(legacy));
        ASSERT_TRUE(has_raw_head(native));
        const auto softcap_path = write_projection_model(256, type, payload, 0, true, false, true);
        ASSERT_FALSE(softcap_path.empty());
        const auto softcap_legacy = load_legacy(softcap_path);
        const auto softcap_native = ov::frontend::gguf::adapt_to_genai(frontend->convert(
            frontend->load(softcap_path))).at("language");
        ASSERT_TRUE(has_raw_head(softcap_native));
        const auto softcap_ops = softcap_native->get_ordered_ops();
        ASSERT_TRUE(std::any_of(softcap_ops.begin(), softcap_ops.end(), [](const auto& node) {
            return ov::is_type<ov::op::v0::Tanh>(node);
        }));
        const auto softcap_lowered = softcap_native->clone();
        ov::util::lower_gguf_vocab_projection(*softcap_lowered);
        EXPECT_FALSE(has_raw_head(softcap_lowered));
        EXPECT_TRUE(has_vocab_projection(*legacy));
        auto lowered = native->clone();
        ov::util::lower_gguf_vocab_projection(*lowered);
        EXPECT_TRUE(has_vocab_projection(*lowered));
        const auto native_ops = native->get_ordered_ops();
        const auto candidate = std::find_if(native_ops.begin(), native_ops.end(), [&](const auto& node) {
            const auto fc = ov::as_type_ptr<ov::op::internal::FullyConnectedCompressed>(node);
            return fc && fc->get_input_element_type(1) == raw_type;
        });
        ASSERT_NE(candidate, native_ops.end());
        auto fc_inputs = (*candidate)->input_values();
        auto activation = std::make_shared<ov::op::v0::Parameter>(fc_inputs[0].get_element_type(),
                                                                  fc_inputs[0].get_partial_shape());
        fc_inputs[0] = activation;
        const auto ordinary_fc = (*candidate)->clone_with_new_inputs(fc_inputs);
        auto non_vocab = std::make_shared<ov::Model>(ov::OutputVector{ordinary_fc}, ov::ParameterVector{activation});
        ov::util::lower_gguf_vocab_projection(*non_vocab);
        EXPECT_TRUE(has_raw_head(non_vocab));

        std::shared_ptr<ov::Model> tied_legacy;
        std::shared_ptr<ov::Model> tied_native;
        std::shared_ptr<ov::Model> tied_native_gather_reference;
        if (type == GGUF_TYPE_Q4_K) {
            const auto tied_path = write_projection_model(256, type, payload, 0, true, true);
            ASSERT_FALSE(tied_path.empty());
            const auto tied = frontend->convert(frontend->load(tied_path));
            EXPECT_TRUE(has_raw_head(tied));
            auto lowered_tied = tied->clone();
            ov::util::lower_gguf_vocab_projection(*lowered_tied);
            EXPECT_TRUE(has_vocab_projection(*lowered_tied));
            const auto ops = tied->get_ordered_ops();
            EXPECT_TRUE(std::any_of(ops.begin(), ops.end(), [&](const auto& node) {
                return ov::is_type<ov::op::internal::GatherCompressed>(node) && node->get_input_element_type(0) == raw_type;
            }));
            tied_legacy = load_legacy(tied_path);
            tied_native = ov::frontend::gguf::adapt_to_genai(tied).at("language");
            tied_native_gather_reference = tied_native->clone();
            auto head_output = [](const ov::Model& model) {
                auto value = model.get_results().front()->input_value(0);
                if (const auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(value.get_node_shared_ptr()))
                    value = reshape->input_value(0);
                return value.get_node_shared_ptr();
            };
            const auto reference_head = ov::as_type_ptr<ov::op::v0::MatMul>(head_output(*tied_legacy));
            const auto raw_head = ov::as_type_ptr<ov::op::internal::FullyConnectedCompressed>(
                head_output(*tied_native_gather_reference));
            ASSERT_TRUE(reference_head && raw_head);
            ov::replace_node_update_name(raw_head,
                                         std::make_shared<ov::op::v0::MatMul>(raw_head->input_value(0),
                                                                            reference_head->input_value(1),
                                                                            false,
                                                                            true));
        }
        std::vector<std::string> targets{"CPU"};
#ifdef GGUF_GPU_NATIVE_TESTS
        targets.emplace_back("GPU");
#endif
        for (const auto& device : targets) {
            if (std::find(devices.begin(), devices.end(), device) == devices.end())
                continue;
            SCOPED_TRACE(device);
            int64_t tokens[] = {1, 7, 3, 5, 2, 8};
            int64_t mask[] = {1, 1, 1, 1, 1, 1};
            int64_t positions[] = {0, 1, 2, 3, 4, 5};
            int32_t beam[] = {0};
            constexpr size_t steps = 4;
            using KernelHistory = std::vector<std::vector<std::string>>;
            const auto run = [&](const std::shared_ptr<ov::Model>& model, const char* label) {
                auto request = core.compile_model(model, device, ov::hint::inference_precision(ov::element::f32),
                                                  ov::hint::dynamic_quantization_group_size(0),
                                                  ov::enable_profiling(device == "GPU")).create_infer_request();
                request.set_tensor("beam_idx", ov::Tensor(ov::element::i32, {1}, beam));
                ov::Tensor history(ov::element::f32, {1, steps, vocabulary_size});
                KernelHistory kernels(steps);
                for (size_t step = 0; step < steps; ++step) {
                    const size_t count = step == 0 ? 3 : 1;
                    const size_t offset = step == 0 ? 0 : step + 2;
                    request.set_tensor("input_ids", ov::Tensor(ov::element::i64, {1, count}, tokens + offset));
                    request.set_tensor("attention_mask", ov::Tensor(ov::element::i64, {1, offset + count}, mask));
                    request.set_tensor("position_ids", ov::Tensor(ov::element::i64, {1, count}, positions + offset));
                    try {
                        request.infer();
                    } catch (const std::exception& error) {
                        throw std::runtime_error(std::string(format) + ' ' + device + ' ' + label + " step " +
                                                 std::to_string(step) + ": " + error.what());
                    }
                    const auto logits = request.get_tensor("logits");
                    OPENVINO_ASSERT(logits.get_shape() == (ov::Shape{1, 1, vocabulary_size}) &&
                                        logits.get_element_type() == ov::element::f32,
                                    "Expected one full vocabulary row per step");
                    const auto* values = logits.data<const float>();
                    EXPECT_TRUE(std::all_of(values, values + logits.get_size(), [](float value) {
                        return std::isfinite(value);
                    })) << "step " << step;
                    std::copy_n(values, logits.get_size(), history.data<float>() + step * logits.get_size());
                    if (device == "GPU") {
                        for (const auto& entry : request.get_profiling_info()) {
                            if (entry.status == ov::ProfilingInfo::Status::EXECUTED)
                                kernels[step].push_back(entry.node_type + ':' + entry.exec_type);
                        }
                        std::sort(kernels[step].begin(), kernels[step].end());
                        EXPECT_FALSE(kernels[step].empty()) << label << " step " << step;
                    }
                }
                return std::make_pair(std::move(history), std::move(kernels));
            };
            const auto compare = [&](const auto& reference, const auto& candidate, const std::string& variant) {
                SCOPED_TRACE(variant);
                const auto [expected, expected_kernels] = run(reference, "reference");
                const auto [actual, actual_kernels] = run(candidate, "native");
                for (size_t step = 0; step < steps; ++step) {
                    SCOPED_TRACE(step);
                    EXPECT_EQ(actual_kernels[step], expected_kernels[step]);
                    const auto offset = step * vocabulary_size;
                    EXPECT_EQ(std::memcmp(actual.data<const float>() + offset,
                                          expected.data<const float>() + offset,
                                          vocabulary_size * sizeof(float)), 0);
                }
                RecordProperty(variant + '_' + device + "_logits", static_cast<int>(actual.get_size()));
            };
            compare(legacy, native, format);
            compare(softcap_legacy, softcap_native, std::string(format) + "_softcap");
            if (tied_native) {
                compare(device == "CPU" ? tied_legacy : tied_native_gather_reference, tied_native, "q4_k_tied");
            }
        }
    }
}

TEST_F(GGUFShards, NativeQ8ExpertsFromFile) {
    constexpr uint32_t width = 64;
    constexpr uint32_t channels = 2;
    constexpr uint32_t experts = 2;
    const std::string weight_name = "blk.0.ffn_gate_exps.weight";
    GgufWriter writer;
    writer.kv_str("general.architecture", "qwen3moe");
    writer.kv_u32("qwen3moe.block_count", 1);
    writer.kv_u32("qwen3moe.embedding_length", width);
    writer.kv_u32("qwen3moe.attention.head_count", 2);
    writer.kv_u32("qwen3moe.attention.head_count_kv", 1);
    writer.kv_u32("qwen3moe.rope.dimension_count", width / 2);
    writer.kv_u32("qwen3moe.context_length", 32);
    writer.kv_f32("qwen3moe.attention.layer_norm_rms_epsilon", 1e-5f);
    writer.kv_u32("qwen3moe.expert_count", experts);
    writer.kv_u32("qwen3moe.expert_used_count", 1);
    writer.kv_u32("qwen3moe.expert_feed_forward_length", channels);
    const std::vector<std::pair<std::string, std::vector<uint64_t>>> tensors{
        {"token_embd.weight", {width, 16}}, {"output_norm.weight", {width}},
        {"blk.0.attn_norm.weight", {width}}, {"blk.0.attn_q.weight", {width, width}},
        {"blk.0.attn_k.weight", {width, width / 2}}, {"blk.0.attn_v.weight", {width, width / 2}},
        {"blk.0.attn_output.weight", {width, width}}, {"blk.0.attn_q_norm.weight", {width / 2}},
        {"blk.0.attn_k_norm.weight", {width / 2}}, {"blk.0.ffn_norm.weight", {width}},
        {"blk.0.ffn_gate_inp.weight", {width, experts}},
        {"blk.0.ffn_up_exps.weight", {width, channels, experts}},
        {"blk.0.ffn_down_exps.weight", {channels, width, experts}},
    };
    for (const auto& [name, shape] : tensors) {
        writer.filled_tensor(name, shape, name.find("norm") != std::string::npos);
    }
    std::vector<uint8_t> payload(experts * channels * (width / 32) * 34, 0);
    for (size_t block = 0; block < payload.size() / 34; ++block) {
        payload[block * 34 + 1] = block % 2 == 0 ? 0x3c : 0x38;
        payload[block * 34 + 2] = static_cast<uint8_t>(block + 1);
    }
    writer.raw_tensor(weight_name, {width, channels, experts}, GGUF_TYPE_Q8_0, payload);
    const auto model_path = m_base + ".gguf";
    m_paths.push_back(model_path);
    ASSERT_TRUE(writer.write(model_path));

    ov::frontend::FrontEndManager manager;
    const auto frontend = manager.load_by_framework("gguf");
    ASSERT_TRUE(frontend);
    const auto legacy = load_legacy_weight_reference(model_path);
    const auto native = frontend->convert(frontend->load(model_path));
    const auto collect_experts = [](const std::shared_ptr<ov::Model>& model) {
        std::vector<std::shared_ptr<ov::op::internal::GatherMatmul>> result;
        for (const auto& node : model->get_ordered_ops()) {
            if (const auto expert = ov::as_type_ptr<ov::op::internal::GatherMatmul>(node))
                result.push_back(expert);
        }
        return result;
    };
    const auto legacy_ops = collect_experts(legacy);
    const auto native_ops = collect_experts(native);
    ASSERT_EQ(legacy_ops.size(), 3u);
    ASSERT_EQ(native_ops.size(), legacy_ops.size());
    const auto gate = std::find_if(native_ops.begin(), native_ops.end(), [](const auto& expert) {
        return expert->get_input_element_type(1) == ov::element::gguf_q8_0;
    });
    ASSERT_NE(gate, native_ops.end());
    const auto& native_gate = *gate;
    const auto& legacy_gate = legacy_ops[std::distance(native_ops.begin(), gate)];
    EXPECT_FALSE(legacy_gate->get_input_element_type(1).is_gguf_block());
    const auto raw = ov::as_type_ptr<ov::op::v0::Constant>(native_gate->get_input_node_shared_ptr(1));
    ASSERT_TRUE(raw);
    EXPECT_EQ(raw->get_element_type(), ov::element::gguf_q8_0);
    EXPECT_EQ(raw->get_shape(), (ov::Shape{experts, channels, width}));
    ASSERT_EQ(raw->get_byte_size(), payload.size());
    EXPECT_EQ(std::memcmp(raw->get_data_ptr(), payload.data(), payload.size()), 0);

    const auto activations = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, width});
    const auto ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 1});
    const auto isolate = [&](const std::shared_ptr<ov::Node>& expert) {
        const auto output = expert->clone_with_new_inputs({activations, expert->input_value(1), ids, expert->input_value(3)});
        return std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{activations, ids});
    };
    const auto legacy_expert = isolate(legacy_gate);
    const auto native_expert = isolate(native_gate);
    const auto legacy_language = ov::frontend::gguf::adapt_to_genai(legacy->clone()).at("language");
    const auto native_language = ov::frontend::gguf::adapt_to_genai(native->clone()).at("language");
#ifdef GGUF_GPU_ZE_FALLBACK_TESTS
    const auto decompose = [](const std::shared_ptr<ov::Model>& model) {
        ov::util::lower_gguf_q8_0_experts(*model);
        using ov::op::v0::Constant;
        for (const auto& node : model->get_ordered_ops()) {
            const auto expert = ov::as_type_ptr<ov::op::internal::GatherMatmul>(node);
            if (!expert)
                continue;
            const auto order = Constant::create(ov::element::i64, {2}, {1, 0});
            const auto axis = Constant::create(ov::element::i64, {}, {0});
            const auto flatten = Constant::create(ov::element::i64, {1}, {-1});
            const auto& shape = expert->get_input_shape(1);
            const auto channels = Constant::create(ov::element::i64, {1}, {shape[1]});
            const auto width = Constant::create(ov::element::i64, {1}, {shape[2]});
            const auto type = expert->get_input_element_type(0);
            const auto weights = std::make_shared<ov::op::v0::Convert>(expert->input_value(1),
                                                                       type);
            const auto ids_by_slot = std::make_shared<ov::op::v1::Transpose>(expert->input_value(2), order);
            const auto flat_ids = std::make_shared<ov::op::v1::Reshape>(ids_by_slot, flatten, false);
            const auto routing_shape = std::make_shared<ov::op::v3::ShapeOf>(ids_by_slot);
            const auto activation_shape = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{routing_shape, width}, 0);
            const auto activations = std::make_shared<ov::op::v3::Broadcast>(expert->input_value(0), activation_shape);
            const auto flat_input_shape = Constant::create(ov::element::i64, {2}, {int64_t{-1}, static_cast<int64_t>(shape[2])});
            const auto rows = std::make_shared<ov::op::v1::Reshape>(activations, flat_input_shape, false);
            const auto row_count = std::make_shared<ov::op::v3::ShapeOf>(flat_ids);
            const auto result_shape = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{row_count, channels}, 0);
            ov::Output<ov::Node> result = std::make_shared<ov::op::v3::Broadcast>(Constant::create(type, {}, {0}), result_shape);
            for (size_t id = 0; id < shape[0]; ++id) {
                const auto expert_id = Constant::create(flat_ids->get_element_type(), {}, {id});
                const auto mask = std::make_shared<ov::op::v1::Equal>(flat_ids, expert_id);
                const auto indices = std::make_shared<ov::op::v3::NonZero>(mask, ov::element::i32);
                const auto positions = std::make_shared<ov::op::v1::Reshape>(indices, flatten, false);
                const auto selected = std::make_shared<ov::op::v8::Gather>(rows, positions, axis);
                const auto weight = std::make_shared<ov::op::v8::Gather>(weights, expert_id, axis);
                const auto product = std::make_shared<ov::op::v0::MatMul>(selected, weight, false, true);
                result = std::make_shared<ov::op::v3::ScatterUpdate>(result, positions, product, axis);
            }
            const auto output_shape = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{routing_shape, channels}, 0);
            ov::replace_node(expert, std::make_shared<ov::op::v1::Reshape>(result, output_shape, false));
        }
        model->validate_nodes_and_infer_types();
    };
    if (const auto* count = std::getenv("OV_GGUF_ZE_BENCH_EXPERTS")) {
        const auto expert_count = std::stoul(count);
        ASSERT_TRUE(expert_count == 2 || expert_count == 8 || expert_count == 32 || expert_count == 128);
        constexpr size_t input_width = 256;
        constexpr size_t output_width = 128;
        constexpr size_t slots = 2;
        const ov::Shape weight_shape{expert_count, output_width, input_width};
        std::vector<uint8_t> blocks(ov::shape_size(weight_shape) / 32 * 34, 0);
        for (size_t expert = 0; expert < expert_count; ++expert) {
            for (size_t row = 0; row < output_width; ++row) {
                const auto offset = (expert * output_width + row) * (input_width / 32) * 34;
                blocks[offset + 1] = 0x3c;
                blocks[offset + 2] = static_cast<uint8_t>(1 + (expert + row) % 31);
            }
        }
        const auto bench_weights = std::make_shared<ov::op::v0::Constant>(ov::element::gguf_q8_0, weight_shape, blocks.data());
        const auto bench_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{slots, -1, input_width});
        const auto bench_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, slots});
        const auto expert = std::make_shared<ov::op::internal::GatherMatmul>(bench_input, bench_weights, bench_ids);
        const auto model = std::make_shared<ov::Model>(ov::OutputVector{expert}, ov::ParameterVector{bench_input, bench_ids});
        using Clock = std::chrono::steady_clock;
        const auto elapsed = [](Clock::time_point start) {
            return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        };
        ov::Core bench_core;
        auto start = Clock::now();
        decompose(model);
        RecordProperty("decompose_ms", std::to_string(elapsed(start)));
        RecordProperty("graph_ops", static_cast<int>(model->get_ordered_ops().size()));
        start = Clock::now();
        auto compiled = bench_core.compile_model(model, "GPU", ov::hint::inference_precision(ov::element::f16));
        RecordProperty("compile_ms", std::to_string(elapsed(start)));
        RecordProperty("experts", static_cast<int>(expert_count));
        RecordProperty("device", bench_core.get_property("GPU", ov::device::full_name));
        auto request = compiled.create_infer_request();
        for (size_t tokens : {1u, 32u, 128u}) {
            SCOPED_TRACE(tokens);
            ov::Tensor input(ov::element::f32, {slots, tokens, input_width});
            ov::Tensor routes(ov::element::i32, {tokens, slots});
            std::fill_n(input.data<float>(), input.get_size(), 0.f);
            for (size_t token = 0; token < tokens; ++token) {
                for (size_t slot = 0; slot < slots; ++slot) {
                    routes.data<int32_t>()[token * slots + slot] = static_cast<int32_t>((token * slots + slot) % expert_count);
                    input.data<float>()[(slot * tokens + token) * input_width] = static_cast<float>(slot + 1);
                }
            }
            request.set_input_tensor(0, input);
            request.set_input_tensor(1, routes);
            start = Clock::now();
            request.infer();
            RecordProperty("first_ms_t" + std::to_string(tokens), std::to_string(elapsed(start)));
            std::vector<double> samples;
            for (size_t trial = 0; trial < 11; ++trial) {
                start = Clock::now();
                request.infer();
                samples.push_back(elapsed(start));
            }
            std::sort(samples.begin(), samples.end());
            RecordProperty("median_ms_t" + std::to_string(tokens), std::to_string(samples[samples.size() / 2]));
            const auto output = request.get_output_tensor();
            ASSERT_EQ(output.get_shape(), (ov::Shape{slots, tokens, output_width}));
            ASSERT_EQ(output.get_element_type(), ov::element::f32);
            for (size_t slot = 0; slot < slots; ++slot) {
                for (size_t token = 0; token < tokens; ++token) {
                    const auto id = (token * slots + slot) % expert_count;
                    for (size_t row = 0; row < output_width; ++row) {
                        const auto expected = static_cast<float>((slot + 1) * (1 + (id + row) % 31));
                        ASSERT_EQ(output.data<const float>()[(slot * tokens + token) * output_width + row], expected);
                    }
                }
            }
        }
        return;
    }
#endif
    std::vector<float> activation_data(3 * width, 0.f);
    for (size_t token = 0; token < 3; ++token) {
        activation_data[token * width] = static_cast<float>(token + 1);
        activation_data[token * width + 32] = static_cast<float>(token + 1);
    }
    int32_t expert_ids[] = {1, 0, 1};
    const ov::Tensor input(ov::element::f32, {1, 3, width}, activation_data.data());
    const ov::Tensor indices(ov::element::i32, {3, 1}, expert_ids);
    const float oracle[] = {8.f, 11.f, 4.f, 10.f, 24.f, 33.f};
    ov::Core core;
    const auto available = core.get_available_devices();
    std::vector<std::pair<std::string, ov::element::Type>> configurations{{"CPU", ov::element::f32}};
#if defined(GGUF_GPU_NATIVE_TESTS) || defined(GGUF_GPU_ZE_FALLBACK_TESTS)
    configurations.emplace_back("GPU", ov::element::f16);
#endif
    for (const auto& [device, precision] : configurations) {
        if (std::find(available.begin(), available.end(), device) == available.end())
            continue;
        SCOPED_TRACE(device);
        const auto compile = [&](const std::shared_ptr<ov::Model>& model) {
#ifdef GGUF_GPU_ZE_FALLBACK_TESTS
            if (device == "GPU") {
                const auto decomposed = model->clone();
                decompose(decomposed);
                return core.compile_model(decomposed, device, ov::hint::inference_precision(precision));
            }
#endif
            return core.compile_model(model, device, ov::hint::inference_precision(precision));
        };
        const auto run = [&](const std::shared_ptr<ov::Model>& model) {
            auto request = compile(model).create_infer_request();
            request.set_input_tensor(0, input);
            request.set_input_tensor(1, indices);
            request.infer();
            return request.get_output_tensor();
        };
        const auto expected = run(legacy_expert);
        const auto actual = run(native_expert);
        ASSERT_EQ(expected.get_shape(), (ov::Shape{1, 3, channels}));
        ASSERT_EQ(expected.get_element_type(), ov::element::f32);
        EXPECT_EQ(std::memcmp(expected.data(), oracle, sizeof(oracle)), 0);
        ASSERT_EQ(actual.get_shape(), expected.get_shape());
        ASSERT_EQ(actual.get_element_type(), expected.get_element_type());
        EXPECT_EQ(std::memcmp(actual.data(), expected.data(), expected.get_byte_size()), 0);
        RecordProperty(device, core.get_property(device, ov::device::full_name));

#ifdef GGUF_GPU_ZE_FALLBACK_TESTS
        if (device == "GPU") {
            constexpr size_t token_count = 128;
            constexpr size_t slots = 2;
            const auto slot_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                                           ov::PartialShape{slots, -1, width});
            const auto slot_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i32,
                                                                         ov::PartialShape{-1, slots});
            ov::Tensor inputs(ov::element::f32, {slots, token_count, width});
            ov::Tensor routes(ov::element::i32, {token_count, slots});
            std::fill_n(inputs.data<float>(), inputs.get_size(), 0.f);
            for (size_t token = 0; token < token_count; ++token) {
                for (size_t slot = 0; slot < slots; ++slot) {
                    routes.data<int32_t>()[token * slots + slot] = static_cast<int32_t>((token / 2 + slot) % experts);
                    const auto offset = (slot * token_count + token) * width;
                    inputs.data<float>()[offset] = static_cast<float>(slot + 1);
                    inputs.data<float>()[offset + 32] = 1.f;
                }
            }
            for (const auto& gate : {legacy_gate, native_gate}) {
                const auto output = gate->clone_with_new_inputs({slot_input, gate->input_value(1), slot_ids, gate->input_value(3)});
                const auto model = std::make_shared<ov::Model>(ov::OutputVector{output},
                                                               ov::ParameterVector{slot_input, slot_ids});
                auto request = compile(model).create_infer_request();
                request.set_input_tensor(0, inputs);
                request.set_input_tensor(1, routes);
                request.infer();
                const auto rows = request.get_output_tensor();
                ASSERT_EQ(rows.get_shape(), (ov::Shape{slots, token_count, channels}));
                ASSERT_EQ(rows.get_element_type(), ov::element::f32);
                for (size_t slot = 0; slot < slots; ++slot) {
                    for (size_t token = 0; token < token_count; ++token) {
                        for (size_t row = 0; row < channels; ++row) {
                            const auto expert = (token / 2 + slot) % experts;
                            const auto first_code = 2 * (expert * channels + row) + 1;
                            const auto reference = static_cast<float>((slot + 1) * first_code) +
                                                   0.5f * static_cast<float>(first_code + 1);
                            EXPECT_EQ(rows.data<const float>()[(slot * token_count + token) * channels + row], reference);
                        }
                    }
                }
            }
            RecordProperty("ZE_grouped_expert_tokens", static_cast<int>(token_count));
        }
#endif
        int64_t tokens[] = {1, 7, 3};
        int64_t mask[] = {1, 1, 1};
        int64_t positions[] = {0, 1, 2};
        int32_t beam[] = {0};
        const auto run_model = [&](ov::CompiledModel& model, size_t count) {
            auto request = model.create_infer_request();
            request.set_tensor("input_ids", ov::Tensor(ov::element::i64, {1, count}, tokens));
            request.set_tensor("attention_mask", ov::Tensor(ov::element::i64, {1, count}, mask));
            request.set_tensor("position_ids", ov::Tensor(ov::element::i64, {1, count}, positions));
            request.set_tensor("beam_idx", ov::Tensor(ov::element::i32, {1}, beam));
            request.infer();
            return request.get_tensor("logits");
        };
        SCOPED_TRACE("full model logits");
        auto legacy_compiled = compile(legacy_language);
        auto native_compiled = compile(native_language);
        for (size_t count : {1u, 2u, 3u}) {
            SCOPED_TRACE(count);
            const auto legacy_logits = run_model(legacy_compiled, count);
            const auto native_logits = run_model(native_compiled, count);
            ASSERT_EQ(legacy_logits.get_shape(), (ov::Shape{1, 1, 16}));
            ASSERT_EQ(legacy_logits.get_element_type(), ov::element::f32);
            ASSERT_EQ(native_logits.get_shape(), legacy_logits.get_shape());
            ASSERT_EQ(native_logits.get_element_type(), legacy_logits.get_element_type());
            const auto* values = legacy_logits.data<const float>();
            EXPECT_TRUE(std::all_of(values, values + legacy_logits.get_size(), [](float value) {
                return std::isfinite(value);
            }));
            EXPECT_EQ(std::memcmp(native_logits.data(), legacy_logits.data(), legacy_logits.get_byte_size()), 0);
        }
        RecordProperty(device + "_full_model", 48);
    }
    EXPECT_EQ(native_gate->get_input_node_shared_ptr(1), raw);
    EXPECT_EQ(std::memcmp(raw->get_data_ptr(), payload.data(), payload.size()), 0);
}

class GGUFQ8Projection : public GGUFProjection {
protected:
    std::string write_projection_model();
};

std::string GGUFQ8Projection::write_projection_model() {
    std::vector<uint8_t> payload(32 * 34, 1);
    for (size_t block = 0; block < 32; ++block) {
        payload[block * 34] = static_cast<uint8_t>(block);
        payload[block * 34 + 1] = 0x3c;
        payload[block * 34 + 2] = static_cast<uint8_t>(240 + block);
    }
    return GGUFProjection::write_projection_model(32, GGUF_TYPE_Q8_0, payload);
}

TEST_F(GGUFQ8Projection, FileLoadPreservesNativeWeights) {
    const auto model_path = write_projection_model();
    ASSERT_FALSE(model_path.empty());
    std::vector<uint8_t> payload(32 * 34, 1);
    for (size_t block = 0; block < 32; ++block) {
        payload[block * 34] = static_cast<uint8_t>(block);
        payload[block * 34 + 1] = 0x3c;
        payload[block * 34 + 2] = static_cast<uint8_t>(240 + block);
    }
    FrontEnd frontend;
    const auto model = frontend.convert(frontend.load(model_path));
    size_t raw_constants = 0;
    for (const auto& node : model->get_ordered_ops()) {
        if (node->get_output_size() && node->get_output_element_type(0).is_gguf_block()) {
            const auto weight = ov::as_type_ptr<ov::op::v0::Constant>(node);
            ASSERT_NE(weight, nullptr);
            EXPECT_EQ(weight->get_element_type(), ov::element::gguf_q8_0);
            EXPECT_EQ(weight->get_shape(), (ov::Shape{32, 32}));
            ASSERT_EQ(weight->get_byte_size(), payload.size());
            EXPECT_EQ(std::memcmp(weight->get_data_ptr(), payload.data(), payload.size()), 0);
            const auto consumers = weight->output(0).get_target_inputs();
            ASSERT_EQ(consumers.size(), 1);
            ASSERT_TRUE(ov::is_type<ov::op::internal::FullyConnectedCompressed>(consumers.begin()->get_node()));
            EXPECT_EQ(consumers.begin()->get_index(), 1);
            ++raw_constants;
        }
    }
    EXPECT_EQ(raw_constants, 1u);
    EXPECT_THROW(frontend.load(model_path, ov::AnyMap{{"UNKNOWN_GGUF_PROPERTY", true}}), ov::Exception);
}

TEST_F(GGUFQ8Projection, NativeFCMatchesCompatibleCPU) {
    ov::Core core;
    const auto devices = core.get_available_devices();
    if (std::find(devices.begin(), devices.end(), "CPU") == devices.end()) {
        GTEST_SKIP() << "CPU device unavailable";
    }
    const auto model_path = write_projection_model();
    ASSERT_FALSE(model_path.empty());
    FrontEnd frontend;
    const auto legacy = load_legacy_weight_reference(model_path);
    const auto native = frontend.convert(frontend.load(model_path));
    const auto constants = [](const std::shared_ptr<ov::Model>& model) {
        std::vector<std::shared_ptr<ov::op::v0::Constant>> result;
        for (const auto& node : model->get_ordered_ops()) {
            const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
            if (constant && ((constant->get_element_type() == ov::element::i8 &&
                              constant->get_shape() == ov::Shape{32, 1, 32}) ||
                             (constant->get_element_type() == ov::element::f16 &&
                              constant->get_shape() == ov::Shape{32, 1, 1}))) {
                result.push_back(constant);
            }
        }
        return result;
    };
    std::shared_ptr<ov::op::internal::FullyConnectedCompressed> fc;
    for (const auto& node : native->get_ordered_ops()) {
        const auto candidate = ov::as_type_ptr<ov::op::internal::FullyConnectedCompressed>(node);
        if (candidate && candidate->get_input_element_type(1) == ov::element::gguf_q8_0) {
            fc = candidate;
        }
    }
    ASSERT_NE(fc, nullptr);
    const auto compatible = ov::util::make_gguf_q8_0_fc_compatible(*fc);
    ASSERT_NE(compatible, nullptr);
    const auto actual = constants(std::make_shared<ov::Model>(ov::OutputVector{compatible}, native->get_parameters()));
    const auto expected = constants(legacy);
    ASSERT_EQ(actual.size(), 2);
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t index = 0; index < actual.size(); ++index) {
        EXPECT_EQ(actual[index]->get_element_type(), expected[index]->get_element_type());
        EXPECT_EQ(actual[index]->get_shape(), expected[index]->get_shape());
        ASSERT_EQ(actual[index]->get_byte_size(), expected[index]->get_byte_size());
        EXPECT_EQ(std::memcmp(actual[index]->get_data_ptr(), expected[index]->get_data_ptr(), actual[index]->get_byte_size()), 0);
    }
    auto reference = core.compile_model(legacy, "CPU", ov::hint::inference_precision(ov::element::f32));
    auto compiled = core.compile_model(native, "CPU", ov::hint::inference_precision(ov::element::f32));
    auto reference_request = reference.create_infer_request();
    auto native_request = compiled.create_infer_request();
    for (const auto& port : reference.inputs()) {
        const auto shape = port.get_partial_shape();
        ASSERT_TRUE(shape.rank().is_static());
        auto concrete = shape.get_min_shape();
        for (size_t dim = 0; dim < concrete.size(); ++dim) {
            if (shape[dim].is_dynamic() && concrete[dim] == 0) {
                concrete[dim] = 1;
            }
        }
        ov::Tensor input(port.get_element_type(), concrete);
        std::memset(input.data(), 0, input.get_byte_size());
        reference_request.set_tensor(port.get_any_name(), input);
        native_request.set_tensor(port.get_any_name(), input);
    }
    reference_request.infer();
    native_request.infer();
    ASSERT_EQ(compiled.outputs().size(), reference.outputs().size());
    for (size_t port = 0; port < reference.outputs().size(); ++port) {
        SCOPED_TRACE(port);
        const auto reference_output = reference_request.get_output_tensor(port);
        const auto native_output = native_request.get_output_tensor(port);
        ASSERT_EQ(native_output.get_shape(), reference_output.get_shape());
        const auto type = native_output.get_element_type();
        ASSERT_EQ(type, reference_output.get_element_type());
        ASSERT_TRUE(type == ov::element::f32 || type == ov::element::f16);
        const auto value_at = [type](const ov::Tensor& tensor, size_t index) {
            return type == ov::element::f32 ? tensor.data<const float>()[index]
                                            : static_cast<float>(tensor.data<const ov::float16>()[index]);
        };
        for (size_t index = 0; index < native_output.get_size(); ++index) {
            EXPECT_NEAR(value_at(native_output, index), value_at(reference_output, index), 1e-5f);
        }
    }
    const auto nodes = native->get_ordered_ops();
    EXPECT_NE(std::find(nodes.begin(), nodes.end(), fc), nodes.end());
    EXPECT_EQ(fc->get_input_element_type(1), ov::element::gguf_q8_0);
}

TEST_F(GGUFQ8Projection, SharedModelCompilesInEitherDeviceOrder) {
    ov::Core core;
    const auto devices = core.get_available_devices();
    if (std::find(devices.begin(), devices.end(), "CPU") == devices.end() ||
        std::find(devices.begin(), devices.end(), "GPU") == devices.end()) {
        GTEST_SKIP() << "CPU and GPU devices are required";
    }
    const auto path = write_projection_model();
    ASSERT_FALSE(path.empty());
    FrontEnd frontend;
    const auto native = frontend.convert(frontend.load(path));
    std::shared_ptr<ov::op::internal::FullyConnectedCompressed> projection;
    for (const auto& node : native->get_ordered_ops()) {
        const auto candidate = ov::as_type_ptr<ov::op::internal::FullyConnectedCompressed>(node);
        if (candidate && candidate->get_input_element_type(1) == ov::element::gguf_q8_0) {
            projection = candidate;
            break;
        }
    }
    ASSERT_NE(projection, nullptr);
    const auto weights = ov::as_type_ptr<ov::op::v0::Constant>(projection->get_input_node_shared_ptr(1));
    ASSERT_NE(weights, nullptr);
    const auto raw = static_cast<const uint8_t*>(weights->get_data_ptr());
    const std::vector<uint8_t> snapshot(raw, raw + weights->get_byte_size());
    const auto activation = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 32});
    auto inputs = projection->input_values();
    inputs[0] = activation;
    const auto fc = projection->clone_with_new_inputs(inputs);
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{activation});

    auto compatible = model->clone();
    ov::util::lower_gguf_q8_0_fc(*compatible);
    EXPECT_TRUE(ov::is_type<ov::op::v0::MatMul>(compatible->get_results()[0]->get_input_node_ptr(0)));

    for (const auto& order : {std::array{"CPU", "GPU"}, std::array{"GPU", "CPU"}}) {
        for (const auto* device : order) {
            SCOPED_TRACE(device);
            auto compiled = core.compile_model(model, device, ov::hint::inference_precision(ov::element::f32),
                                               ov::enable_profiling(true));
            auto request = compiled.create_infer_request();
            ov::Tensor input(ov::element::f32, {1, 32});
            std::fill_n(input.data<float>(), input.get_size(), 1.f);
            request.set_input_tensor(input);
            request.infer();
            const auto output = request.get_output_tensor();
            ASSERT_EQ(output.get_shape(), (ov::Shape{1, 32}));
            ASSERT_EQ(output.get_element_type(), ov::element::f32);
            for (size_t channel = 0; channel < 32; ++channel) {
                const auto scale = static_cast<float>(ov::float16::from_bits(static_cast<uint16_t>(0x3c00 + channel)));
                EXPECT_NEAR(output.data<const float>()[channel], (15.f + static_cast<float>(channel)) * scale, 1e-4f);
            }
#if defined(GGUF_GPU_NATIVE_TESTS) || defined(GGUF_GPU_RAW_OCL_TESTS)
            if (std::string(device) == "GPU") {
                const auto ops = compiled.get_runtime_model()->get_ordered_ops();
                std::vector<std::string> native_kernels;
                for (const auto& op : ops) {
                    const auto& info = op->get_rt_info();
                    const auto precision = info.find("weights_precision");
                    const auto kernels = info.find("gguf_last_executed_kernels");
                    if (kernels != info.end()) {
                        ASSERT_NE(precision, info.end());
                        EXPECT_EQ(precision->second.as<std::string>(), ov::element::gguf_q8_0.get_type_name());
                        native_kernels.push_back(kernels->second.as<std::string>());
                    }
                }
                ASSERT_EQ(native_kernels.size(), 1u);
                EXPECT_FALSE(native_kernels.front().empty());
#ifdef GGUF_GPU_RAW_OCL_TESTS
                EXPECT_EQ(native_kernels.front().find("fc_gguf_opt"), 0u);
#endif
                RecordProperty("gpu_native_kernels", native_kernels.front());
            }
#endif
            EXPECT_EQ(model->get_results()[0]->get_input_node_shared_ptr(0), fc);
            ASSERT_EQ(fc->get_input_node_shared_ptr(1), weights);
            EXPECT_EQ(std::memcmp(weights->get_data_ptr(), snapshot.data(), snapshot.size()), 0);
        }
    }
}

#if defined(GGUF_GPU_NATIVE_TESTS) || defined(GGUF_GPU_RAW_OCL_TESTS) || defined(GGUF_GPU_ZE_FALLBACK_TESTS)
static std::shared_ptr<ov::Model> round_trip_dense_projection(ov::Core& core,
                                                              const std::shared_ptr<ov::Model>& model,
                                                              const std::filesystem::path& gguf_path) {
    auto xml = gguf_path;
    auto binary = gguf_path;
    xml.replace_extension(".xml");
    binary.replace_extension(".bin");
    ov::pass::Serialize(xml, binary).run_on_model(model);
    const auto restored = core.read_model(xml);
    std::filesystem::remove(xml);
    std::filesystem::remove(binary);
    const auto original_fc = model->get_results().at(0)->get_input_node_shared_ptr(0);
    const auto restored_fc = ov::as_type_ptr<ov::op::internal::FullyConnectedCompressed>(
        restored->get_results().at(0)->get_input_node_shared_ptr(0));
    OPENVINO_ASSERT(restored_fc, "IR did not preserve FullyConnectedCompressed");
    const auto source = ov::as_type_ptr<ov::op::v0::Constant>(original_fc->get_input_node_shared_ptr(1));
    const auto weights = ov::as_type_ptr<ov::op::v0::Constant>(restored_fc->get_input_node_shared_ptr(1));
    OPENVINO_ASSERT(source && weights && source->get_element_type() == weights->get_element_type() &&
                        source->get_shape() == weights->get_shape() && source->get_byte_size() == weights->get_byte_size(),
                    "IR did not preserve native weights metadata");
    OPENVINO_ASSERT(std::memcmp(source->get_data_ptr(), weights->get_data_ptr(), source->get_byte_size()) == 0,
                    "IR did not preserve native weight bytes");
    return restored;
}
#endif

#if defined(GGUF_GPU_NATIVE_TESTS) || defined(GGUF_GPU_RAW_OCL_TESTS)
enum class DenseLayout { Raw, Shuffled, RawUnaligned, ShuffledGemmUnaligned };

struct DenseDispatchCase {
    GgufTensorType type;
    const char* fixture;
    const char* kernel_suffix;
    size_t bytes_per_row;
    DenseLayout layout;
};

class GGUFDenseDispatch : public GGUFProjection, public testing::WithParamInterface<DenseDispatchCase> {};

TEST_P(GGUFDenseDispatch, FileExecutesAtMultipleRowCounts) {
    const auto& test_case = GetParam();
    const auto layout_case = test_case.layout;
    const std::string fixture = test_case.fixture;
    const std::string weight_type = "gguf_" + fixture;
    const std::string kernel_suffix = test_case.kernel_suffix;
    const bool shuffled = layout_case == DenseLayout::Shuffled || layout_case == DenseLayout::ShuffledGemmUnaligned;
    const bool gemm_aligned = layout_case == DenseLayout::Shuffled;
    GgufScopedEnvironment shuffle("OV_GPU_GGUF_SHUFFLE", layout_case == DenseLayout::Raw ? "0" : "1");
    GgufScopedEnvironment threshold("OV_GPU_GGUF_PREFILL_THRESHOLD", "32");
    GgufScopedEnvironment prefill("OV_GPU_GGUF_PREFILL_GEMM", "ocl");
    GgufScopedEnvironment prefill_f16("OV_GPU_GGUF_PREFILL_F16", "1");
    GgufScopedEnvironment prefill_dq("OV_GPU_GGUF_PREFILL_DQ_ACT", "0");
    const auto* strict_env = std::getenv("OV_GPU_GGUF_PREFILL_GEMM_STRICT");
    const bool require_gemm = gemm_aligned && strict_env && std::string(strict_env) == "1";
    GgufScopedEnvironment strict("OV_GPU_GGUF_PREFILL_GEMM_STRICT", require_gemm ? "1" : "0");
    ov::Core core;
    const auto devices = core.get_available_devices();
    if (std::find(devices.begin(), devices.end(), "GPU") == devices.end()) {
        GTEST_SKIP() << "GPU device unavailable";
    }
    constexpr size_t width = 256;
    const uint32_t ffn_channels = layout_case == DenseLayout::RawUnaligned            ? 52
                                  : layout_case == DenseLayout::ShuffledGemmUnaligned ? 48
                                                                                      : 0;
    const auto channels = ffn_channels ? ffn_channels : width;
    RecordProperty("device", core.get_property("GPU", ov::device::full_name));
    const auto blocks = load_npy<uint8_t>(fixture + "_qbytes");
    const auto decoded = load_npy<float>(fixture + "_deq");
    ASSERT_EQ(blocks.size(), 4 * test_case.bytes_per_row);
    ASSERT_EQ(decoded.size(), 4 * width);
    std::vector<uint8_t> payload;
    for (size_t group = 0; group < channels / 4; ++group) {
        payload.insert(payload.end(), blocks.begin(), blocks.end());
    }
    const auto path = write_projection_model(width, test_case.type, payload, ffn_channels);
    ASSERT_FALSE(path.empty());
    const auto native = [&] {
        FrontEnd frontend;
        return frontend.convert(frontend.load(path));
    }();
    std::shared_ptr<ov::op::internal::FullyConnectedCompressed> projection;
    for (const auto& node : native->get_ordered_ops()) {
        const auto candidate = ov::as_type_ptr<ov::op::internal::FullyConnectedCompressed>(node);
        if (candidate && candidate->get_input_element_type(1).get_type_name() == weight_type) {
            projection = candidate;
            break;
        }
    }
    ASSERT_TRUE(projection);
    const auto weights = ov::as_type_ptr<ov::op::v0::Constant>(projection->get_input_node_shared_ptr(1));
    ASSERT_TRUE(weights);
    ASSERT_EQ(weights->get_byte_size(), payload.size());
    ASSERT_EQ(std::memcmp(weights->get_data_ptr(), payload.data(), payload.size()), 0);
    const auto activation = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, width});
    auto inputs = projection->input_values();
    inputs[0] = activation;
    const auto fc = projection->clone_with_new_inputs(inputs);
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{activation});
    const auto restored = layout_case == DenseLayout::Raw ? round_trip_dense_projection(core, model, path) : nullptr;
    if (layout_case == DenseLayout::Raw && std::find(devices.begin(), devices.end(), "CPU") != devices.end()) {
        const auto precision = ov::hint::inference_precision(ov::element::f32);
        bool cpu_supports_fc = false;
        try {
            const auto supported = core.query_model(model, "CPU", precision);
            cpu_supports_fc = supported.count(fc->get_friendly_name()) != 0;
            RecordProperty("cpu_query", cpu_supports_fc ? "supported" : "not_listed");
        } catch (const ov::Exception& error) {
            ASSERT_NE(std::string(error.what()).find("CPU plugin does not support " + weight_type), std::string::npos)
                << error.what();
            RecordProperty("cpu_query", "unsupported_weight_type");
        }
        RecordProperty("cpu_device", core.get_property("CPU", ov::device::full_name));
        ov::CompiledModel cpu;
        bool cpu_compiles = false;
        try {
            cpu = core.compile_model(model, "CPU", precision);
            cpu_compiles = true;
            RecordProperty("cpu_compile", "supported");
        } catch (const ov::Exception& error) {
            ASSERT_NE(std::string(error.what()).find("CPU plugin does not support " + weight_type), std::string::npos)
                << error.what();
            RecordProperty("cpu_compile", "unsupported_weight_type");
        }
        if (test_case.type == GGUF_TYPE_Q8_0 || test_case.type == GGUF_TYPE_Q4_0 || test_case.type == GGUF_TYPE_Q4_1 ||
            test_case.type == GGUF_TYPE_Q6_K || test_case.type == GGUF_TYPE_Q4_K || test_case.type == GGUF_TYPE_Q5_K) {
            EXPECT_TRUE(cpu_supports_fc);
            EXPECT_TRUE(cpu_compiles);
        }
        if (cpu_compiles) {
            EXPECT_TRUE(cpu_supports_fc);
            auto cpu_request = cpu.create_infer_request();
            ov::Tensor input(ov::element::f32, {2, width});
            std::fill_n(input.data<float>(), input.get_size(), 0.f);
            input.data<float>()[13] = 1.f;
            input.data<float>()[width + 20] = 1.f;
            cpu_request.set_input_tensor(input);
            cpu_request.infer();
            const auto output = cpu_request.get_output_tensor();
            ASSERT_EQ(output.get_shape(), (ov::Shape{2, channels}));
            ASSERT_EQ(output.get_element_type(), ov::element::f32);
            auto restored_cpu_request = core.compile_model(restored, "CPU", precision).create_infer_request();
            restored_cpu_request.set_input_tensor(input);
            restored_cpu_request.infer();
            const auto restored_output = restored_cpu_request.get_output_tensor();
            ASSERT_EQ(restored_output.get_shape(), output.get_shape());
            ASSERT_EQ(restored_output.get_element_type(), output.get_element_type());
            EXPECT_EQ(std::memcmp(restored_output.data(), output.data(), output.get_byte_size()), 0);
            if (test_case.type == GGUF_TYPE_Q4_1 || test_case.type == GGUF_TYPE_Q6_K ||
                test_case.type == GGUF_TYPE_Q4_K || test_case.type == GGUF_TYPE_Q5_K) {
                const bool q6_k = test_case.type == GGUF_TYPE_Q6_K;
                const bool q5_k = test_case.type == GGUF_TYPE_Q5_K;
                GgufTensor source{};
                source.type = test_case.type;
                source.ndim = 2;
                source.dim[0] = width;
                source.dim[1] = channels;
                source.num_weights = channels * width;
                source.bsize = payload.size();
                source.weights_data = payload.data();
                WeightTensors tensors;
                tensors.weight = ov::Tensor(q6_k || q5_k ? ov::element::i8 : ov::element::u32,
                                            {channels, q6_k || q5_k ? width : width / 8});
                tensors.scales = ov::Tensor(ov::element::f16, {channels, width / (q6_k ? 16 : 32)});
                if (q6_k) {
                    gguf_fill_sym(source, tensors.weight, tensors.scales);
                } else if (q5_k) {
                    gguf_fill_q5_k_i8(source, tensors.weight, tensors.scales);
                } else {
                    tensors.zero_point = ov::Tensor(ov::element::u8, {channels, width / 32});
                    gguf_fill_asym(source, tensors.weight, tensors.scales, tensors.zero_point);
                }
                const auto legacy_weights = make_weight_node(tensors, test_case.type, "blk.0.ffn_up.weight");
                const auto converted = std::make_shared<ov::op::v0::Convert>(legacy_weights, ov::element::f32);
                const auto legacy_fc = std::make_shared<ov::op::v0::MatMul>(activation, converted, false, true);
                const auto legacy_model = std::make_shared<ov::Model>(ov::OutputVector{legacy_fc},
                                                                      ov::ParameterVector{activation});
                auto legacy_request = core.compile_model(legacy_model, "CPU", precision).create_infer_request();
                legacy_request.set_input_tensor(input);
                legacy_request.infer();
                const auto reference = legacy_request.get_output_tensor();
                ASSERT_EQ(reference.get_shape(), output.get_shape());
                for (size_t index = 0; index < output.get_size(); ++index) {
                    EXPECT_NEAR(output.data<const float>()[index], reference.data<const float>()[index], 1e-4f);
                }
                const auto native_fc = ov::as_type_ptr<ov::op::internal::FullyConnectedCompressed>(fc);
                ASSERT_TRUE(native_fc);
                const auto lowered = ov::util::make_gguf_fc_compatible(
                    *native_fc, {weights->get_element_type()});
                ASSERT_TRUE(lowered);
                const auto lowered_model = std::make_shared<ov::Model>(ov::OutputVector{lowered},
                                                                       ov::ParameterVector{activation});
                auto gpu_lowered = core.compile_model(lowered_model, "GPU", precision).create_infer_request();
                auto gpu_legacy = core.compile_model(legacy_model, "GPU", precision).create_infer_request();
                gpu_lowered.set_input_tensor(input);
                gpu_legacy.set_input_tensor(input);
                gpu_lowered.infer();
                gpu_legacy.infer();
                const auto gpu_output = gpu_lowered.get_output_tensor();
                const auto gpu_reference = gpu_legacy.get_output_tensor();
                ASSERT_EQ(gpu_output.get_shape(), output.get_shape());
                ASSERT_EQ(gpu_reference.get_shape(), output.get_shape());
                for (size_t index = 0; index < gpu_output.get_size(); ++index) {
                    EXPECT_NEAR(gpu_output.data<const float>()[index], gpu_reference.data<const float>()[index], 1e-4f);
                }
            } else {
                for (size_t channel = 0; channel < channels; ++channel) {
                    EXPECT_NEAR(output.data<const float>()[channel], decoded[(channel % 4) * width + 13], 1e-4f);
                    EXPECT_NEAR(output.data<const float>()[channels + channel],
                                decoded[(channel % 4) * width + 20], 1e-4f);
                }
            }
        }
        ASSERT_EQ(model->get_results()[0]->get_input_node_shared_ptr(0), fc);
        ASSERT_EQ(fc->get_input_node_shared_ptr(1), weights);
        EXPECT_EQ(std::memcmp(weights->get_data_ptr(), payload.data(), payload.size()), 0);
    }
    auto compiled = core.compile_model(model, "GPU", ov::hint::inference_precision(ov::element::f32),
                                       ov::enable_profiling(true));
    std::stringstream blob;
    compiled.export_model(blob);
    auto imported = [&] {
        GgufScopedEnvironment opposite_shuffle("OV_GPU_GGUF_SHUFFLE", layout_case == DenseLayout::Raw ? "1" : "0");
        return core.import_model(blob, "GPU", ov::hint::inference_precision(ov::element::f32), ov::enable_profiling(true));
    }();
    const auto executed_kernels = [&](const ov::CompiledModel& gpu_model) {
        std::vector<std::string> executed;
        for (const auto& op : gpu_model.get_runtime_model()->get_ordered_ops()) {
            const auto& info = op->get_rt_info();
            const auto kernels = info.find("gguf_last_executed_kernels");
            if (kernels != info.end()) {
                EXPECT_EQ(info.at("weights_precision").as<std::string>(), weight_type);
                executed.push_back(kernels->second.as<std::string>());
            }
        }
        return executed;
    };
    auto request = compiled.create_infer_request();
    auto imported_request = imported.create_infer_request();
    ov::CompiledModel restored_gpu;
    ov::InferRequest restored_request;
    if (restored) {
        restored_gpu = core.compile_model(restored, "GPU", ov::hint::inference_precision(ov::element::f32),
                                          ov::enable_profiling(true));
        restored_request = restored_gpu.create_infer_request();
    }
    for (size_t rows : {1u, 33u, 128u, 257u, 1u}) {
        SCOPED_TRACE(rows);
        ov::Tensor input(ov::element::f32, {rows, width});
        std::fill_n(input.data<float>(), input.get_size(), 0.f);
        for (size_t row = 0; row < rows; ++row) {
            input.data<float>()[row * width + (7 * row + 13) % width] = 1.f;
        }
        request.set_input_tensor(input);
        request.infer();
        const auto output = request.get_output_tensor();
        ASSERT_EQ(output.get_shape(), (ov::Shape{rows, channels}));
        ASSERT_EQ(output.get_element_type(), ov::element::f32);
        const auto executed = executed_kernels(compiled);
        ASSERT_EQ(executed.size(), 1u);
        if (restored) {
            restored_request.set_input_tensor(input);
            restored_request.infer();
            const auto restored_output = restored_request.get_output_tensor();
            ASSERT_EQ(restored_output.get_shape(), output.get_shape());
            ASSERT_EQ(restored_output.get_element_type(), output.get_element_type());
            EXPECT_EQ(std::memcmp(restored_output.data(), output.data(), output.get_byte_size()), 0);
            EXPECT_EQ(executed_kernels(restored_gpu), executed);
        }
        imported_request.set_input_tensor(input);
        imported_request.infer();
        const auto imported_output = imported_request.get_output_tensor();
        ASSERT_EQ(imported_output.get_shape(), output.get_shape());
        ASSERT_EQ(imported_output.get_element_type(), output.get_element_type());
        EXPECT_EQ(std::memcmp(imported_output.data<const float>(), output.data<const float>(), output.get_byte_size()), 0);
        EXPECT_EQ(executed_kernels(imported), executed);
        const auto& kernel = executed.front();
        const bool gemm = kernel.find("fc_gguf_gemm_" + kernel_suffix) == 0;
        const bool sg = kernel.find("fc_gguf_" + kernel_suffix + "_sg") == 0;
        EXPECT_FALSE(gemm && !gemm_aligned) << kernel;
        const bool transcode = kernel.find("fc_gguf_transcode_" + weight_type + "_") == 0;
    #ifdef GGUF_GPU_NATIVE_TESTS
        if (rows > 32 && !gemm) {
            EXPECT_FALSE(require_gemm);
            EXPECT_TRUE(transcode) << kernel;
        } else
    #endif
        if (!shuffled) {
            EXPECT_EQ(kernel.find("fc_gguf_opt_" + weight_type + "_"), 0u);
        } else if (rows == 1) {
            EXPECT_TRUE(sg) << kernel;
        } else if (require_gemm) {
            EXPECT_TRUE(gemm) << kernel;
        } else {
            EXPECT_TRUE(gemm || sg) << kernel;
        }
        for (size_t row = 0; row < rows; ++row) {
            for (size_t channel = 0; channel < channels; ++channel) {
                const auto value = decoded[(channel % 4) * width + (7 * row + 13) % width];
                const auto expected = gemm || transcode ? static_cast<float>(ov::float16(value)) : value;
                EXPECT_NEAR(output.data<const float>()[row * channels + channel], expected, 1e-4f);
            }
        }
        RecordProperty("kernel_m" + std::to_string(rows), executed.front());
        EXPECT_EQ(fc->get_input_node_shared_ptr(1), weights);
        EXPECT_EQ(std::memcmp(weights->get_data_ptr(), payload.data(), payload.size()), 0);
    }
}

INSTANTIATE_TEST_SUITE_P(
    OclLayouts,
    GGUFDenseDispatch,
    testing::Values(DenseDispatchCase{GGUF_TYPE_Q4_K, "q4_k", "q4k", 144, DenseLayout::Raw},
                    DenseDispatchCase{GGUF_TYPE_Q4_K, "q4_k", "q4k", 144, DenseLayout::Shuffled},
                    DenseDispatchCase{GGUF_TYPE_Q4_K, "q4_k", "q4k", 144, DenseLayout::RawUnaligned},
                    DenseDispatchCase{GGUF_TYPE_Q4_K, "q4_k", "q4k", 144, DenseLayout::ShuffledGemmUnaligned},
                    DenseDispatchCase{GGUF_TYPE_Q5_K, "q5_k", "q5k", 176, DenseLayout::Raw},
                    DenseDispatchCase{GGUF_TYPE_Q5_K, "q5_k", "q5k", 176, DenseLayout::Shuffled},
                    DenseDispatchCase{GGUF_TYPE_Q6_K, "q6_k", "q6k", 210, DenseLayout::Raw},
                    DenseDispatchCase{GGUF_TYPE_Q6_K, "q6_k", "q6k", 210, DenseLayout::Shuffled},
                    DenseDispatchCase{GGUF_TYPE_Q4_0, "q4_0", "q4_0", 144, DenseLayout::Raw},
                    DenseDispatchCase{GGUF_TYPE_Q4_0, "q4_0", "q4_0", 144, DenseLayout::Shuffled},
                    DenseDispatchCase{GGUF_TYPE_Q4_1, "q4_1", "q4_1", 160, DenseLayout::Raw},
                    DenseDispatchCase{GGUF_TYPE_Q4_1, "q4_1", "q4_1", 160, DenseLayout::Shuffled},
                    DenseDispatchCase{GGUF_TYPE_Q8_0, "q8_0", "q8_0", 272, DenseLayout::Raw},
                    DenseDispatchCase{GGUF_TYPE_Q8_0, "q8_0", "q8_0", 272, DenseLayout::Shuffled}),
    [](const testing::TestParamInfo<DenseDispatchCase>& info) {
        const auto layout = info.param.layout;
        const char* name = layout == DenseLayout::Raw                 ? "Raw"
                           : layout == DenseLayout::Shuffled          ? "Shuffled"
                           : layout == DenseLayout::RawUnaligned      ? "RawUnaligned"
                                                                      : "ShuffledGemmUnaligned";
        return std::string(info.param.fixture) + "_" + name;
    });
#endif

#ifdef GGUF_GPU_ZE_FALLBACK_TESTS
class GGUFZeDenseFallback : public GGUFProjection,
                            public testing::WithParamInterface<std::pair<GgufTensorType, const char*>> {};

TEST_P(GGUFZeDenseFallback, CompilesNativeFileProjectionWithoutMutatingSource) {
    const auto [type, fixture] = GetParam();
    ov::Core core;
    try {
        RecordProperty("device", core.get_property("GPU", ov::device::full_name));
    } catch (const ov::Exception& error) {
        GTEST_SKIP() << error.what();
    }
    constexpr size_t width = 256;
    constexpr size_t channels = 4;
    const auto payload = load_npy<uint8_t>(std::string(fixture) + "_qbytes");
    const auto path = write_projection_model(width, type, payload, channels);
    ASSERT_FALSE(path.empty());
    FrontEnd frontend;
    const auto graph = frontend.convert(frontend.load(path));
    std::shared_ptr<ov::op::internal::FullyConnectedCompressed> projection;
    const std::string weight_type = std::string("gguf_") + fixture;
    for (const auto& node : graph->get_ordered_ops()) {
        const auto candidate = ov::as_type_ptr<ov::op::internal::FullyConnectedCompressed>(node);
        if (candidate && candidate->get_input_element_type(1).get_type_name() == weight_type) {
            projection = candidate;
            break;
        }
    }
    ASSERT_TRUE(projection);
    const auto activation = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, width});
    auto inputs = projection->input_values();
    inputs[0] = activation;
    const auto fc = ov::as_type_ptr<ov::op::internal::FullyConnectedCompressed>(projection->clone_with_new_inputs(inputs));
    ASSERT_TRUE(fc);
    const auto weights = ov::as_type_ptr<ov::op::v0::Constant>(fc->get_input_node_shared_ptr(1));
    ASSERT_TRUE(weights);
    ASSERT_EQ(weights->get_byte_size(), payload.size());
    ASSERT_EQ(std::memcmp(weights->get_data_ptr(), payload.data(), payload.size()), 0);
    const auto make_model = [&](const std::shared_ptr<ov::Node>& output) {
        return std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{activation});
    };
    const auto model = make_model(fc);
    const auto restored = round_trip_dense_projection(core, model, path);
    const auto lowered = ov::util::make_gguf_fc_compatible(*fc, {weights->get_element_type()});
    ASSERT_TRUE(lowered);
    const auto precision = ov::hint::inference_precision(ov::element::f32);
    EXPECT_EQ(core.query_model(model, "GPU", precision).count(fc->get_friendly_name()), 1u);
    EXPECT_EQ(core.query_model(restored, "GPU", precision).count(fc->get_friendly_name()), 1u);
    auto actual = core.compile_model(model, "GPU", precision).create_infer_request();
    auto from_ir = core.compile_model(restored, "GPU", precision).create_infer_request();
    auto reference = core.compile_model(make_model(lowered), "GPU", precision).create_infer_request();
    for (const size_t rows : {1u, 33u}) {
        SCOPED_TRACE(rows);
        ov::Tensor input(ov::element::f32, {rows, width});
        std::fill_n(input.data<float>(), input.get_size(), 0.f);
        for (size_t row = 0; row < rows; ++row) {
            input.data<float>()[row * width + (7 * row + 13) % width] = 1.f;
        }
        actual.set_input_tensor(input);
        from_ir.set_input_tensor(input);
        reference.set_input_tensor(input);
        actual.infer();
        from_ir.infer();
        reference.infer();
        const auto result = actual.get_output_tensor();
        const auto restored_result = from_ir.get_output_tensor();
        const auto expected = reference.get_output_tensor();
        ASSERT_EQ(result.get_shape(), (ov::Shape{rows, channels}));
        ASSERT_EQ(result.get_shape(), expected.get_shape());
        ASSERT_EQ(result.get_shape(), restored_result.get_shape());
        ASSERT_EQ(result.get_element_type(), restored_result.get_element_type());
        EXPECT_EQ(std::memcmp(result.data(), restored_result.data(), result.get_byte_size()), 0);
        for (size_t index = 0; index < result.get_size(); ++index) {
            EXPECT_NEAR(result.data<const float>()[index], expected.data<const float>()[index], 1e-4f);
        }
    }
    EXPECT_EQ(model->get_results()[0]->get_input_node_shared_ptr(0), fc);
    ASSERT_EQ(fc->get_input_node_shared_ptr(1), weights);
    EXPECT_EQ(std::memcmp(weights->get_data_ptr(), payload.data(), payload.size()), 0);
}

INSTANTIATE_TEST_SUITE_P(Ze,
                         GGUFZeDenseFallback,
                         testing::Values(std::make_pair(GGUF_TYPE_Q4_0, "q4_0"),
                                         std::make_pair(GGUF_TYPE_Q4_1, "q4_1"),
                                         std::make_pair(GGUF_TYPE_Q4_K, "q4_k"),
                                         std::make_pair(GGUF_TYPE_Q5_K, "q5_k"),
                                         std::make_pair(GGUF_TYPE_Q6_K, "q6_k"),
                                         std::make_pair(GGUF_TYPE_Q8_0, "q8_0")),
                         [](const testing::TestParamInfo<GGUFZeDenseFallback::ParamType>& info) {
                             return info.param.second;
                         });
#endif

TEST_F(GGUFShards, FrontendConvertsTheSameConstantsFromSingleFileAndShardInputs) {
    GgufWriter complete;
    GgufWriter first;
    GgufWriter second;
    for (auto* writer : {&complete, &first}) {
        writer->kv_str("general.architecture", "qwen3");
        writer->kv_u32("qwen3.block_count", 1);
        writer->kv_u32("qwen3.embedding_length", 8);
        writer->kv_u32("qwen3.attention.head_count", 2);
        writer->kv_u32("qwen3.attention.head_count_kv", 1);
        writer->kv_u32("qwen3.rope.dimension_count", 4);
        writer->kv_u32("qwen3.context_length", 32);
        writer->kv_f32("qwen3.attention.layer_norm_rms_epsilon", 1e-5f);
    }
    const std::vector<std::pair<std::string, std::vector<uint64_t>>> tensors{
        {"token_embd.weight", {8, 16}},
        {"output_norm.weight", {8}},
        {"blk.0.attn_norm.weight", {8}},
        {"blk.0.attn_q.weight", {8, 8}},
        {"blk.0.attn_k.weight", {8, 4}},
        {"blk.0.attn_v.weight", {8, 4}},
        {"blk.0.attn_output.weight", {8, 8}},
        {"blk.0.attn_q_norm.weight", {4}},
        {"blk.0.attn_k_norm.weight", {4}},
        {"blk.0.ffn_norm.weight", {8}},
        {"blk.0.ffn_gate.weight", {8, 12}},
        {"blk.0.ffn_up.weight", {8, 12}},
        {"blk.0.ffn_down.weight", {12, 8}},
    };
    size_t i = 0;
    for (auto* writer : {&first, &second}) {
        writer->kv_u16("split.no", static_cast<uint16_t>(i++));
        writer->kv_u16("split.count", 2);
        writer->kv_i32("split.tensors.count", static_cast<int32_t>(tensors.size()));
    }
    i = 0;
    for (const auto& [name, shape] : tensors) {
        const auto norm = name.find("norm") != std::string::npos;
        complete.filled_tensor(name, shape, norm);
        (i++ % 2 == 0 ? first : second).filled_tensor(name, shape, norm);
    }
    const auto reference_path = m_base + ".gguf";
    m_paths = {reference_path, path(0), path(1)};
    ASSERT_TRUE(complete.write(reference_path));
    ASSERT_TRUE(first.write(path(0)));
    ASSERT_TRUE(second.write(path(1)));
    ov::frontend::gguf::FrontEnd frontend;
    const auto reference = frontend.convert(frontend.load(reference_path));
    auto comparator = FunctionsComparator::with_default();
    comparator.enable(FunctionsComparator::CONST_VALUES).enable(FunctionsComparator::ATTRIBUTES);
    const std::vector<ov::AnyVector> representations{
        {path(0), true},
        {path(1)},
        {std::vector<std::string>{path(1), path(0)}, true},
        {path(0), path(1)},
        {std::vector<std::filesystem::path>{path(1), path(0)}},
    };
    for (const auto& representation : representations) {
        EXPECT_TRUE(frontend.supported(representation));
        const auto model = frontend.convert(frontend.load(representation));
        EXPECT_EQ(model->get_ordered_ops().size(), reference->get_ordered_ops().size());
        const auto comparison = comparator.compare(model, reference);
        EXPECT_TRUE(comparison.valid) << comparison.message;
    }
    auto unicode_prefix = ov::util::make_path(m_base);
    unicode_prefix += ov::util::make_path(reinterpret_cast<const char*>(u8"_模型_żółć"));
    const auto unicode_first =
        std::filesystem::path(unicode_prefix.native() + std::filesystem::path("-00001-of-00002.gguf").native());
    const auto unicode_second =
        std::filesystem::path(unicode_prefix.native() + std::filesystem::path("-00002-of-00002.gguf").native());
    const auto unicode_single =
        std::filesystem::path(unicode_prefix.native() + std::filesystem::path(".gguf").native());
    std::filesystem::rename(path(0), unicode_first);
    std::filesystem::rename(path(1), unicode_second);
    std::filesystem::rename(reference_path, unicode_single);
    m_paths.insert(m_paths.end(), {unicode_first, unicode_second, unicode_single});
    for (const ov::Any& representation : {ov::Any(unicode_first),
                                          ov::Any(unicode_second),
                                          ov::Any(unicode_single),
                                          ov::Any(std::vector<std::filesystem::path>{unicode_second, unicode_first})}) {
        EXPECT_TRUE(frontend.supported(representation));
        const auto comparison = comparator.compare(frontend.convert(frontend.load(representation)), reference);
        EXPECT_TRUE(comparison.valid) << comparison.message;
    }
    EXPECT_FALSE(frontend.supported(std::vector<std::string>{}));
    EXPECT_FALSE(frontend.supported(std::vector<std::string>{path(0), m_base + "_missing.gguf"}));
}

}  // namespace
