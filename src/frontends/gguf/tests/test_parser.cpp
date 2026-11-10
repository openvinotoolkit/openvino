// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <tuple>
#include <utility>
#include <pugixml.hpp>

#include "builder/graph_emitter.hpp"
#include "builder/gguf_builder_decoder.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/graph_comparator.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "gguf_writer.hpp"
#include "op_test_utils.hpp"
#include "openvino/frontend/gguf/frontend.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/properties.hpp"
#include "ov_ops/fully_connected_compressed.hpp"
#include "quant/weights.hpp"

using namespace ov_gguf_test;

namespace {

#ifdef GGUF_GPU_NATIVE_TESTS
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
        std::filesystem::remove(ov::util::make_path(m_path + u8"_\u6a21\u578b_\u017c\u00f3\u0142\u0107.gguf"));
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
        const auto& arrays = std::get<1>(loaded);
        EXPECT_EQ(std::get<2>(loaded).at(name + ".qtype"), GGUF_TYPE_Q8_1);
        const auto& weights = arrays.at(name + ".weight");
        const auto& scales = arrays.at(name + ".scales");
        ASSERT_EQ(weights.get_element_type(), ov::element::i8);
        ASSERT_EQ(scales.get_element_type(), ov::element::f16);
        EXPECT_EQ(weights.get_shape(), (ov::Shape{4, 256}));
        EXPECT_EQ(scales.get_shape(), (ov::Shape{4, 8}));
        EXPECT_EQ(weights.get_byte_size() + scales.get_byte_size(), 4 * 256 / 32 * 34);
        EXPECT_EQ(arrays.count(name + ".zp"), 0);
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

TEST_F(GGUFParser, Q8_1KeepsHigherRankGroups) {
    const auto bytes = load_npy<uint8_t>("q8_1_qbytes");
    const auto reference = load_npy<float>("q8_1_deq");
    ASSERT_NO_FATAL_FAILURE(write_tensor("experts.weight", GGUF_TYPE_Q8_1, 256, 2, bytes, {}, {2}));
    const auto loaded = get_gguf_data(m_path);
    const auto& arrays = std::get<1>(loaded);
    const auto& weights = arrays.at("experts.weight");
    const auto& scales = arrays.at("experts.scales");
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
        const auto loaded = get_gguf_data(m_path);
        const auto& arrays = std::get<1>(loaded);
        tensors = {arrays.at("output.weight"), arrays.at("output.scales"), {}};
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
    const auto loaded = get_gguf_data(m_path);
    const auto& arrays = std::get<1>(loaded);
    const auto& weights = arrays.at("token_embd.weight");
    const auto& scales = arrays.at("token_embd.scales");
    ASSERT_EQ(weights.get_element_type(), ov::element::u1);
    EXPECT_EQ(weights.get_shape(), (ov::Shape{4, 256}));
    EXPECT_EQ(weights.get_byte_size(), 128);
    EXPECT_EQ(scales.get_byte_size(), 16);
    EXPECT_EQ(arrays.count("token_embd.zp"), 0);
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
        const auto loaded = get_gguf_data(m_path);
        const auto& arrays = std::get<1>(loaded);
        const auto& weights = arrays.at(name + ".weight");
        const auto& scales = arrays.at(name + ".scales");
        const auto& zp = arrays.at(name + ".zp");
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
        const auto& arrays = std::get<1>(loaded);
        EXPECT_EQ(std::get<2>(loaded).at("token_embd.qtype"), expected_type);
        EXPECT_EQ(arrays.at("token_embd.scales").get_shape(), (ov::Shape{4, 2}));
        const auto node = make_weight_node(
            {arrays.at("token_embd.weight"), arrays.at("token_embd.scales"), arrays.at("token_embd.zp")},
            expected_type,
            "token_embd");
        const auto model = std::make_shared<ov::Model>(ov::OutputVector{node}, ov::ParameterVector{});
        const auto output = run_on_cpu(model, {});
        ASSERT_EQ(output.get_size(), reference.size());
        for (size_t i = 0; i < reference.size(); ++i)
            EXPECT_FLOAT_EQ(output.data<const float>()[i], reference[i]);
    }
}

TEST(GGUFHadamardMetadata, RejectsInvalidBlocksWidthsAndUnusedWeights) {
    std::unordered_map<std::string, ov::Tensor> weights{{"output.weight", ov::Tensor(ov::element::f32, {4, 16})},
                                                        {"token_embd.weight", ov::Tensor(ov::element::f32, {4, 12})}};
    std::unordered_map<std::string, GgufTensorType> qtypes;
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
        ov::frontend::gguf::GraphEmitter emitter(weights, qtypes, "qwen35");
        EXPECT_ANY_THROW(emitter.configure_hadamard(invalid));
    }
    auto inverse = config;
    inverse[p + "inverse_weight_names"] = std::vector<std::string>{"token_embd.weight"};
    ov::frontend::gguf::GraphEmitter bad_inverse(weights, qtypes, "qwen35");
    EXPECT_ANY_THROW(bad_inverse.configure_hadamard(inverse));
    ov::frontend::gguf::GraphEmitter unused(weights, qtypes, "qwen35");
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
        const auto loaded = get_gguf_data(m_path);
        const auto& arrays = std::get<1>(loaded);
        const auto& zp = arrays.at("v.blk.0.attn_q.zp");
        ASSERT_EQ(zp.get_element_type(), expected_zp);
        const auto node = make_weight_node({arrays.at("v.blk.0.attn_q.weight"), arrays.at("v.blk.0.attn_q.scales"), zp},
                                           GGUF_TYPE_Q4_K,
                                           "v.blk.0.attn_q.weight");
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
            EXPECT_FALSE(std::get<1>(legacy).at("blk.0.attn_q.weight").get_element_type().is_gguf_block());
            const auto original_option = get_gguf_data(m_path, GGUFLoadOptions{true});
            EXPECT_EQ(std::get<1>(original_option).at("blk.0.attn_q.weight").get_element_type().is_gguf_block(),
                      format.type == GGUF_TYPE_Q4_K);
            const auto decomposed = get_gguf_data(m_path, unselected);
            EXPECT_FALSE(std::get<1>(decomposed).at("blk.0.attn_q.weight").get_element_type().is_gguf_block());
        }
        const GGUFLoadOptions options{true, true, true, true, true, true, true, true, true, true, true};
        std::shared_ptr<ov::op::v0::Constant> constant;
        std::weak_ptr<ov::MappedMemory> owner;
        {
            const auto loaded = get_gguf_data(m_path, options);
            const auto& arrays = std::get<1>(loaded);
            ASSERT_EQ(arrays.size(), 1);
            const auto& raw = arrays.at("blk.0.attn_q.weight");
            ASSERT_EQ(raw.get_element_type(), gguf_raw_element_type(format.type));
            EXPECT_EQ(raw.get_shape(), (ov::Shape{4, 256}));
            ASSERT_EQ(raw.get_byte_size(), bytes.size());
            EXPECT_EQ(std::memcmp(raw.data(), bytes.data(), bytes.size()), 0);
            EXPECT_EQ(std::get<2>(loaded).at("blk.0.attn_q.qtype"), format.type);
            owner = std::get<3>(loaded).at(0);
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
    EXPECT_EQ(std::get<1>(legacy).at("blk.0.attn_q.weight").get_element_type(), ov::element::u32);
    EXPECT_EQ(std::get<1>(legacy).count("blk.0.attn_q.scales"), 1);
    EXPECT_EQ(std::get<1>(legacy).count("blk.0.attn_q.zp"), 1);
    const auto loaded = get_gguf_data(std::vector<std::filesystem::path>{m_path}, GGUFLoadOptions{true});
    const auto& raw = std::get<1>(loaded).at("blk.0.attn_q.weight");
    OV_EXPECT_THROW(make_weight_node({raw, {}, {}}, GGUF_TYPE_Q4_K, "blk.0.attn_q"),
                    ov::Exception,
                    testing::HasSubstr("raw GGUF weights require a supported graph boundary"));
}

class GGUFParserGpu : public GGUFParser, public ::testing::WithParamInterface<GgufRawFormat> {};

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
        GraphEmitter emitter(std::get<1>(loaded), std::get<2>(loaded), "test");
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
    GraphEmitter emitter(std::get<1>(loaded), std::get<2>(loaded), "test");
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
            const auto unicode_path = ov::util::make_path(m_path + u8"_\u6a21\u578b_\u017c\u00f3\u0142\u0107.gguf");
            ASSERT_TRUE(std::filesystem::copy_file(m_path, unicode_path));
            auto unicode_data = get_gguf_data(unicode_path, options);
            GraphEmitter unicode_emitter(std::get<1>(unicode_data), std::get<2>(unicode_data), "test",
                                         nullptr, &std::get<4>(unicode_data));
            unicode_emitter.add_weight("projection.weight");
            const auto unicode_raw = unicode_emitter.value("projection.weight").get_node_shared_ptr();
            EXPECT_EQ(unicode_raw->get_rt_info().at("gguf_ext_source").as<std::string>(),
                      ov::util::path_to_string(unicode_path.filename()));
            auto unicode_model = std::make_shared<ov::Model>(
                ov::OutputVector{unicode_raw, ov::op::v0::Constant::create(ov::element::f32, {1}, {1.f})},
                ov::ParameterVector{});
            ov::pass::Serialize(m_path + ".unicode.xml", m_path + ".unicode.bin").run_on_model(unicode_model);
            const auto restored = core.read_model(m_path + ".unicode.xml");
            const auto restored_raw = ov::as_type_ptr<ov::op::v0::Constant>(
                restored->get_results().at(0)->get_input_node_shared_ptr(0));
            ASSERT_TRUE(restored_raw);
            EXPECT_EQ(std::memcmp(restored_raw->get_data_ptr(), payload.data(), payload.size()), 0);
            std::stringstream unicode_xml, unicode_binary;
            ov::pass::Serialize(unicode_xml, unicode_binary).run_on_model(restored);
            EXPECT_LT(unicode_binary.str().size(), payload.size());
            RecordProperty("unicode_automatic_reference_roundtrip", "1");
        }
        auto loaded = get_gguf_data(m_path, options);
        auto& weights = std::get<1>(loaded);
        const auto& raw = weights.at("projection.weight");
        ASSERT_EQ(raw.get_element_type(), gguf_raw_element_type(format.type));
        ASSERT_EQ(raw.get_shape(), (ov::Shape{channels, columns}));
        ASSERT_EQ(raw.get_byte_size(), payload.size());
        ASSERT_EQ(std::memcmp(raw.data(), payload.data(), payload.size()), 0);
        owner = std::get<3>(loaded).at(0);
        GraphEmitter emitter(weights, std::get<2>(loaded), "test", nullptr, &std::get<4>(loaded));
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
        const auto alias = ov::as_type_ptr<ov::op::v0::Constant>(emitter.value("alias.weight").get_node_shared_ptr());
        ASSERT_TRUE(alias);
        EXPECT_EQ(alias->get_rt_info().at("gguf_ext_offset").as<uint64_t>(),
              constant->get_rt_info().at("gguf_ext_offset").as<uint64_t>());
        ov::Tensor copied(raw.get_element_type(), raw.get_shape());
        std::memcpy(copied.data(), raw.data(), raw.get_byte_size());
        emitter.emit_weight_op("copied.weight", {copied, {}, {}}, format.type, "projection.weight");
        const auto copy = emitter.value("copied.weight").get_node_shared_ptr();
        EXPECT_EQ(copy->get_rt_info().count("gguf_ext_source"), 0);
        ov::Tensor sliced(raw.get_element_type(), {channels / 2, columns});
        std::memcpy(sliced.data(), raw.data(), sliced.get_byte_size());
        emitter.emit_weight_op("sliced.weight", {sliced, {}, {}}, format.type, "projection.weight");
        const auto slice = emitter.value("sliced.weight").get_node_shared_ptr();
        EXPECT_EQ(slice->get_rt_info().count("gguf_ext_source"), 0);
        RecordProperty("automatic_external_source", "1");
        RecordProperty("copied_sliced_sources_absent", "1");
    }
    EXPECT_FALSE(owner.expired());
    std::set<std::string> executed_stages;
    bool ir_roundtrip = false;
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
        const bool cm_requested = std::getenv("OV_GPU_GGUF_GEMM_CM") &&
                                  std::string(std::getenv("OV_GPU_GGUF_GEMM_CM")) == "1";
        GgufScopedEnvironment cm("OV_GPU_GGUF_GEMM_CM", cm_requested ? "1" : "0");
        const bool shuffled = std::string(shuffle_value) == "1";
        const std::string scenario = std::string(binary_mode ? binary_mode : "plain") + "_" +
            inference_type.get_type_name() + "_s" + shuffle_value + (threshold_value ? "_native" : "_default") +
            (static_rows ? "_static" : "_dynamic") + (rebind_broadcast ? "_broadcast" : "") +
            (ir_roundtrip ? "_ir" : "") + (external_reference ? "_external" : "") +
            (compiled_import ? "_import" : "") + (stream_count == 2 ? "_streams2" : "");
        SCOPED_TRACE(scenario);
        const bool check_cache = !cache_directory.empty() && !binary_mode && !ir_roundtrip;
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
                    const std::string expected_stage = !threshold_value && rows > 32
                        ? "fc_gguf_transcode_" + weight_name : "fc_gguf_opt_" + weight_name;
                    EXPECT_NE(iteration_stages.find(expected_stage), std::string::npos) << iteration_stages;
                    EXPECT_EQ(iteration_stages.find("_sg"), std::string::npos) << iteration_stages;
                    EXPECT_EQ(iteration_stages.find("gemm_q"), std::string::npos) << iteration_stages;
                }
                EXPECT_EQ(iteration_stages.find("fc_gguf_act_dq"), std::string::npos) << iteration_stages;
                if (binary_mode) {
                    const bool expect_fused = !static_rows || (cm_requested && shuffled);
                    EXPECT_EQ(iteration_fusions.find("file_binary_post") != std::string::npos, expect_fused) << iteration_fusions;
                    EXPECT_EQ(iteration_fusions.rfind("0:", 0) == 0u, !expect_fused) << iteration_fusions;
                    RecordProperty(request_scenario + "_fusions_" + std::to_string(iteration), iteration_fusions);
                }
                if (binary_mode || stream_count == 2) {
                    const bool prefill = !threshold_value && rows > 32;
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
                    EXPECT_NE(iteration_stages.find(expected_stage), std::string::npos) << iteration_stages;
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
                            const float reference = !threshold_value && rows > 32
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
                EXPECT_EQ(owner.expired(), ir_roundtrip);
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
        ov::pass::Serialize(external_xml, external_binary).run_on_model(model);
        EXPECT_LT(external_binary.str().size(), payload.size());
        RecordProperty("external_reference_bin_bytes", std::to_string(external_binary.str().size()));
        RecordProperty("external_reference_raw_bytes", std::to_string(payload.size()));
        ov::pass::Serialize(m_path + ".xml", m_path + ".bin").run_on_model(model);
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
            ov::pass::Serialize(chained_xml, chained_binary).run_on_model(external_model);
            RecordProperty("external_reference_chained_bin_bytes", std::to_string(chained_binary.str().size()));
            EXPECT_EQ(chained_binary.str().size(), external_binary.str().size());
            pugi::xml_document chained_document;
            ASSERT_TRUE(chained_document.load_string(chained_xml.str().c_str()));
            size_t chained_sources = 0;
            for (const auto layer : chained_document.child("net").child("layers").children("layer")) {
                if (const auto source = layer.child("data").attribute("source")) {
                    ++chained_sources;
                    EXPECT_EQ(std::string(source.value()), std::filesystem::path(m_path).filename().string());
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
                ov::pass::Serialize(next_xml, next_binary).run_on_model(chained_model);
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
            GgufScopedEnvironment sharing_cm("OV_GPU_GGUF_GEMM_CM", "0");
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
        info["gguf_ext_size"] = static_cast<uint64_t>(payload.size() - 1);
        std::stringstream invalid_xml, invalid_binary;
        EXPECT_ANY_THROW(ov::pass::Serialize(invalid_xml, invalid_binary).run_on_model(model));
        info["gguf_ext_size"] = static_cast<uint64_t>(payload.size());
        ov::pass::Serialize(m_path + ".xml", m_path + ".bin").run_on_model(model);
        info.erase("gguf_ext_source");
        info.erase("gguf_ext_offset");
        info.erase("gguf_ext_size");
        info.erase("gguf_ext_source_path");
    }
    std::stringstream xml, binary;
    ov::pass::Serialize(xml, binary).run_on_model(model);
    const auto serialized = binary.str();
    ov::Tensor constants(ov::element::u8, {serialized.size()});
    std::memcpy(constants.data(), serialized.data(), serialized.size());
    model = core.read_model(xml.str(), constants);
    ir_roundtrip = true;
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
    RecordProperty("ir_roundtrip_raw_constants", std::to_string(raw_constants));
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
        const auto replacement_model = core.read_model(m_path + ".xml");
        const auto replacement_fc = ov::as_type_ptr<ov::op::internal::FullyConnectedCompressed>(
            replacement_model->get_results().at(0)->get_input_node_shared_ptr(0));
        ASSERT_TRUE(replacement_fc);
        const auto replacement_raw = ov::as_type_ptr<ov::op::v0::Constant>(replacement_fc->get_input_node_shared_ptr(1));
        ASSERT_TRUE(replacement_raw);
        EXPECT_EQ(std::memcmp(replacement_raw->get_data_ptr(), replacement_payload.data(), payload.size()), 0);
        const auto original_fc = ov::as_type_ptr<ov::op::internal::FullyConnectedCompressed>(
            model->get_results().at(0)->get_input_node_shared_ptr(0));
        ASSERT_TRUE(original_fc);
        const auto original_raw = ov::as_type_ptr<ov::op::v0::Constant>(original_fc->get_input_node_shared_ptr(1));
        ASSERT_TRUE(original_raw);
        EXPECT_EQ(std::memcmp(original_raw->get_data_ptr(), payload.data(), payload.size()), 0);
        std::stringstream stale_xml, stale_binary;
        EXPECT_ANY_THROW(ov::pass::Serialize(stale_xml, stale_binary).run_on_model(model));
        RecordProperty("external_reference_stale_snapshot_rejected", "1");
        const auto materialized_model = model->clone();
        for (const auto& operation : materialized_model->get_ordered_ops()) {
            auto& metadata = operation->get_rt_info();
            for (const auto* key : {"gguf_ext_source", "gguf_ext_offset", "gguf_ext_size", "gguf_ext_source_path"}) {
                metadata.erase(key);
            }
        }
        std::stringstream materialized_xml, materialized_binary;
        ov::pass::Serialize(materialized_xml, materialized_binary).run_on_model(materialized_model);
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
        std::stringstream fresh_xml, fresh_binary;
        EXPECT_NO_THROW(ov::pass::Serialize(fresh_xml, fresh_binary).run_on_model(replacement_model));
        EXPECT_LT(fresh_binary.str().size(), payload.size());
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
        const auto& raw = std::get<1>(empty).at("empty.weight");
        EXPECT_EQ(raw.get_element_type(), ov::element::gguf_q4_k);
        EXPECT_EQ(raw.get_shape(), (ov::Shape{0, 256}));
        EXPECT_EQ(raw.get_byte_size(), 0);
    }
    ASSERT_NO_FATAL_FAILURE(write_tensor("other.weight", GGUF_TYPE_Q8_0, 32, 1, std::vector<uint8_t>(34)));
    const auto other = get_gguf_data(m_path, GGUFLoadOptions{true});
    EXPECT_EQ(std::get<1>(other).at("other.weight").get_element_type(), ov::element::i8);
    EXPECT_EQ(std::get<1>(other).count("other.scales"), 1);
}

TEST_F(GGUFParser, QuantizedBuffersOutliveParserResult) {
    const auto bytes = load_npy<uint8_t>("q2_k_qbytes");
    const auto reference = load_npy<float>("q2_k_deq");
    ASSERT_NO_FATAL_FAILURE(write_tensor("blk.0.attn_q.weight", GGUF_TYPE_Q2_K, 256, 4, bytes));
    WeightTensors tensors;
    {
        const auto loaded = get_gguf_data(m_path);
        const auto& arrays = std::get<1>(loaded);
        tensors = {arrays.at("blk.0.attn_q.weight"), arrays.at("blk.0.attn_q.scales"), arrays.at("blk.0.attn_q.zp")};
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
        const auto loaded = get_gguf_data(m_path);
        const auto& arrays = std::get<1>(loaded);
        EXPECT_EQ(arrays.at("empty.weight").get_shape(), (ov::Shape{2, 0}));
        EXPECT_EQ(arrays.at("empty.scales").get_shape(), (ov::Shape{2, 0}));
        for (const auto& entry : arrays) {
            EXPECT_EQ(entry.second.get_size(), 0);
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
    ov::Tensor retained;
    for (size_t index : {0, 1}) {
        const auto loaded = get_gguf_data(path(index));
        EXPECT_EQ(std::get<3>(loaded).size(), 2);
        EXPECT_EQ(std::get<std::string>(std::get<0>(loaded).at("general.architecture")), "qwen3");
        const auto& arrays = std::get<1>(loaded);
        ASSERT_EQ(arrays.size(), 2);
        EXPECT_FLOAT_EQ(arrays.at("weight0.weight").data<const float>()[0], 1.f);
        EXPECT_FLOAT_EQ(arrays.at("weight1.weight").data<const float>()[0], 2.f);
        retained = arrays.at("weight1.weight");
    }
    EXPECT_FLOAT_EQ(retained.data<const float>()[1], 3.f);
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
        EXPECT_TRUE(std::get<4>(loaded).empty());
        EXPECT_EQ(std::get<1>(loaded).at("weight0.weight").get_element_type(),
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
                EXPECT_EQ(std::get<2>(loaded).at(stem + ".qtype"), GGUF_TYPE_PQ2_0);
                const auto& arrays = std::get<1>(loaded);
                const auto& expected = std::get<1>(reference);
                for (const auto& suffix : {".weight", ".scales", ".zp"}) {
                    const auto& actual = arrays.at(stem + suffix);
                    const auto& value = expected.at(stem + suffix);
                    ASSERT_EQ(actual.get_element_type(), value.get_element_type());
                    ASSERT_EQ(actual.get_shape(), value.get_shape());
                    EXPECT_EQ(std::memcmp(actual.data(), value.data(), actual.get_byte_size()), 0);
                }
                const auto node = make_weight_node(
                    {arrays.at(stem + ".weight"), arrays.at(stem + ".scales"), arrays.at(stem + ".zp")},
                    GGUF_TYPE_PQ2_0);
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
    const auto unicode_prefix = ov::util::make_path(m_base + u8"_模型_żółć");
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
