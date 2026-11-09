// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <tuple>
#include <utility>

#include "builder/graph_emitter.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/graph_comparator.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "gguf_writer.hpp"
#include "op_test_utils.hpp"
#include "openvino/frontend/gguf/frontend.hpp"
#include "quant/weights.hpp"

using namespace ov_gguf_test;

namespace {

class GGUFParser : public ::testing::Test {
protected:
    void SetUp() override {
        m_path = ov::test::utils::generateTestFilePrefix() + ".gguf";
    }

    void TearDown() override {
        ov::test::utils::removeFile(m_path);
    }

    // One tensor, no metadata. Dimensions use GGUF order (columns first).
    void write_tensor(const std::string& name,
                      GgufTensorType type,
                      uint64_t cols,
                      uint64_t rows,
                      const std::vector<uint8_t>& data = {},
                      const std::string& architecture = {},
                      const std::vector<uint64_t>& extra_dims = {}) {
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
        put(uint64_t{0});  // tensor offset
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
                     const std::string& architecture = {}) {
        GgufWriter writer;
        writer.kv_u16("split.no", number);
        writer.kv_u16("split.count", count);
        writer.kv_i32("split.tensors.count", total);
        if (index == 0 || !architecture.empty()) {
            writer.kv_str("general.architecture", architecture.empty() ? "qwen3" : architecture);
        }
        writer.tensor(name.empty() ? "weight" + std::to_string(index) + ".weight" : name,
                      {2},
                      {float(index + 1), float(index + 2)});
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
    ASSERT_NO_FATAL_FAILURE(write_shard(0, 0));
    ASSERT_NO_FATAL_FAILURE(write_shard(1, 1));
    const auto renamed =
        (std::filesystem::temp_directory_path() / (std::filesystem::path(m_base).filename().string() + "_renamed.gguf"))
            .string();
    m_paths.push_back(renamed);
    ASSERT_TRUE(std::filesystem::copy_file(path(1), renamed));
    const auto loaded = get_gguf_data(std::vector<std::filesystem::path>{renamed, path(0)});
    EXPECT_EQ(std::get<1>(loaded).size(), 2);
    EXPECT_EQ(std::get<std::string>(std::get<0>(loaded).at("general.architecture")), "qwen3");
    OV_EXPECT_THROW(get_gguf_data(renamed), ov::Exception, testing::HasSubstr("explicit list"));
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
