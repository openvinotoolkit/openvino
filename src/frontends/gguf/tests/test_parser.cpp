// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <array>
#include <filesystem>
#include <fstream>
#include <utility>

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
                      const std::string& architecture = {}) {
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
        put(uint32_t{2});
        put(cols);
        put(rows);
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
    std::filesystem::rename(path(1), renamed);
    m_paths.push_back(renamed);
    const auto loaded = get_gguf_data(std::vector<std::filesystem::path>{renamed, path(0)});
    EXPECT_EQ(std::get<1>(loaded).size(), 2);
    EXPECT_EQ(std::get<std::string>(std::get<0>(loaded).at("general.architecture")), "qwen3");
    OV_EXPECT_THROW(get_gguf_data(renamed), ov::Exception, testing::HasSubstr("explicit list"));
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
        writer.kv_u16("split.no", index);
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
            writer.kv_u16("split.no", index);
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
        writer->kv_u16("split.no", i++);
        writer->kv_u16("split.count", 2);
        writer->kv_i32("split.tensors.count", tensors.size());
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
