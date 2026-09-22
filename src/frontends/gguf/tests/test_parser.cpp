// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <array>
#include <fstream>
#include <utility>

#include "common_test_utils/common_utils.hpp"
#include "op_test_utils.hpp"
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
                      const std::vector<uint8_t>& data = {}) {
        std::ofstream out(m_path, std::ios::binary);
        ASSERT_TRUE(out);
        const auto put = [&out](auto value) {
            out.write(reinterpret_cast<const char*>(&value), sizeof(value));
        };
        put(uint32_t{0x46554747});
        put(uint32_t{3});
        put(uint64_t{1});  // tensor count
        put(uint64_t{0});  // metadata count
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

}  // namespace
