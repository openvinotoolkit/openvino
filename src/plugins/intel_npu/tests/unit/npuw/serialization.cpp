// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <iostream>

#include "intel_npu/config/config.hpp"
#include "intel_npu/config/npuw.hpp"
#include "openvino/openvino.hpp"
#include "serialization.hpp"
#include "spatial.hpp"

TEST(SerializationTest, BasicTypes_string) {
    using namespace ov::npuw::s11n;

    std::string var("NPUW");
    std::string res;

    std::stringstream ss;

    write(ss, var);
    read(ss, res);

    EXPECT_EQ(var, res);
}

TEST(SerializationTest, BasicTypes_bool) {
    using namespace ov::npuw::s11n;

    bool var = true;
    bool res;

    std::stringstream ss;

    write(ss, var);
    read(ss, res);

    EXPECT_EQ(var, res);
}

TEST(SerializationTest, BasicTypes_float) {
    using namespace ov::npuw::s11n;

    float var = 3.1415;
    float res;

    std::stringstream ss;

    write(ss, var);
    read(ss, res);

    EXPECT_EQ(var, res);
}

TEST(SerializationTest, BasicTypes_streampos) {
    using namespace ov::npuw::s11n;

    std::stringstream buf;
    buf.write("NPUW", 4);

    std::streampos var = buf.tellp();
    std::streampos res;

    std::stringstream ss;

    write(ss, var);
    read(ss, res);

    EXPECT_EQ(var, res);
}

TEST(SerializationTest, BasicTypes_Tensor) {
    using namespace ov::npuw::s11n;

    std::vector<uint8_t> data {0, 1, 2, 3};
    ov::Tensor var(ov::element::u8, ov::Shape({2, 2}), data.data());
    ov::Tensor res;

    std::stringstream ss;

    write(ss, var);
    read(ss, res);

    EXPECT_EQ(res.get_element_type(), ov::element::u8);
    EXPECT_EQ(res.get_shape(), ov::Shape({2, 2}));

    std::vector<uint8_t> data_res(4, 0);
    std::memcpy(data_res.data(), res.data(), 4);

    EXPECT_EQ(data, data_res);
}

TEST(SerializationTest, BasicTypes_Spatial) {
    using namespace ov::npuw::s11n;

    ov::npuw::compiled::Spatial var;
    var.params = {{0, 1}, {2, 3}};
    var.range = 3;
    var.nway = 5;
    var.out_dim = 1;
    var.nway_iters = 10;
    var.tail_size = 3;

    ov::npuw::compiled::Spatial res;

    std::stringstream ss;

    write(ss, var);
    read(ss, res);

    EXPECT_EQ(var.params[0].idx, res.params[0].idx);
    EXPECT_EQ(var.params[0].dim, res.params[0].dim);
    EXPECT_EQ(var.params[1].idx, res.params[1].idx);
    EXPECT_EQ(var.params[1].dim, res.params[1].dim);
    EXPECT_EQ(var.range, res.range);
    EXPECT_EQ(var.nway, res.nway);
    EXPECT_EQ(var.out_dim, res.out_dim);
    EXPECT_EQ(var.nway_iters, res.nway_iters);
    EXPECT_EQ(var.tail_size, res.tail_size);
}

TEST(SerializationTest, BasicTypes_Config) {
    using namespace ov::npuw::s11n;

    auto options_desc(std::make_shared<::intel_npu::OptionsDesc>());
    options_desc->add<::intel_npu::NPUW_LLM_BATCH_DIM>();
    ::intel_npu::Config var(options_desc);
    ::intel_npu::Config res(options_desc);

    std::map<std::string, std::string> tmp;
    tmp["NPUW_LLM_BATCH_DIM"] = "42";
    var.update(tmp);

    std::stringstream ss;

    write(ss, var);
    read(ss, res);

    EXPECT_EQ(res.get<::intel_npu::NPUW_LLM_BATCH_DIM>(), 42);
}

TEST(SerializationTest, BasicTypes_Any) {
    using namespace ov::npuw::s11n;

    int tmp = 42;
    ov::Any var(tmp);
    ov::Any res;

    std::stringstream ss;

    write_any(ss, var);
    read_any(ss, res);

    EXPECT_EQ(var.as<int>(), res.as<int>());
}

TEST(SerializationTest, BasicTypes_array) {
    using namespace ov::npuw::s11n;

    std::array<uint8_t, 6> res;

    std::stringstream ss;

    write(ss, NPUW_SERIALIZATION_INDICATOR);
    read(ss, res);

    EXPECT_EQ(NPUW_SERIALIZATION_INDICATOR, res);
}

TEST(SerializationTest, BasicTypes_pair) {
    using namespace ov::npuw::s11n;

    std::pair<int, float> var{42, 3.14};
    std::pair<int, float> res;

    std::stringstream ss;

    write(ss, var);
    read(ss, res);

    EXPECT_EQ(var, res);
}

TEST(SerializationTest, BasicTypes_vector) {
    using namespace ov::npuw::s11n;

    std::vector<int> var{1, 2, 3, 45};
    std::vector<int> res;

    std::stringstream ss;

    write(ss, var);
    read(ss, res);

    EXPECT_EQ(var, res);
}

TEST(SerializationTest, BasicTypes_map) {
    using namespace ov::npuw::s11n;

    std::map<int, std::string> var{{1, "a"}, {2, "b"}};
    std::map<int, std::string> res;

    std::stringstream ss;

    write(ss, var);
    read(ss, res);

    EXPECT_EQ(var, res);
}

TEST(SerializationTest, BasicTypes_unordered_set) {
    using namespace ov::npuw::s11n;

    std::unordered_set<std::string> var{"a", "b", "c"};
    std::unordered_set<std::string> res;

    std::stringstream ss;

    write(ss, var);
    read(ss, res);

    EXPECT_EQ(var, res);
}

TEST(SerializationTest, BasicTypes_optional) {
    using namespace ov::npuw::s11n;

    std::optional<int> var = 1;
    std::optional<int> res;

    std::optional<int> var2 = std::nullopt;
    std::optional<int> res2;

    std::stringstream ss;

    write(ss, var);
    write(ss, var2);
    read(ss, res);
    read(ss, res2);

    EXPECT_EQ(var, res);
    EXPECT_EQ(var2, res2);
}

TEST(SerializationTest, BasicTypes_Tensor_with_weights) {
    using namespace ov::npuw::s11n;

    std::vector<uint8_t> data {0, 1, 2, 3};
    ov::Tensor var(ov::element::u8, ov::Shape({2, 2}), data.data());
    std::vector<ov::Tensor> res;

    std::stringstream ss;

    std::unordered_map<const void*, std::size_t> const_offset;
    Context ctx(false, const_offset);

    write_weightless(ss, {var}, ctx);
    read_weightless(ss, res, nullptr);

    EXPECT_EQ(res[0].get_element_type(), ov::element::u8);
    EXPECT_EQ(res[0].get_shape(), ov::Shape({2, 2}));

    std::vector<uint8_t> data_res(4, 0);
    std::memcpy(data_res.data(), res[0].data(), 4);

    EXPECT_EQ(data, data_res);
}

// TODO: add tests on CompiledModel and LLMCompiledModel once tests have access to any model to test on
