// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "serialization.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <iostream>

#include "attention.hpp"
#include "common_test_utils/file_utils.hpp"
#include "compiled_model.hpp"
#include "host_flash_attention.hpp"
#include "intel_npu/config/config.hpp"
#include "intel_npu/config/npuw.hpp"
#include "lazy_tensor.hpp"
#include "moe_transformations/moe_transformation.hpp"
#include "model_builder.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/openvino.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/mmap_object.hpp"
#include "pyramid_attention.hpp"
#include "spatial.hpp"
#include "weights_bank.hpp"

using ov::test::npuw::ModelBuilder;

namespace {

void expect_tensors_equal(const ov::Tensor& expected, const ov::Tensor& actual) {
    ASSERT_EQ(static_cast<bool>(expected), static_cast<bool>(actual));
    if (!expected) {
        return;
    }

    EXPECT_EQ(expected.get_element_type(), actual.get_element_type());
    EXPECT_EQ(expected.get_shape(), actual.get_shape());
    ASSERT_EQ(expected.get_byte_size(), actual.get_byte_size());
    EXPECT_EQ(std::memcmp(expected.data(), actual.data(), expected.get_byte_size()), 0);
}

template <typename T>
std::shared_ptr<ov::op::v0::Constant> make_weightless_constant(const ov::element::Type& type,
                                                               const ov::Shape& shape,
                                                               const std::vector<T>& data,
                                                               std::size_t offset) {
    auto constant = std::make_shared<ov::op::v0::Constant>(type, shape, data);
    constant->get_rt_info()[ov::WeightlessCacheAttribute::get_type_info_static()] =
        ov::WeightlessCacheAttribute(constant->get_byte_size(), offset, type);
    return constant;
}

ov::npuw::s11n::WeightsContext::ConstsCache make_consts_cache(
    const std::vector<std::shared_ptr<ov::op::v0::Constant>>& constants) {
    ov::npuw::s11n::WeightsContext::ConstsCache cache;
    for (const auto& constant : constants) {
        const auto attr =
            constant->get_rt_info().at(ov::WeightlessCacheAttribute::get_type_info_static())
                .as<ov::WeightlessCacheAttribute>();
        cache[{attr.bin_offset, constant->get_byte_size()}] = constant;
    }
    return cache;
}

void expect_attention_equal(const ov::npuw::compiled::Attention& expected, const ov::npuw::compiled::Attention& actual) {
    EXPECT_EQ(expected.query_size, actual.query_size);
    EXPECT_EQ(expected.context_size, actual.context_size);
    EXPECT_EQ(expected.params.size(), actual.params.size());
    for (std::size_t i = 0; i < expected.params.size(); ++i) {
        EXPECT_EQ(expected.params[i].idx, actual.params[i].idx);
        EXPECT_EQ(expected.params[i].dim, actual.params[i].dim);
    }
    EXPECT_EQ(expected.mask_idx, actual.mask_idx);
    expect_tensors_equal(expected.attend_all, actual.attend_all);
}

void expect_pyramid_attention_equal(const ov::npuw::compiled::PyramidAttention& expected,
                                    const ov::npuw::compiled::PyramidAttention& actual) {
    EXPECT_EQ(expected.query_size, actual.query_size);
    EXPECT_EQ(expected.full_context_size, actual.full_context_size);
    EXPECT_EQ(expected._context_lengths, actual._context_lengths);
    ASSERT_EQ(expected._attention_infos.size(), actual._attention_infos.size());
    for (std::size_t i = 0; i < expected._attention_infos.size(); ++i) {
        const auto& lhs = expected._attention_infos[i];
        const auto& rhs = actual._attention_infos[i];
        EXPECT_EQ(lhs.mask_idx, rhs.mask_idx);
        EXPECT_EQ(lhs.query_size, rhs.query_size);
        EXPECT_EQ(lhs.context_length, rhs.context_length);
        EXPECT_EQ(lhs.params.size(), rhs.params.size());
        for (std::size_t j = 0; j < lhs.params.size(); ++j) {
            EXPECT_EQ(lhs.params[j].idx, rhs.params[j].idx);
            EXPECT_EQ(lhs.params[j].dim, rhs.params[j].dim);
        }
    }
}

void expect_host_flash_attention_equal(const ov::npuw::compiled::HostFlashAttention& expected,
                                       const ov::npuw::compiled::HostFlashAttention& actual) {
    const auto& lhs = expected._sdpa_attention_info;
    const auto& rhs = actual._sdpa_attention_info;
    EXPECT_EQ(lhs._query_size, rhs._query_size);
    EXPECT_EQ(lhs._context_size, rhs._context_size);
    EXPECT_EQ(lhs._k_seq_dim, rhs._k_seq_dim);
    EXPECT_EQ(lhs._v_seq_dim, rhs._v_seq_dim);
    EXPECT_EQ(lhs._sdpa_indices.query, rhs._sdpa_indices.query);
    EXPECT_EQ(lhs._sdpa_indices.past_key, rhs._sdpa_indices.past_key);
    EXPECT_EQ(lhs._sdpa_indices.past_value, rhs._sdpa_indices.past_value);
    EXPECT_EQ(lhs._sdpa_indices.present_key, rhs._sdpa_indices.present_key);
    EXPECT_EQ(lhs._sdpa_indices.present_value, rhs._sdpa_indices.present_value);
    EXPECT_EQ(lhs._sdpa_indices.attention_mask, rhs._sdpa_indices.attention_mask);
    EXPECT_EQ(lhs._tile_input_indices.q, rhs._tile_input_indices.q);
    EXPECT_EQ(lhs._tile_input_indices.k, rhs._tile_input_indices.k);
    EXPECT_EQ(lhs._tile_input_indices.v, rhs._tile_input_indices.v);
    EXPECT_EQ(lhs._tile_input_indices.mask, rhs._tile_input_indices.mask);
    EXPECT_EQ(lhs._tile_input_indices.acc, rhs._tile_input_indices.acc);
    EXPECT_EQ(lhs._tile_input_indices.max, rhs._tile_input_indices.max);
    EXPECT_EQ(lhs._tile_input_indices.d, rhs._tile_input_indices.d);
    EXPECT_EQ(lhs._tile_output_indices.acc, rhs._tile_output_indices.acc);
    EXPECT_EQ(lhs._tile_output_indices.max, rhs._tile_output_indices.max);
    EXPECT_EQ(lhs._tile_output_indices.d, rhs._tile_output_indices.d);
    EXPECT_EQ(expected._tile_size, actual._tile_size);
    EXPECT_EQ(expected._can_use_tensor_view, actual._can_use_tensor_view);
}

void expect_moe_experts_equal(const ov::npuw::compiled::MoEExperts& expected,
                              const ov::npuw::compiled::MoEExperts& actual) {
    EXPECT_EQ(expected.num_experts, actual.num_experts);
    EXPECT_EQ(expected.expert_hidden_dim, actual.expert_hidden_dim);
    EXPECT_EQ(expected.num_active_experts, actual.num_active_experts);
    EXPECT_EQ(expected.input_token_count, actual.input_token_count);
    EXPECT_EQ(expected._router_scores_idx, actual._router_scores_idx);
    EXPECT_EQ(expected._expert_input_param_idx, actual._expert_input_param_idx);
    EXPECT_EQ(expected._param_mapping, actual._param_mapping);
}

void expect_moe_downstream_equal(const ov::npuw::compiled::MoEDownstream& expected,
                                 const ov::npuw::compiled::MoEDownstream& actual) {
    EXPECT_EQ(expected.total_experts_num, actual.total_experts_num);
    EXPECT_EQ(expected.active_experts_num, actual.active_experts_num);
    EXPECT_EQ(expected.expert_output_param_idx, actual.expert_output_param_idx);
}

void expect_lazy_tensor_transform_types_equal(const ov::npuw::weights::LazyTensor& expected,
                                              const ov::npuw::weights::LazyTensor& actual) {
    const auto expected_transforms = expected.get_transformations();
    const auto actual_transforms = actual.get_transformations();
    ASSERT_EQ(expected_transforms.size(), actual_transforms.size());
    for (std::size_t i = 0; i < expected_transforms.size(); ++i) {
        EXPECT_EQ(expected_transforms[i].index(), actual_transforms[i].index());
    }
}

}  // namespace

// FIXME: parametrize all the tests below

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

    float var = 3.1415f;
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

TEST(SerializationTest, OVTypes_Tensor) {
    using namespace ov::npuw::s11n;

    std::vector<uint8_t> data{0, 1, 2, 3};
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

TEST(SerializationTest, OVTypes_Spatial) {
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

TEST(SerializationTest, OVTypes_Config) {
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

TEST(SerializationTest, OVTypes_Any) {
    using namespace ov::npuw::s11n;

    std::vector<ov::Any> var;
    var.push_back(42);
    var.push_back("42");
    var.push_back(3.14f);
    var.push_back(true);
    std::vector<ov::Any> res;
    res.resize(var.size());

    std::stringstream ss;

    for (std::size_t i = 0; i < var.size(); ++i) {
        write_any(ss, var[i]);
        read_any(ss, res[i]);
        EXPECT_EQ(var[i], res[i]);
    }
}

TEST(SerializationTest, BasicTypes_Indicator) {
    using namespace ov::npuw::s11n;

    IndicatorType res;

    std::stringstream ss;

    write(ss, NPUW_SERIALIZATION_INDICATOR);
    read(ss, res);

    EXPECT_EQ(NPUW_SERIALIZATION_INDICATOR, res);
}

TEST(SerializationTest, BasicTypes_pair) {
    using namespace ov::npuw::s11n;

    std::pair<int, float> var{42, 3.14f};
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

TEST(SerializationTest, BasicTypes_array) {
    using namespace ov::npuw::s11n;

    std::array<int, 4> var{1, 2, 3, 4};
    std::array<int, 4> res{};

    std::stringstream ss;

    write(ss, var);
    read(ss, res);

    EXPECT_EQ(var, res);
}

TEST(SerializationTest, BasicTypes_vector_bool) {
    using namespace ov::npuw::s11n;

    std::vector<bool> var{true, false, true, true, false};
    std::vector<bool> res;

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

TEST(SerializationTest, OVTypes_AnyMap) {
    using namespace ov::npuw::s11n;

    ov::AnyMap var = {{"NPU_USE_NPUW", std::string("YES")},
                      {"NPUW_FUNCALL_FOR_ALL", true},
                      {"NPUW_ACC_THRESH", 0.125f},
                      {"NPUW_LLM_BATCH_DIM", 7}};
    ov::AnyMap res;

    std::stringstream ss;

    write(ss, var);
    read(ss, res);

    ASSERT_EQ(var.size(), res.size());
    EXPECT_EQ(res.at("NPU_USE_NPUW").as<std::string>(), "YES");
    EXPECT_EQ(res.at("NPUW_FUNCALL_FOR_ALL").as<bool>(), true);
    EXPECT_FLOAT_EQ(res.at("NPUW_ACC_THRESH").as<float>(), 0.125f);
    EXPECT_EQ(res.at("NPUW_LLM_BATCH_DIM").as<int>(), 7);
}

TEST(SerializationTest, OVTypes_ElementType) {
    using namespace ov::npuw::s11n;

    ov::element::Type var = ov::element::f16;
    ov::element::Type res;

    std::stringstream ss;

    write(ss, var);
    read(ss, res);

    EXPECT_EQ(var, res);
}

TEST(SerializationTest, OVTypes_CacheMode) {
    using namespace ov::npuw::s11n;

    ov::CacheMode var = ov::CacheMode::OPTIMIZE_SPEED;
    ov::CacheMode res = ov::CacheMode::OPTIMIZE_SIZE;

    std::stringstream ss;

    write(ss, var);
    read(ss, res);

    EXPECT_EQ(var, res);
}

TEST(SerializationTest, OVTypes_PerformanceMode) {
    using namespace ov::npuw::s11n;

    ov::hint::PerformanceMode var = ov::hint::PerformanceMode::LATENCY;
    ov::hint::PerformanceMode res = ov::hint::PerformanceMode::THROUGHPUT;

    std::stringstream ss;

    write(ss, var);
    read(ss, res);

    EXPECT_EQ(var, res);
}

TEST(SerializationTest, OVTypes_Attention) {
    using namespace ov::npuw::s11n;

    ov::npuw::compiled::Attention var;
    var.query_size = 3;
    var.context_size = 5;
    var.params = {{1, 2}, {3, 4}};
    var.mask_idx = 7;
    var.attend_all = ov::Tensor(ov::element::f32, ov::Shape{1, 1, 3, 5});
    std::vector<float> mask_data(var.attend_all.get_size());
    std::iota(mask_data.begin(), mask_data.end(), -2.0f);
    std::memcpy(var.attend_all.data(), mask_data.data(), var.attend_all.get_byte_size());

    ov::npuw::compiled::Attention res;

    std::stringstream ss;

    write(ss, var);
    read(ss, res);

    expect_attention_equal(var, res);
}

TEST(SerializationTest, OVTypes_PyramidAttention) {
    using namespace ov::npuw::s11n;

    ov::npuw::compiled::PyramidAttention var;
    var.query_size = 16;
    var.full_context_size = 128;
    var._context_lengths = {16, 32, 64, 128};
    var._attention_infos = {{{{0, 2}, {1, 3}}, 4, 16, 32}, {{{2, 1}}, 5, 16, 64}};

    ov::npuw::compiled::PyramidAttention res;

    std::stringstream ss;

    write(ss, var);
    read(ss, res);

    expect_pyramid_attention_equal(var, res);
}

TEST(SerializationTest, OVTypes_HostFlashAttention) {
    using namespace ov::npuw::s11n;

    ov::npuw::compiled::HostFlashAttention var;
    var._sdpa_attention_info._query_size = 8;
    var._sdpa_attention_info._context_size = 32;
    var._sdpa_attention_info._k_seq_dim = 1;
    var._sdpa_attention_info._v_seq_dim = 2;
    var._sdpa_attention_info._sdpa_indices = {3, 4, 5, 6, 7, 8};
    var._sdpa_attention_info._tile_input_indices = {9, 10, 11, 12, 13, 14, 15};
    var._sdpa_attention_info._tile_output_indices = {16, 17, 18};
    var._tile_size = 64;
    var._can_use_tensor_view = true;

    ov::npuw::compiled::HostFlashAttention res;

    std::stringstream ss;

    write(ss, var);
    read(ss, res);

    expect_host_flash_attention_equal(var, res);
}

TEST(SerializationTest, OVTypes_MoEExperts) {
    using namespace ov::npuw::s11n;

    ov::npuw::compiled::MoEExperts var;
    var.num_experts = 16;
    var.expert_hidden_dim = 128;
    var.num_active_experts = 2;
    var.input_token_count = 64;
    var._router_scores_idx = 5;
    var._expert_input_param_idx = 3;
    var._param_mapping = {{0, {1, 2}}, {4, {6, 7, 8}}};

    ov::npuw::compiled::MoEExperts res;

    std::stringstream ss;

    write(ss, var);
    read(ss, res);

    expect_moe_experts_equal(var, res);
}

TEST(SerializationTest, OVTypes_MoEDownstream) {
    using namespace ov::npuw::s11n;

    ov::npuw::compiled::MoEDownstream var;
    var.total_experts_num = 16;
    var.active_experts_num = 2;
    var.expert_output_param_idx = 9;

    ov::npuw::compiled::MoEDownstream res;

    std::stringstream ss;

    write(ss, var);
    read(ss, res);

    expect_moe_downstream_equal(var, res);
}

// "with_weights" is an option for read/write_weightless() when Constant in our model is not present in the original
// it reads/writes the whole ov::Tensor
TEST(SerializationTest, OVTypes_Tensor_with_weights) {
    using namespace ov::npuw::s11n;

    std::vector<uint8_t> data{0, 1, 2, 3};
    ov::Tensor var(ov::element::u8, ov::Shape({2, 2}), data.data());
    std::vector<ov::Tensor> res;

    std::stringstream ss;

    std::unordered_map<const void*, std::size_t> const_offset;
    const_offset[nullptr] = 0;
    WeightsContext ctx(false, const_offset);

    WeightsContext::ConstsCache consts_cache;
    consts_cache[{0, 0}] = nullptr;
    WeightsContext des_ctx(nullptr, "", consts_cache, {});

    write_weightless(ss, {var}, ctx);
    read_weightless(ss, res, des_ctx);

    EXPECT_EQ(res[0].get_element_type(), ov::element::u8);
    EXPECT_EQ(res[0].get_shape(), ov::Shape({2, 2}));

    std::vector<uint8_t> data_res(4, 0);
    std::memcpy(data_res.data(), res[0].data(), 4);

    EXPECT_EQ(data, data_res);
}

TEST(SerializationTest, OVTypes_Tensor_empty) {
    using namespace ov::npuw::s11n;

    ov::Tensor var;
    ov::Tensor res(ov::element::u8, ov::Shape{1});

    std::stringstream ss;

    write(ss, var);
    read(ss, res);

    EXPECT_FALSE(res);
}

TEST(SerializationTest, OVTypes_Tensor_non_contiguous) {
    using namespace ov::npuw::s11n;

    std::vector<float> storage{1.f, 2.f, 100.f, 3.f, 4.f, 200.f};
    ov::Strides strides{3 * sizeof(float), sizeof(float)};
    ov::Tensor var(ov::element::f32, ov::Shape{2, 2}, storage.data(), strides);
    ASSERT_FALSE(var.is_continuous());

    ov::Tensor res;
    std::stringstream ss;

    write(ss, var);
    read(ss, res);

    ASSERT_TRUE(res.is_continuous());
    std::vector<float> expected_values{1.f, 2.f, 3.f, 4.f};
    ov::Tensor expected(ov::element::f32, ov::Shape{2, 2}, expected_values.data());
    expect_tensors_equal(expected, res);
}

TEST(SerializationTest, OVTypes_Tensor_allocator) {
    using namespace ov::npuw::s11n;

    ov::Tensor var(ov::element::i32, ov::Shape{2, 2});
    std::vector<int32_t> values{1, 2, 3, 4};
    std::memcpy(var.data(), values.data(), var.get_byte_size());

    std::stringstream ss;
    auto output_stream = Stream::writer(ss);
    transfer_tensor(output_stream, var);

    bool allocator_called = false;
    ov::Tensor res;
    auto input_stream = Stream::reader(ss);
    transfer_tensor(input_stream,
                    res,
                    [&](const ov::element::Type& type, const ov::Shape& shape) {
                        allocator_called = true;
                        return ov::Tensor(type, shape);
                    });

    EXPECT_TRUE(allocator_called);
    expect_tensors_equal(var, res);
}

TEST(SerializationTest, OVTypes_Tensor_weightless_bf16_to_f16) {
    using namespace ov::npuw::s11n;

    std::vector<ov::float16> expected_values = {ov::float16(1.0f), ov::float16(-2.5f), ov::float16(3.25f)};
    ov::Tensor var(ov::element::f16, ov::Shape{3}, expected_values.data());

    std::stringstream ss;
    WeightsContext export_ctx(true, {{var.data(), 0}});
    write_weightless(ss, {var}, export_ctx);

    std::filesystem::path file_path = ov::test::utils::generateTestFilePrefix() + "_npuw_bf16_weights.bin";
    {
        std::ofstream os(file_path, std::ios::binary);
        std::vector<ov::bfloat16> bf16_values = {ov::bfloat16(1.0f), ov::bfloat16(-2.5f), ov::bfloat16(3.25f)};
        os.write(reinterpret_cast<const char*>(bf16_values.data()),
                 static_cast<std::streamsize>(bf16_values.size() * sizeof(ov::bfloat16)));
    }

    std::vector<ov::Tensor> res;
    {
        auto mapped = ov::load_mmap_object(file_path);
        ASSERT_NE(mapped, nullptr);
        auto weights = std::make_shared<Weights>(reinterpret_cast<char*>(mapped->data()), mapped->size(), mapped);

        WeightsContext import_ctx(weights, file_path.string(), {}, {{0, var.get_byte_size()}});
        read_weightless(ss, res, import_ctx);
    }  // mapped + weights released here, file handle closed on Windows

    ASSERT_EQ(res.size(), 1);
    expect_tensors_equal(var, res.front());
    std::filesystem::remove(file_path);
}

TEST(SerializationTest, OVTypes_OutputPort_roundtrips_into_parameter_pointer) {
    using namespace ov::npuw::s11n;

    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3});
    param->set_friendly_name("input");
    param->output(0).get_tensor().set_names({"input", "alias"});

    std::shared_ptr<ov::op::v0::Parameter> res;
    std::stringstream ss;
    ov::Output<const ov::Node> output = param->output(0);

    write(ss, output);
    read(ss, res);

    ASSERT_NE(res, nullptr);
    EXPECT_EQ(res->get_element_type(), param->get_element_type());
    EXPECT_EQ(res->get_partial_shape(), param->get_partial_shape());
    EXPECT_EQ(res->output(0).get_tensor().get_names(), param->output(0).get_tensor().get_names());
}

TEST(SerializationTest, OVTypes_OutputPort_roundtrips_into_node_pointer) {
    using namespace ov::npuw::s11n;

    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{2, 4});
    param->set_friendly_name("node_input");
    param->output(0).get_tensor().set_names({"node_input"});

    std::shared_ptr<ov::Node> res;
    std::stringstream ss;
    ov::Output<const ov::Node> output = param->output(0);

    write(ss, output);
    read(ss, res);

    ASSERT_NE(res, nullptr);
    EXPECT_EQ(res->get_friendly_name(), "node_input");
    EXPECT_EQ(res->output(0).get_tensor().get_names(), param->output(0).get_tensor().get_names());
}

TEST(SerializationTest, OVTypes_OutputPort_throws_on_read) {
    using namespace ov::npuw::s11n;

    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1});
    std::stringstream ss;
    ov::Output<const ov::Node> output = param->output(0);
    write(ss, output);

    EXPECT_THROW(read(ss, output), ov::Exception);
}

TEST(SerializationTest, OVTypes_ParameterPointer_throws_on_write) {
    using namespace ov::npuw::s11n;

    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1});
    std::stringstream ss;

    EXPECT_THROW(write(ss, param), ov::Exception);
}

TEST(SerializationTest, OVTypes_NodePointer_throws_on_write) {
    using namespace ov::npuw::s11n;

    std::shared_ptr<ov::Node> node =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1});
    std::stringstream ss;

    EXPECT_THROW(write(ss, node), ov::Exception);
}

TEST(SerializationTest, OVTypes_LazyTensor_uninitialized) {
    using namespace ov::npuw::s11n;

    ov::npuw::weights::LazyTensor var;
    ov::npuw::weights::LazyTensor res;

    std::stringstream ss;

    write(ss, var);
    read(ss, res);

    EXPECT_FALSE(var);
    EXPECT_FALSE(res);
    EXPECT_EQ(var, res);
}

TEST(SerializationTest, OVTypes_LazyTensor_const_roundtrip) {
    using namespace ov::npuw::s11n;

    auto constant = make_weightless_constant<float>(ov::element::f32, ov::Shape{2}, {1.0f, 2.0f}, 0);
    ov::npuw::weights::LazyTensor var(constant);
    ov::npuw::weights::LazyTensor res;

    std::stringstream ss;
    write(ss, var);
    read(ss, res);
    res.read_weight(WeightsContext(nullptr, "", make_consts_cache({constant}), {}));

    expect_lazy_tensor_transform_types_equal(var, res);
    EXPECT_EQ(var.eval_meta().shape, res.eval_meta().shape);
    EXPECT_EQ(var.eval_meta().type, res.eval_meta().type);
    expect_tensors_equal(var.eval(), res.eval());
}

TEST(SerializationTest, OVTypes_LazyTensor_const_with_embedded_weight_roundtrip) {
    using namespace ov::npuw::s11n;

    auto constant = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{2}, std::vector<float>{1.0f, 2.0f});
    ov::npuw::weights::LazyTensor var(constant);
    ov::npuw::weights::LazyTensor res;

    std::stringstream ss;
    write(ss, var);
    read(ss, res);

    expect_lazy_tensor_transform_types_equal(var, res);
    EXPECT_EQ(var.eval_meta().shape, res.eval_meta().shape);
    EXPECT_EQ(var.eval_meta().type, res.eval_meta().type);
    expect_tensors_equal(var.eval(), res.eval());
}

TEST(SerializationTest, OVTypes_LazyTensor_concat_permute_convert_roundtrip) {
    using namespace ov::npuw::s11n;

    auto first = make_weightless_constant<float>(ov::element::f32, ov::Shape{1, 2}, {1.0f, 2.0f}, 0);
    auto second = make_weightless_constant<float>(ov::element::f32, ov::Shape{1, 2}, {3.0f, 4.0f}, first->get_byte_size());
    ov::npuw::weights::LazyTensor concat(std::vector<ov::npuw::weights::LazyTensor>{
                                             ov::npuw::weights::LazyTensor(first),
                                             ov::npuw::weights::LazyTensor(second)},
                                         0);
    auto var = concat.permute({1, 0}).convert(ov::element::f16);
    ov::npuw::weights::LazyTensor res;

    std::stringstream ss;
    write(ss, var);
    read(ss, res);
    res.read_weight(WeightsContext(nullptr, "", make_consts_cache({first, second}), {}));

    expect_lazy_tensor_transform_types_equal(var, res);
    EXPECT_EQ(var.get_hash(), res.get_hash());
    EXPECT_EQ(var.eval_meta().shape, res.eval_meta().shape);
    EXPECT_EQ(var.eval_meta().type, res.eval_meta().type);
}

TEST(SerializationTest, OVTypes_LazyTensor_unpack_roundtrip) {
    using namespace ov::npuw::s11n;

    auto w = make_weightless_constant<uint8_t>(ov::element::u8, ov::Shape{8}, {1, 2, 3, 4, 5, 6, 7, 8}, 0);
    auto s = make_weightless_constant<ov::float16>(ov::element::f16, ov::Shape{8}, {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f}, w->get_byte_size());
    ov::npuw::weights::LazyTensor var(ov::npuw::weights::LazyTensor(w),
                                      ov::npuw::weights::LazyTensor(),
                                      ov::npuw::weights::LazyTensor(s),
                                      ov::element::f16,
                                      ov::Shape{8});
    ov::npuw::weights::LazyTensor res;

    std::stringstream ss;
    write(ss, var);
    read(ss, res);
    res.read_weight(WeightsContext(nullptr, "", make_consts_cache({w, s}), {}));

    expect_lazy_tensor_transform_types_equal(var, res);
    EXPECT_EQ(var.eval_meta().shape, res.eval_meta().shape);
    EXPECT_EQ(var.eval_meta().type, res.eval_meta().type);
}

TEST(SerializationTest, OVTypes_LazyTensor_gather_roundtrip) {
    using namespace ov::npuw::s11n;

    auto w = make_weightless_constant<uint8_t>(ov::element::u8, ov::Shape{16}, {0, 1, 2, 3, 4, 5, 6, 7,
                                                                                8, 9, 10, 11, 12, 13, 14, 15},
                                               0);
    std::vector<uint8_t> indices{0, 1, 2, 3};
    ov::Tensor lut(ov::element::f8e4m3, ov::Shape{4}, indices.data());
    ov::npuw::weights::LazyTensor var(ov::npuw::weights::LazyTensor(w), lut, ov::element::f16, ov::Shape{8});
    ov::npuw::weights::LazyTensor res;

    std::stringstream ss;
    write(ss, var);
    read(ss, res);
    res.read_weight(WeightsContext(nullptr, "", make_consts_cache({w}), {}));

    expect_lazy_tensor_transform_types_equal(var, res);
    EXPECT_EQ(var.eval_meta().shape, res.eval_meta().shape);
    EXPECT_EQ(var.eval_meta().type, res.eval_meta().type);
}

TEST(SerializationTest, OVTypes_WeightsBank_cpu_roundtrip) {
    using namespace ov::npuw::s11n;

    auto first = make_weightless_constant<float>(ov::element::f32, ov::Shape{2}, {1.0f, 2.0f}, 0);
    auto second = make_weightless_constant<float>(ov::element::f32, ov::Shape{2}, {3.0f, 4.0f}, first->get_byte_size());

    ov::npuw::weights::Bank var(nullptr, "CPU", "test-bank");
    const auto uid0 = var.registerLT(ov::npuw::weights::LazyTensor(first), "CPU");
    const auto uid0_dup = var.registerLT(ov::npuw::weights::LazyTensor(first), "CPU");
    const auto uid1 = var.registerLT(ov::npuw::weights::LazyTensor(second), "CPU");
    EXPECT_EQ(uid0, uid0_dup);
    EXPECT_NE(uid0, uid1);

    var.evaluate_and_allocate();

    ov::npuw::weights::Bank res(nullptr, "CPU", "restored-bank");
    std::stringstream ss;

    write(ss, var);
    read(ss, res);

    expect_tensors_equal(var.get(uid0, "CPU"), res.get(uid0, "CPU"));
    expect_tensors_equal(var.get(uid1, "CPU"), res.get(uid1, "CPU"));
}

// TODO: add tests on CompiledModel and LLMCompiledModel once tests have access to any model to test on
