// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <map>
#include <vector>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/acos.hpp"
#include "openvino/op/acosh.hpp"
#include "openvino/op/asin.hpp"
#include "openvino/op/asinh.hpp"
#include "openvino/op/atan.hpp"
#include "openvino/op/atanh.hpp"
#include "openvino/op/batch_norm.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/ceiling.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/cosh.hpp"
#include "openvino/op/deformable_convolution.hpp"
#include "openvino/op/depth_to_space.hpp"
#include "openvino/op/dft.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/einsum.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/erf.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/experimental_detectron_generate_proposals.hpp"
#include "openvino/op/experimental_detectron_prior_grid_generator.hpp"
#include "openvino/op/eye.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/gru_sequence.hpp"
#include "openvino/op/hard_sigmoid.hpp"
#include "openvino/op/hsigmoid.hpp"
#include "openvino/op/hswish.hpp"
#include "openvino/op/idft.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/irdft.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/logical_xor.hpp"
#include "openvino/op/lrn.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/matrix_nms.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/mish.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/non_max_suppression.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/op/proposal.hpp"
#include "openvino/op/random_uniform.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/rdft.hpp"
#include "openvino/op/reduce_l1.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/reduce_logical_and.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/region_yolo.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/rnn_sequence.hpp"
#include "openvino/op/roi_align.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/selu.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/sign.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/sinh.hpp"
#include "openvino/op/softplus.hpp"
#include "openvino/op/softsign.hpp"
#include "openvino/op/space_to_batch.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/tan.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"

namespace ov {
namespace test {
namespace utils {

static ov::test::utils::InputGenerateData get_range_by_type(ov::element::Type temp_type) {
    double min_start = 0 - (int32_t)round(testing::internal::Random::kMaxRange / 2);
    uint32_t max_range = testing::internal::Random::kMaxRange - 1;

    ov::test::utils::InputGenerateData inData(min_start, max_range);
#define CASE_T(X)                                                                             \
    case X: {                                                                                 \
        auto lowest = std::numeric_limits<element_type_traits<X>::value_type>::lowest();      \
        auto max = std::numeric_limits<element_type_traits<X>::value_type>::max();            \
        double tmp_range = static_cast<double>(max) - static_cast<double>(lowest);            \
        if (tmp_range < testing::internal::Random::kMaxRange) {                               \
            inData.start_from = lowest;                                                       \
            inData.range = (uint32_t)round(tmp_range);                                        \
        } else {                                                                              \
            inData.range = testing::internal::Random::kMaxRange - 1;                          \
            inData.start_from = lowest > min_start ? static_cast<double>(lowest) : min_start; \
        }                                                                                     \
        break;                                                                                \
    }

    switch (temp_type) {
    case (ov::element::Type_t::undefined): {
        inData.start_from = min_start;
        inData.range = max_range;
        break;
    }
    case (ov::element::Type_t::dynamic): {
        inData.start_from = min_start;
        inData.range = max_range;
        break;
    }
    case (ov::element::Type_t::boolean): {
        inData.start_from = 0;
        inData.range = 2;
        break;
    }
    case (ov::element::Type_t::bf16): {
        ov::bfloat16 lowest_tmp = std::numeric_limits<ov::bfloat16>::lowest();
        ov::bfloat16 max_tmp = std::numeric_limits<ov::bfloat16>::max();

        double lowest = 0 - static_cast<double>(lowest_tmp.to_bits());
        double max = max_tmp.to_bits();

        double tmp_range = max - lowest;
        if (tmp_range < testing::internal::Random::kMaxRange) {
            inData.start_from = lowest;
            inData.range = (uint32_t)round(tmp_range);
        } else {
            inData.start_from = lowest > min_start ? lowest : min_start;
            inData.range = testing::internal::Random::kMaxRange - 1;
        }

        break;
    }
    case ov::element::Type_t::f16: {
        ov::float16 lowest_tmp = std::numeric_limits<ov::float16>::lowest();
        ov::float16 max_tmp = std::numeric_limits<ov::float16>::max();

        double lowest = 0 - static_cast<double>(lowest_tmp.to_bits());
        double max = max_tmp.to_bits();

        double tmp_range = max - lowest;
        if (tmp_range < testing::internal::Random::kMaxRange) {
            inData.start_from = lowest;
            inData.range = (uint32_t)round(tmp_range);
        } else {
            inData.start_from = lowest > min_start ? lowest : min_start;
            inData.range = testing::internal::Random::kMaxRange - 1;
        }

        break;
    }
    case ov::element::Type_t::f8e4m3: {
        ov::float8_e4m3 lowest_tmp = std::numeric_limits<ov::float8_e4m3>::lowest();
        ov::float8_e4m3 max_tmp = std::numeric_limits<ov::float8_e4m3>::max();

        double lowest = 0 - static_cast<double>(lowest_tmp.to_bits());
        double max = max_tmp.to_bits();

        double tmp_range = max - lowest;
        if (tmp_range < testing::internal::Random::kMaxRange) {
            inData.start_from = lowest;
            inData.range = (uint32_t)round(tmp_range);
        } else {
            inData.start_from = lowest > min_start ? lowest : min_start;
            inData.range = testing::internal::Random::kMaxRange - 1;
        }

        break;
    }
    case ov::element::Type_t::f8e5m2: {
        ov::float8_e5m2 lowest_tmp = std::numeric_limits<ov::float8_e5m2>::lowest();
        ov::float8_e5m2 max_tmp = std::numeric_limits<ov::float8_e5m2>::max();

        double lowest = 0 - static_cast<double>(lowest_tmp.to_bits());
        double max = max_tmp.to_bits();

        double tmp_range = max - lowest;
        if (tmp_range < testing::internal::Random::kMaxRange) {
            inData.start_from = lowest;
            inData.range = (uint32_t)round(tmp_range);
        } else {
            inData.start_from = lowest > min_start ? lowest : min_start;
            inData.range = testing::internal::Random::kMaxRange - 1;
        }

        break;
    }
    case ov::element::Type_t::string: {
        auto lowest = std::numeric_limits<char>::lowest();
        auto max = std::numeric_limits<char>::max();

        double tmp_range = static_cast<double>(max) - static_cast<double>(lowest);
        if (tmp_range < testing::internal::Random::kMaxRange) {
            inData.start_from = lowest;
            inData.range = (uint32_t)round(tmp_range);
        } else {
            inData.start_from = lowest > min_start ? lowest : min_start;
            inData.range = testing::internal::Random::kMaxRange - 1;
        }

        break;
    }
        CASE_T(ov::element::Type_t::f32)
        CASE_T(ov::element::Type_t::f64)
        CASE_T(ov::element::Type_t::i4)
        CASE_T(ov::element::Type_t::i8)
        CASE_T(ov::element::Type_t::i16)
        CASE_T(ov::element::Type_t::i32)
        CASE_T(ov::element::Type_t::i64)
        CASE_T(ov::element::Type_t::u1)
        CASE_T(ov::element::Type_t::u2)
        CASE_T(ov::element::Type_t::u3)
        CASE_T(ov::element::Type_t::u4)
        CASE_T(ov::element::Type_t::u6)
        CASE_T(ov::element::Type_t::u8)
        CASE_T(ov::element::Type_t::nf4)
        CASE_T(ov::element::Type_t::u16)
        CASE_T(ov::element::Type_t::u32)
        CASE_T(ov::element::Type_t::u64)
        break;
    }

    return inData;
}

struct RangeByType {
    std::map<ov::element::Type, ov::test::utils::InputGenerateData> data;

    RangeByType() {
        for (auto& type : ov::element::Type::get_known_types()) {
            data[*type] = get_range_by_type(*type);
        }
    }

    ov::test::utils::InputGenerateData get_range(ov::element::Type type) {
        if (data.count(type) > 0) {
            return data.at(type);
        } else {
            throw std::runtime_error("Couln't find Type in typeMap: " + type.to_string());
        }
    }
};

static RangeByType rangeByType;

struct Range {
    std::vector<ov::test::utils::InputGenerateData> int_port_ranges;
    std::vector<ov::test::utils::InputGenerateData> real_port_ranges;

    Range(const std::vector<ov::test::utils::InputGenerateData>& int_ranges = {},
          const std::vector<ov::test::utils::InputGenerateData>& real_ranges = {})
        : int_port_ranges(int_ranges),
          real_port_ranges(real_ranges) {
        size_t max_known_port = std::max(real_port_ranges.size(), int_port_ranges.size());
        max_known_port = std::max(static_cast<int>(max_known_port), 1);
        for (size_t port = 0; port < max_known_port; port++) {
            std::map<ov::element::Type, ov::test::utils::InputGenerateData> type_map;
            for (auto& type : ov::element::Type::get_known_types()) {
                ov::test::utils::InputGenerateData new_range = rangeByType.get_range(*type);
                if (type->is_real() && port < real_port_ranges.size()) {
                    new_range.correct_range(real_port_ranges.at(port));
                    new_range.input_attribute = real_port_ranges.at(port).input_attribute;
                } else if (type->is_integral() && port < int_port_ranges.size()) {
                    new_range.correct_range(int_port_ranges.at(port));
                    new_range.input_attribute = int_port_ranges.at(port).input_attribute;
                }
                type_map[*type] = new_range;
            }
            data.push_back(type_map);
        }
    }

    std::vector<std::map<ov::element::Type, ov::test::utils::InputGenerateData>> data;

    ov::test::utils::InputGenerateData get_data(size_t port, ov::element::Type type) {
        if (port < data.size()) {
            return data.at(port).at(type);
        } else {
            return data.at(0).at(type);
        }
    }
};

static std::map<ov::NodeTypeInfo, Range> inputRanges = {
    {ov::op::v0::Erf::get_type_info_static(), Range({{-3, 6}}, {{-3, 6, 10}})},
    {ov::op::v1::Divide::get_type_info_static(), Range({{101, 100}}, {{2, 2, 128}})},
    {ov::op::v1::FloorMod::get_type_info_static(), Range({{2, 4}}, {{2, 2, 128}})},
    {ov::op::v1::Mod::get_type_info_static(), Range({{2, 4}}, {{2, 2, 128}})},
    {ov::op::v1::ReduceMax::get_type_info_static(), Range({{0, 5}}, {{-5, 5, 1000}})},
    {ov::op::v1::ReduceMean::get_type_info_static(), Range({{0, 5, 1000}}, {{0, 5, 1000}})},
    {ov::op::v1::ReduceMin::get_type_info_static(), Range({{0, 5}}, {{0, 5, 1000}})},
    {ov::op::v1::ReduceProd::get_type_info_static(), Range({{0, 5}}, {{0, 5, 1000}})},
    {ov::op::v1::ReduceSum::get_type_info_static(), Range({{0, 5}}, {{0, 5, 1000}})},
    {ov::op::v1::ReduceSum::get_type_info_static(), Range({{0, 5}}, {{0, 5, 1000}})},
    {ov::op::v1::ReduceSum::get_type_info_static(), Range({{0, 5}}, {{0, 5, 1000}})},
    {ov::op::v1::Power::get_type_info_static(), Range({{2, 4}}, {{2, 2, 128}})},
    {ov::op::v4::Proposal::get_type_info_static(), Range({{0, 255, 1, 8234231}}, {{0, 1, 1000, 8234231}})},
    {ov::op::v4::ReduceL1::get_type_info_static(), Range({{0, 5}}, {{0, 5, 1000}})},
    {ov::op::v4::ReduceL2::get_type_info_static(), Range({{0, 5}}, {{0, 5, 1000}})},
    {ov::op::v7::DFT::get_type_info_static(), Range({{0, 1}}, {{0, 1, 1000000}})},
    {ov::op::v9::RDFT::get_type_info_static(), Range({{0, 1}}, {{0, 1, 1000000}})},
    {ov::op::v1::LogicalAnd::get_type_info_static(), Range({{0, 2}}, {{0, 2, 1}})},
    {ov::op::v1::LogicalOr::get_type_info_static(), Range({{0, 2}}, {{0, 2, 1}})},
    {ov::op::v1::LogicalNot::get_type_info_static(), Range({{0, 2}}, {{0, 2, 1}})},
    {ov::op::v1::LogicalXor::get_type_info_static(), Range({{0, 2}}, {{0, 2, 1}})},
    {ov::op::v7::IDFT::get_type_info_static(), Range({{0, 1}}, {{0, 1, 1000000}})},
    {ov::op::v9::IRDFT::get_type_info_static(), Range({{0, 1}}, {{0, 1, 1000000}})},
    {ov::op::v0::Sigmoid::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Tanh::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Relu::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::PRelu::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Exp::get_type_info_static(), Range({{0, 15}}, {{-10, 20, 32768}})},
    {ov::op::v0::Log::get_type_info_static(), Range({{0, 15}}, {{1, 20, 32768}})},
    {ov::op::v0::Sign::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Abs::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Clamp::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Negative::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Acos::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v3::Acosh::get_type_info_static(), Range({{1, 15}}, {{1, 200, 32768}})},
    {ov::op::v0::Asin::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v3::Asinh::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Atan::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v3::Atanh::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Cos::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Cosh::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Floor::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Sin::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Sinh::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Sqrt::get_type_info_static(), Range({{0, 15}}, {{1, 20, 32768}})},
    {ov::op::v0::Tan::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Elu::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Erf::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::HardSigmoid::get_type_info_static(),
     Range({{0, 15}}, {{-1, 2, 32768}, {0.2, 0, 1, 1, true}, {0.5, 0, 1, 1, true}})},
    {ov::op::v0::Selu::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Sigmoid::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Tanh::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Relu::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Exp::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Log::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Sign::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Abs::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Gelu::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Ceiling::get_type_info_static(), Range({{0, 15}}, {{-1000, 2000, 32768}})},
    {ov::op::v4::Mish::get_type_info_static(), Range({{0, 15}}, {{-10, 60, 32768}})},
    {ov::op::v4::HSwish::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v4::SoftPlus::get_type_info_static(), Range({{0, 15}}, {{-100, 200, 32768}})},
    {ov::op::v4::Swish::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v5::HSigmoid::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v5::Round::get_type_info_static(), Range({{0, 15}}, {{-10, 20, 4}})},
    {ov::op::v7::Gelu::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v8::MaxPool::get_type_info_static(), Range({{0, 10, 1, 1}}, {{0, 10, 1, 1}})},
    {ov::op::v9::SoftSign::get_type_info_static(), Range({{0, 15}}, {{-100, 200, 32768}})},
    // new temp
    {ov::op::v1::Convolution::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v1::ConvolutionBackpropData::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v1::GroupConvolution::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v1::GroupConvolutionBackpropData::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v12::ScatterElementsUpdate::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v3::ScatterUpdate::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v0::Unsqueeze::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v0::RegionYolo::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v0::MatMul::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v11::Interpolate::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v0::LRN::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v1::Pad::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v3::Broadcast::get_type_info_static(), Range({{0, 2000}}, {{0, 2000, 32768}})},
    {ov::op::v5::NonMaxSuppression::get_type_info_static(), Range({{0, 15}, {0, 1, 1000, 1, true}},
                                                                  {{0, 8, 32}, {0, 1, 1000, 1, true}})},
    {ov::op::v9::NonMaxSuppression::get_type_info_static(), Range({{0, 15}, {0, 1, 1000, 1, true}},
                                                                  {{0, 8, 32}, {0, 1, 1000, 1, true}})},
    {ov::op::v8::MatrixNms::get_type_info_static(), Range({{0, 15}, {0, 1, 1000, 1, true}},
                                                          {{0, 8, 32}, {0, 1, 1000, 1, true}})},
    {ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage::get_type_info_static(), Range({{1, 0, 1, 1}}, {{1, 0, 1, 1}})},
    {ov::op::v6::ExperimentalDetectronPriorGridGenerator::get_type_info_static(), Range({{0, 0, 1}},
                                                                                        {{-100, 200, 2, 1}, {0, 0, 1, 1, true}, {0, 0, 1, 1, true}})},
    {ov::op::v8::DeformableConvolution::get_type_info_static(), Range({{0, 15}, {0, 2, 10, 1, true}, {0, 1, 20, 1, true}},
                                                                      {{0, 8, 32}, {0, 2, 10, 1, true}, {0, 1, 20, 1, true}})},
    {ov::op::v5::GRUSequence::get_type_info_static(), Range({{0, 15}, {0, 15}, {0, 10, 1, 1, true}}, {{0, 8, 32}})},
    {ov::op::v5::BatchNormInference::get_type_info_static(), Range({{0, 3}}, {{0, 3, 1}})},
    {ov::op::v5::RNNSequence::get_type_info_static(),  Range({{0, 15}, {0, 15}, {0, 10, 1, 1, true}},
                                                             {{0, 8, 32}, {0, 8, 32}, {0, 10, 1, 1, true}})},
    {ov::op::v1::LogicalAnd::get_type_info_static(), Range({{0, 2}}, {{0, 2}})},
    {ov::op::v1::LogicalNot::get_type_info_static(), Range({{0, 2}}, {{0, 2}})},
    {ov::op::v1::LogicalOr::get_type_info_static(), Range({{0, 2}}, {{0, 2}})},
    {ov::op::v1::LogicalXor::get_type_info_static(), Range({{0, 2}}, {{0, 2}})},
    {ov::op::v1::ReduceLogicalAnd::get_type_info_static(), Range({{0, 2}}, {{0, 2}})},
    {ov::op::v1::ReduceLogicalOr::get_type_info_static(), Range({{0, 2}}, {{0, 2}})},
    {ov::op::v1::Reshape::get_type_info_static(), Range({{-1000, 2000}, {0, 256, 1, 1, true}}, {{-100, 200, 32768}})},
    {ov::op::v4::Interpolate::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v3::TopK::get_type_info_static(), Range({{-1000, 2000}, {0, 1000, 1, 1, true}}, {{-1000, 2000, 32768}})},
    {ov::op::v11::TopK::get_type_info_static(), Range({{-1000, 2000}, {0, 1000, 1, 1, true}}, {{-1000, 2000, 32768}})},
    {ov::op::v4::Range::get_type_info_static(), Range({{0, 15}, {1, 1000, 1, 1, true}},
                                                      {{-1000, 2000, 32768}, {1, 1000, 1, 1, true}})},
    {ov::op::v11::Interpolate::get_type_info_static(), Range({{0, 15}}, {{-1000, 2000, 32768}})},
    {ov::op::v9::ROIAlign::get_type_info_static(), Range({{0, 15}, {0, 1000, 1, 1, true}, {0, 1000, 1, 1, true}},
                                                         {{-1000, 2000, 32768}, {0, 1000, 1, 1, true}, {0, 1000, 1, 1, true}})},
    {ov::op::v0::Convert::get_type_info_static(), Range({{0, 1000}}, {{-100, 200, 32768}})},
    {ov::op::v0::FakeQuantize::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v0::FakeQuantize::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v1::Select::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v1::Multiply::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v1::StridedSlice::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v5::LSTMSequence::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v1::VariadicSplit::get_type_info_static(), Range({{0, 10}}, {{0, 8, 32}})},
    {ov::op::v1::Subtract::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v1::SpaceToBatch::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v8::GatherND::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v8::Gather::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v0::DepthToSpace::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v7::Einsum::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v8::RandomUniform::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v9::Eye::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
};

class ModelRange {
    // key for map calculated in get_range_id and contais [Parameter Name]_[parameter type]
    std::map<std::string, std::shared_ptr<ov::test::utils::InputGenerateData>> node_ranges;

public:
    void find_mode_ranges(const std::shared_ptr<ov::Model>& function);
    std::string get_range_id(const std::shared_ptr<ov::Node>& node);
    ov::Tensor generate_input(std::shared_ptr<ov::Node> node, size_t port, const ov::Shape& targetShape);

    const std::shared_ptr<ov::test::utils::InputGenerateData> get_range_for_param(
        const std::shared_ptr<ov::Node>& node);
};

}  // namespace utils
}  // namespace test
}  // namespace ov
