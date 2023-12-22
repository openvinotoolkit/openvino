// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <vector>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/node.hpp"

#include "openvino/op/proposal.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/erf.hpp"
#include "openvino/op/non_max_suppression.hpp"
#include "openvino/op/reduce_l1.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/dft.hpp"
#include "openvino/op/idft.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/logical_xor.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/acos.hpp"
#include "openvino/op/acosh.hpp"
#include "openvino/op/asin.hpp"
#include "openvino/op/asinh.hpp"
#include "openvino/op/atan.hpp"
#include "openvino/op/atanh.hpp"
#include "openvino/op/ceiling.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/cosh.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/erf.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/hard_sigmoid.hpp"
#include "openvino/op/hsigmoid.hpp"
#include "openvino/op/hswish.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/mish.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/selu.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/sign.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/sinh.hpp"
#include "openvino/op/softplus.hpp"
#include "openvino/op/softsign.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/tan.hpp"
#include "openvino/op/tanh.hpp"

namespace ov {
namespace test {
namespace utils {

static std::map<ov::NodeTypeInfo, std::vector<std::vector<ov::test::utils::InputGenerateData>>> inputRanges = {
        // NodeTypeInfo: {IntRanges{}, RealRanges{}} (Ranges are used by generate<ov::Node>)
        { ov::op::v0::Erf::get_type_info_static(), {{{-3, 6}}, {{-3, 6, 10}}} },
        { ov::op::v1::Divide::get_type_info_static(), {{{101, 100}}, {{2, 2, 128}}} },
        { ov::op::v1::FloorMod::get_type_info_static(), {{{2, 4}}, {{2, 2, 128}}} },
        { ov::op::v1::Mod::get_type_info_static(), {{{2, 4}}, {{2, 2, 128}}} },
        { ov::op::v1::ReduceMax::get_type_info_static(), {{{0, 5}}, {{-5, 5, 1000}}} },
        { ov::op::v1::ReduceMean::get_type_info_static(), {{{0, 5, 1000}}, {{0, 5, 1000}}} },
        { ov::op::v1::ReduceMin::get_type_info_static(), {{{0, 5}}, {{0, 5, 1000}}} },
        { ov::op::v1::ReduceProd::get_type_info_static(), {{{0, 5}}, {{0, 5, 1000}}} },
        { ov::op::v1::ReduceSum::get_type_info_static(), {{{0, 5}}, {{0, 5, 1000}}} },
        { ov::op::v1::ReduceSum::get_type_info_static(), {{{0, 5}}, {{0, 5, 1000}}} },
        { ov::op::v1::ReduceSum::get_type_info_static(), {{{0, 5}}, {{0, 5, 1000}}} },
        { ov::op::v1::Power::get_type_info_static(), {{{2, 4}}, {{2, 2, 128}}} },
        { ov::op::v4::Proposal::get_type_info_static(), {{{0, 1, 1000, 8234231}}, {{0, 1, 1000, 8234231}}} },
        { ov::op::v4::ReduceL1::get_type_info_static(), {{{0, 5}}, {{0, 5, 1000}}} },
        { ov::op::v4::ReduceL2::get_type_info_static(), {{{0, 5}}, {{0, 5, 1000}}} },
        { ov::op::v7::DFT::get_type_info_static(), {{{0, 1}}, {{0, 1, 1000000}}} },
        { ov::op::v1::LogicalAnd::get_type_info_static(), {{{0, 2}}, {{0, 2, 1}}} },
        { ov::op::v1::LogicalOr::get_type_info_static(), {{{0, 2}}, {{0, 2, 1}}} },
        { ov::op::v1::LogicalNot::get_type_info_static(), {{{0, 2}}, {{0, 2, 1}}} },
        { ov::op::v1::LogicalXor::get_type_info_static(), {{{0, 2}}, {{0, 2, 1}}} },
        { ov::op::v7::IDFT::get_type_info_static(), {{{0, 1}}, {{0, 1, 1000000}}} },
        { ov::op::v0::Sigmoid::get_type_info_static(), {{{0, 15}}, {{-1, 2, 32768}}} },
        { ov::op::v0::Tanh::get_type_info_static(), {{{0, 15}}, {{-1, 2, 32768}}} },
        { ov::op::v0::Relu::get_type_info_static(), {{{0, 15}}, {{-1, 2, 32768}}} },
        { ov::op::v0::PRelu::get_type_info_static(), {{{0, 15}}, {{-1, 2, 32768}}} },
        { ov::op::v0::Exp::get_type_info_static(), {{{0, 15}}, {{-10, 20, 32768}}} },
        { ov::op::v0::Log::get_type_info_static(), {{{0, 15}}, {{1, 20, 32768}}} },
        { ov::op::v0::Sign::get_type_info_static(), {{{0, 15}}, {{-1, 2, 32768}}} },
        { ov::op::v0::Abs::get_type_info_static(), {{{0, 15}}, {{-1, 2, 32768}}} },
        { ov::op::v0::Clamp::get_type_info_static(), {{{0, 15}}, {{-1, 2, 32768}}} },
        { ov::op::v0::Negative::get_type_info_static(), {{{0, 15}}, {{-1, 2, 32768}}} },
        { ov::op::v0::Acos::get_type_info_static(), {{{0, 15}}, {{-1, 2, 32768}}} },
        { ov::op::v3::Acosh::get_type_info_static(), {{{1, 15}}, {{1, 200, 32768}}} },
        { ov::op::v0::Asin::get_type_info_static(), {{{0, 15}}, {{-1, 2, 32768}}} },
        { ov::op::v3::Asinh::get_type_info_static(), {{{0, 15}}, {{-1, 2, 32768}}} },
        { ov::op::v0::Atan::get_type_info_static(), {{{0, 15}}, {{-1, 2, 32768}}} },
        { ov::op::v3::Atanh::get_type_info_static(), {{{0, 15}}, {{-1, 2, 32768}}} },
        { ov::op::v0::Cos::get_type_info_static(), {{{0, 15}}, {{-1, 2, 32768}}} },
        { ov::op::v0::Cosh::get_type_info_static(), {{{0, 15}}, {{-1, 2, 32768}}} },
        { ov::op::v0::Floor::get_type_info_static(), {{{0, 15}}, {{-1, 2, 32768}}} },
        { ov::op::v0::Sin::get_type_info_static(), {{{0, 15}}, {{-1, 2, 32768}}} },
        { ov::op::v0::Sinh::get_type_info_static(), {{{0, 15}}, {{-1, 2, 32768}}} },
        { ov::op::v0::Sqrt::get_type_info_static(), {{{0, 15}}, {{1, 20, 32768}}} },
        { ov::op::v0::Tan::get_type_info_static(), {{{0, 15}}, {{-1, 2, 32768}}} },
        { ov::op::v0::Elu::get_type_info_static(), {{{0, 15}}, {{-1, 2, 32768}}} },
        { ov::op::v0::Erf::get_type_info_static(), {{{0, 15}}, {{-1, 2, 32768}}} },
        { ov::op::v0::HardSigmoid::get_type_info_static(), {{{0, 15}}, {{-1, 2, 32768}}} },
        { ov::op::v0::Selu::get_type_info_static(), {{{0, 15}}, {{-1, 2, 32768}}} },
        { ov::op::v0::Sigmoid::get_type_info_static(), {{{0, 15}}, {{-1, 2, 32768}}} },
        { ov::op::v0::Tanh::get_type_info_static(), {{{0, 15}}, {{-1, 2, 32768}}} },
        { ov::op::v0::Relu::get_type_info_static(), {{{0, 15}}, {{-1, 2, 32768}}} },
        { ov::op::v0::Exp::get_type_info_static(), {{{0, 15}}, {{-1, 2, 32768}}} },
        { ov::op::v0::Log::get_type_info_static(), {{{0, 15}}, {{-1, 2, 32768}}} },
        { ov::op::v0::Sign::get_type_info_static(), {{{0, 15}}, {{-1, 2, 32768}}} },
        { ov::op::v0::Abs::get_type_info_static(), {{{0, 15}}, {{-1, 2, 32768}}} },
        { ov::op::v0::Gelu::get_type_info_static(), {{{0, 15}}, {{-1, 2, 32768}}} },
        { ov::op::v0::Ceiling::get_type_info_static(), {{{0, 15}}, {{-1000, 2000, 32768}}} },
        { ov::op::v4::Mish::get_type_info_static(), {{{0, 15}}, {{-10, 60, 32768}}} },
        { ov::op::v4::HSwish::get_type_info_static(), {{{0, 15}}, {{-1, 2, 32768}}} },
        { ov::op::v4::SoftPlus::get_type_info_static(), {{{0, 15}}, {{-100, 200, 32768}}} },
        { ov::op::v4::Swish::get_type_info_static(), {{{0, 15}}, {{-1, 2, 32768}}} },
        { ov::op::v5::HSigmoid::get_type_info_static(), {{{0, 15}}, {{-1, 2, 32768}}} },
        { ov::op::v5::Round::get_type_info_static(), {{{0, 15}}, {{-10, 20, 4}}} },
        { ov::op::v7::Gelu::get_type_info_static(), {{{0, 15}}, {{-1, 2, 32768}}} },
        { ov::op::v9::SoftSign::get_type_info_static(), {{{0, 15}}, {{-100, 200, 32768}}} },
};

} // namespace utils
} // namespace test
} // namespace ov
