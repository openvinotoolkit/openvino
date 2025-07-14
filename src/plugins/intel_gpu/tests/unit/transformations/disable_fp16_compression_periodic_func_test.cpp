// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <openvino/core/model.hpp>
#include <openvino/pass/manager.hpp>
#include "common_test_utils/ov_test_utils.hpp"
#include <transformations/utils/utils.hpp>

#include <plugin/transformations/disable_f16_comp_for_periodic_funcs.hpp>
#include <transformations/convert_precision.hpp>
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/ceiling.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/roll.hpp"
#include "openvino/core/graph_util.hpp"

using namespace testing;
using namespace ov::intel_gpu;

static bool fuse_type_to_convert(const std::shared_ptr<ov::Node>& node, const precisions_map& precisions) {
    auto convert = ov::as_type_ptr<ov::op::v0::Convert>(node);
    if (!convert)
        return false;
    const auto& from = node->get_output_element_type(0);
    auto it = precisions.find(from);
    if (it == precisions.end())
        return false;
    const auto& to = it->second;
    convert->set_convert_element_type(to);
    return true;
}

static std::string name_mul = "mul_1";
static std::string name_add = "add_1";
static std::string name_sin = "sin";
static std::string name_cos = "cos";

static std::shared_ptr<ov::Model> create_model_with_periodic_func() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ -1, -1, 3 });
    auto constant_1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, { -1 });
    auto unsqueeze_1 = std::make_shared<ov::op::v0::Unsqueeze>(input, constant_1 );

    auto constant_2_compressed = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 1,1,341 }, { 3.14062f });
    auto constant_2 = std::make_shared<ov::op::v0::Convert>(constant_2_compressed, ov::element::f32);

    auto multiply_1 = std::make_shared<ov::op::v1::Multiply>(unsqueeze_1, constant_2);
    multiply_1->set_friendly_name(name_mul);

    auto constant_3_compressed = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 1,1,341 }, { -1.57031f });
    auto constant_3 = std::make_shared<ov::op::v0::Convert>(constant_3_compressed, ov::element::f32);
    auto constant_4 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 4 }, { 0, 1, 3, 2 });

    auto add_1 = std::make_shared<ov::op::v1::Add>(multiply_1, constant_3);
    add_1->set_friendly_name(name_add);
    auto transpose_1 = std::make_shared<ov::op::v1::Transpose>(add_1, constant_4);
    auto reshape_1 = std::make_shared<ov::op::v1::Reshape>(transpose_1, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 3 }, { 0, 0, 1023 }), false);

    auto sin = std::make_shared<ov::op::v0::Sin>(reshape_1);
    sin->set_friendly_name(name_sin);
    auto cos = std::make_shared<ov::op::v0::Cos>(reshape_1);
    cos->set_friendly_name(name_cos);
    auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{sin, cos}, 2);

    return std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{input});
}

TEST(TransformationTests, DisableFP16CompressionForPeriodicFuncsTest) {
    ov::pass::Manager manager;
    manager.register_pass<DisableFP16CompressionForPeriodicFuncs>();

    type_to_fuse_map empty_fuse_map = {};
    const bool keep_precision_sensitive_in_fp32_1 = true;
    const bool convert_input_output_precision = false;
    const bool store_original_precision_as_rt_attribute = true;

    precisions_map fp_convert_precision_map = {
        {ov::element::f64, ov::element::f32},
        {ov::element::f32, ov::element::f16}
    };

    manager.register_pass<ov::pass::ConvertPrecision>(fp_convert_precision_map,
                                                        empty_fuse_map,
                                                        keep_precision_sensitive_in_fp32_1,
                                                        convert_input_output_precision,
                                                        store_original_precision_as_rt_attribute);

    const bool keep_precision_sensitive_in_fp32_2 = true;
    precisions_map int_convert_precision_map{
        {ov::element::i64, ov::element::i32},
        {ov::element::u64, ov::element::i32},
        {ov::element::i16, ov::element::i32},
        {ov::element::u16, ov::element::i32},
        {ov::element::u32, ov::element::i32},
        {ov::element::boolean, ov::element::u8},
        {ov::element::i4, ov::element::i8},
        {ov::element::u4, ov::element::u8},
    };

    // To convert to f16 input to boolean which is converted to u8, add abs + ceiling + clamp before convert.
    type_to_fuse_map type_to_fuse = {{ov::op::v0::Convert::get_type_info_static(), fuse_type_to_convert}};
    manager.register_pass<ov::pass::ConvertPrecision>(int_convert_precision_map,
                                                        type_to_fuse,
                                                        keep_precision_sensitive_in_fp32_2,
                                                        convert_input_output_precision);
    auto func = create_model_with_periodic_func();
    manager.run_passes(func);

    bool success = false;
    for (auto& ops : func->get_ops()) {
        if (ops->get_friendly_name() == name_sin
            || ops->get_friendly_name() == name_cos
            || ops->get_friendly_name() == name_mul
            || ops->get_friendly_name() == name_add) {
            if (!ov::fp16_compression_is_disabled(ops)) {
                success = false;
                break;
            }
            success = true;
        }
    }
    ASSERT_TRUE(success);
}
