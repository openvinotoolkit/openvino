// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/cpu_opset/common/pass/keep_reduce_prod_i64.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/result.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

using namespace ov::intel_cpu;

namespace {

std::shared_ptr<ov::Node> make_reduce_prod(const ov::Output<ov::Node>& data) {
    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    return std::make_shared<ov::op::v1::ReduceProd>(data, axes, false);
}

std::shared_ptr<ov::Node> keep_i64(const std::shared_ptr<ov::Node>& node) {
    ov::disable_conversion(node, ov::element::i64, ov::element::i32);
    return node;
}

class KeepReduceProdI64Tests : public TransformationTestsF {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<KeepReduceProdI64>();
    }
};

// i32 -> Convert(i64) -> ReduceProd -> Result: both nodes are kept in i64, the graph is unchanged otherwise.
TEST_F(KeepReduceProdI64Tests, ConvertedInputWithResultConsumer) {
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{4});
        auto convert = std::make_shared<ov::op::v0::Convert>(data, ov::element::i64);
        auto prod = make_reduce_prod(convert);
        model = std::make_shared<ov::Model>(ov::OutputVector{prod}, ov::ParameterVector{data});
    }
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{4});
        auto convert = keep_i64(std::make_shared<ov::op::v0::Convert>(data, ov::element::i64));
        auto prod = keep_i64(make_reduce_prod(convert));
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{prod}, ov::ParameterVector{data});
    }
}

// Non-Result consumers read the product through a new, unmarked Convert (later lowered to i32);
// the Result still reads the exact i64 value directly.
TEST_F(KeepReduceProdI64Tests, InternalConsumerGoesThroughConvert) {
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{4});
        auto convert = std::make_shared<ov::op::v0::Convert>(data, ov::element::i64);
        auto prod = make_reduce_prod(convert);
        auto add = std::make_shared<ov::op::v1::Add>(prod, ov::op::v0::Constant::create(ov::element::i64, {}, {1}));
        model = std::make_shared<ov::Model>(ov::OutputVector{prod, add}, ov::ParameterVector{data});
    }
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{4});
        auto convert = keep_i64(std::make_shared<ov::op::v0::Convert>(data, ov::element::i64));
        auto prod = keep_i64(make_reduce_prod(convert));
        auto internal = std::make_shared<ov::op::v0::Convert>(prod, ov::element::i64);
        auto add = std::make_shared<ov::op::v1::Add>(internal, ov::op::v0::Constant::create(ov::element::i64, {}, {1}));
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{prod, add}, ov::ParameterVector{data});
    }
}

// A producer that is not a Convert gets a dedicated i64 Convert inserted in front of ReduceProd.
TEST_F(KeepReduceProdI64Tests, NonConvertProducerGetsInputConvert) {
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{4});
        auto prod = make_reduce_prod(data);
        model = std::make_shared<ov::Model>(ov::OutputVector{prod}, ov::ParameterVector{data});
    }
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{4});
        auto convert = keep_i64(std::make_shared<ov::op::v0::Convert>(data, ov::element::i64));
        auto prod = keep_i64(make_reduce_prod(convert));
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{prod}, ov::ParameterVector{data});
    }
}

// A Convert shared with other consumers is not marked; ReduceProd gets its own Convert instead.
TEST_F(KeepReduceProdI64Tests, SharedConvertIsNotReused) {
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{4});
        auto convert = std::make_shared<ov::op::v0::Convert>(data, ov::element::i64);
        auto prod = make_reduce_prod(convert);
        model = std::make_shared<ov::Model>(ov::OutputVector{prod, convert}, ov::ParameterVector{data});
    }
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{4});
        auto convert = std::make_shared<ov::op::v0::Convert>(data, ov::element::i64);
        auto own_convert = keep_i64(std::make_shared<ov::op::v0::Convert>(convert, ov::element::i64));
        auto prod = keep_i64(make_reduce_prod(own_convert));
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{prod, convert}, ov::ParameterVector{data});
    }
}

TEST_F(KeepReduceProdI64Tests, FloatReduceProdIsNotChanged) {
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4});
    model = std::make_shared<ov::Model>(ov::OutputVector{make_reduce_prod(data)}, ov::ParameterVector{data});
}

TEST_F(KeepReduceProdI64Tests, I32ReduceProdIsNotChanged) {
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{4});
    model = std::make_shared<ov::Model>(ov::OutputVector{make_reduce_prod(data)}, ov::ParameterVector{data});
}

}  // namespace
