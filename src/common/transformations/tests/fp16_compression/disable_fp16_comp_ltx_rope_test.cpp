// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "transformations/fp16_compression/disable_fp16_comp_ltx_rope.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/transpose.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;

namespace {
// Multiply -> Add(Constant) -> Transpose -> Reshape -> Sin/Cos, the decomposed LTX-Video rope angles.
struct RopeChain {
    std::shared_ptr<v0::Parameter> grid;
    std::shared_ptr<v1::Multiply> mul;
    std::shared_ptr<v0::Constant> add_const;
    std::shared_ptr<v1::Add> add;
    std::shared_ptr<v1::Transpose> transpose;
    std::shared_ptr<v1::Reshape> reshape;
    std::shared_ptr<v0::Sin> sin;
    std::shared_ptr<v0::Cos> cos;
    std::shared_ptr<ov::Model> model(const std::string& name) const {
        return std::make_shared<ov::Model>(ov::OutputVector{sin, cos}, ov::ParameterVector{grid}, name);
    }
};

RopeChain make_rope_chain() {
    RopeChain c;
    c.grid = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{1, 8, 1});
    auto freqs = v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 16}, std::vector<float>(16, 1000.0f));
    c.mul = std::make_shared<v1::Multiply>(c.grid, freqs);
    c.add_const = v0::Constant::create(ov::element::f32, ov::Shape{}, {-1.0f});
    c.add = std::make_shared<v1::Add>(c.mul, c.add_const);
    auto order = v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 2, 1});
    c.transpose = std::make_shared<v1::Transpose>(c.add, order);
    auto shape = v0::Constant::create(ov::element::i32, ov::Shape{2}, {1, 128});
    c.reshape = std::make_shared<v1::Reshape>(c.transpose, shape, false);
    c.sin = std::make_shared<v0::Sin>(c.reshape);
    c.cos = std::make_shared<v0::Cos>(c.reshape);
    return c;
}
}  // namespace

// The whole angle chain from the frequency Multiply up to Sin/Cos is kept in f32.
TEST_F(TransformationTestsF, DisableFP16CompForLtxVideoRopeMarksAngleChain) {
    model = make_rope_chain().model("model");

    manager.register_pass<ov::pass::DisableFP16CompForLtxVideoRopePattern>();

    {
        auto c = make_rope_chain();
        for (const auto& node :
             std::vector<std::shared_ptr<ov::Node>>{c.mul, c.add_const, c.add, c.transpose, c.reshape, c.sin, c.cos}) {
            disable_conversion(node, ov::element::f16);
        }
        model_ref = c.model("model_ref");
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
}

// Sin/Cos outside the rope pattern are left untouched, so unrelated subgraphs keep low precision.
TEST_F(TransformationTestsF, DisableFP16CompForLtxVideoRopeSkipsUnrelatedSinCos) {
    auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{1, 8});
    auto sin = std::make_shared<v0::Sin>(x);
    model = std::make_shared<ov::Model>(ov::OutputVector{sin}, ov::ParameterVector{x}, "model");

    manager.register_pass<ov::pass::DisableFP16CompForLtxVideoRopePattern>();
}
