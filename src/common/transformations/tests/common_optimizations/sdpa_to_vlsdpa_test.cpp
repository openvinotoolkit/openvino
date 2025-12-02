// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/sdpa_to_vlsdpa.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/core.hpp"
#include "ov_ops/vl_sdpa.hpp"

using namespace std;
using namespace ov;
using namespace ov::opset13;

namespace {
std::shared_ptr<ov::Model> build_model(const string& mask_name) {
    auto q = std::make_shared<Parameter>(element::f32, PartialShape{-1, 8, 32}); /* L,H,S */
    auto k = std::make_shared<Parameter>(element::f32, PartialShape{-1, 8, 32});
    auto v = std::make_shared<Parameter>(element::f32, PartialShape{-1, 8, 32});

    q->set_friendly_name("q");
    k->set_friendly_name("k");
    v->set_friendly_name("v");

    auto transpose_q = std::make_shared<Transpose>(q, Constant::create(element::i64, Shape{3}, {1, 0, 2}));
    auto transpose_k = std::make_shared<Transpose>(k, Constant::create(element::i64, Shape{3}, {1, 0, 2}));
    auto transpose_v = std::make_shared<Transpose>(v, Constant::create(element::i64, Shape{3}, {1, 0, 2}));
    transpose_q->set_friendly_name("transpose_q");
    transpose_k->set_friendly_name("transpose_k");
    transpose_v->set_friendly_name("transpose_v");

    auto mask = std::make_shared<Parameter>(element::f32, PartialShape{1, -1, -1});
    mask->set_friendly_name(mask_name);
    mask->get_output_tensor(0).set_names({mask_name});

    const auto casual = false;

    auto sdpa =
        std::make_shared<opset13::ScaledDotProductAttention>(transpose_q, transpose_k, transpose_v, mask, casual);
    sdpa->set_friendly_name("sdpa");

    auto transpose_o = std::make_shared<Transpose>(sdpa, Constant::create(element::i64, Shape{3}, {1, 0, 2}));
    transpose_o->set_friendly_name("transpose_o");

    return std::make_shared<ov::Model>(OutputVector{transpose_o}, ParameterVector{q, k, v, mask});
}

std::shared_ptr<ov::Model> build_target_model(const string& mask_name) {
    auto q = std::make_shared<Parameter>(element::f32, PartialShape{-1, 8, 32}); /* L,H,S */
    auto k = std::make_shared<Parameter>(element::f32, PartialShape{-1, 8, 32});
    auto v = std::make_shared<Parameter>(element::f32, PartialShape{-1, 8, 32});
    q->set_friendly_name("q");
    k->set_friendly_name("k");
    v->set_friendly_name("v");

    auto transpose_q = std::make_shared<Transpose>(q, Constant::create(element::i64, Shape{3}, {1, 0, 2}));
    auto transpose_k = std::make_shared<Transpose>(k, Constant::create(element::i64, Shape{3}, {1, 0, 2}));
    auto transpose_v = std::make_shared<Transpose>(v, Constant::create(element::i64, Shape{3}, {1, 0, 2}));
    transpose_q->set_friendly_name("transpose_q");
    transpose_k->set_friendly_name("transpose_k");
    transpose_v->set_friendly_name("transpose_v");

    auto cuseq_mask = std::make_shared<Parameter>(element::i32, PartialShape{-1});
    cuseq_mask->set_friendly_name(mask_name);
    cuseq_mask->get_output_tensor(0).set_names({mask_name});

    auto vlsdpa =
        std::make_shared<ov::op::internal::VLSDPA>(OutputVector{transpose_q, transpose_k, transpose_v, cuseq_mask});

    auto transpose_o = std::make_shared<Transpose>(vlsdpa, Constant::create(element::i64, Shape{3}, {1, 0, 2}));
    transpose_o->set_friendly_name("transpose_o");

    return std::make_shared<ov::Model>(OutputVector{transpose_o}, ParameterVector{q, k, v, cuseq_mask});
}
};  // namespace

TEST_F(TransformationTestsF, SDPA2VLSDPAAttentionMaskTest) {
    disable_rt_info_check();
    {
        model = build_model("attention_mask");
        model->set_rt_info("QWenVL", "model_type_hint");  // request_vl_sdpa_transformations
        manager.register_pass<ov::pass::SDPAToVLSDPA>();
    }
    { model_ref = build_target_model("cu_seq_lens"); }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::NAMES);
}

TEST_F(TransformationTestsF, SDPA2VLSDPAWindowAttentionMaskTest) {
    disable_rt_info_check();
    {
        model = build_model("window_attention_mask");
        model->set_rt_info("QWenVL", "model_type_hint");  // request_vl_sdpa_transformations
        manager.register_pass<ov::pass::SDPAToVLSDPA>();
    }
    { model_ref = build_target_model("cu_window_seqlens"); }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::NAMES);
}