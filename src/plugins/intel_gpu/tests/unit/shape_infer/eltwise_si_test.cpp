// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "eltwise_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;
using namespace ov::op;
using namespace ov;

namespace shape_infer_tests {

struct eltwise_test_params {
    cldnn::layout input1_layout;
    cldnn::layout input2_layout;
    eltwise_mode mode;
    AutoBroadcastSpec auto_broadcast_spec;
    cldnn::layout expected_layout;
    std::vector<tensor> stride;
};

class eltwise_si_test : public testing::TestWithParam<eltwise_test_params> { };

TEST_P(eltwise_si_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto input1_prim = std::make_shared<input_layout>("input1", p.input1_layout);
    auto input2_prim = std::make_shared<input_layout>("input2", p.input2_layout);
    auto eltwise_prim = std::make_shared<eltwise>("output", input_info("input1"), input_info("input2"), p.stride, p.mode, p.auto_broadcast_spec);

    cldnn::program prog(engine);

    auto& input1_node = prog.get_or_create(input1_prim);
    auto& input2_node = prog.get_or_create(input2_prim);
    auto& eltwise_node = prog.get_or_create(eltwise_prim);
    program_wrapper::add_connection(prog, input1_node, eltwise_node);
    program_wrapper::add_connection(prog, input2_node, eltwise_node);
    auto res = eltwise_inst::calc_output_layouts<ov::PartialShape>(eltwise_node, *eltwise_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

TEST_P(eltwise_si_test, shape_infer_const_data) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto const_data = engine.allocate_memory(p.input2_layout);

    auto input1_prim = std::make_shared<input_layout>("input1", p.input1_layout);
    auto const_data_prim = std::make_shared<data>("const_data", const_data);
    auto eltwise_prim = std::make_shared<eltwise>("output", input_info("input1"), input_info("const_data"), p.stride, p.mode, p.auto_broadcast_spec);

    cldnn::program prog(engine);

    auto& input1_node = prog.get_or_create(input1_prim);
    auto& const_data_node = prog.get_or_create(const_data_prim);
    auto& eltwise_node = prog.get_or_create(eltwise_prim);
    program_wrapper::add_connection(prog, input1_node, eltwise_node);
    program_wrapper::add_connection(prog, const_data_node, eltwise_node);
    auto res = eltwise_inst::calc_output_layouts<ov::PartialShape>(eltwise_node, *eltwise_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, eltwise_si_test,
    testing::ValuesIn(std::vector<eltwise_test_params>{
    {{{2, 1, 5}, data_types::f32, format::bfyx},                {{2, 1, 5}, data_types::f32, format::bfyx},                 eltwise_mode::sum,      {AutoBroadcastType::NONE},      {{2, 1, 5}, data_types::f32, format::bfyx},                     {}},
    {{{2, 1, 5}, data_types::f32, format::bfyx},                {{1, 4, 1}, data_types::f32, format::bfyx},                 eltwise_mode::sum,      {AutoBroadcastType::NUMPY},     {{2, 4, 5}, data_types::f32, format::bfyx},                     {}},
    {{{1, 5, 1}, data_types::f32, format::bfyx},                {{5, 2, 1, 3}, data_types::f32, format::bfyx},              eltwise_mode::sum,      {AutoBroadcastType::NUMPY},     {{5, 2, 5, 3}, data_types::f32, format::bfyx},                  {}},
    {{{2, 3, 4, 5}, data_types::f32, format::bfyx},             {{4, 5}, data_types::f32, format::bfyx},                    eltwise_mode::sum,      {AutoBroadcastType::PDPD, -1},  {{2, 3, 4, 5}, data_types::f32, format::bfyx},                  {}},
    {{{2, 3, 4, 5}, data_types::f32, format::bfyx},             {{2, 3}, data_types::f32, format::bfyx},                    eltwise_mode::sum,      {AutoBroadcastType::PDPD, 0},   {{2, 3, 4, 5}, data_types::f32, format::bfyx},                  {}},
    {{{2, 3, 4, 5}, data_types::f32, format::bfyx},             {{3}, data_types::f32, format::bfyx},                       eltwise_mode::sum,      {AutoBroadcastType::PDPD, 1},   {{2, 3, 4, 5}, data_types::f32, format::bfyx},                  {}},
    {{{2, 3, 4, 5}, data_types::f32, format::bfyx},             {{5}, data_types::f32, format::bfyx},                       eltwise_mode::sum,      {AutoBroadcastType::NUMPY},     {{2, 3, 4, 5}, data_types::f32, format::bfyx},                  {}},
    // test for dynamic shape
    {{{1, 5, 1}, data_types::f32, format::bfyx},                {{5, 2, 1, 3}, data_types::f32, format::bfyx},              eltwise_mode::sum,      {AutoBroadcastType::NUMPY},     {{5, 2, 5, 3}, data_types::f32, format::bfyx},                  {}},
    {{{2, -1, 5}, data_types::f32, format::bfyx},               {{1, 4, 1}, data_types::f32, format::bfyx},                 eltwise_mode::sum,      {AutoBroadcastType::NUMPY},     {{2, 4, 5}, data_types::f32, format::bfyx},                     {}},
    {{PartialShape::dynamic(3), data_types::f32, format::bfyx}, {{1, 4, 1}, data_types::f32, format::bfyx},                 eltwise_mode::sum,      {AutoBroadcastType::NUMPY},     {{-1, 4, -1}, data_types::f32, format::bfyx},                   {}},
    {{PartialShape::dynamic(3), data_types::f32, format::bfyx}, {{2, 1, 5}, data_types::f32, format::bfyx},                 eltwise_mode::sum,      {AutoBroadcastType::NUMPY},     {{2, -1, 5}, data_types::f32, format::bfyx},                    {}},
    {{PartialShape::dynamic(3), data_types::f32, format::bfyx}, {{1, 4, 1}, data_types::f32, format::bfyx},                 eltwise_mode::sum,      {AutoBroadcastType::PDPD},      {{-1, 4, -1}, data_types::f32, format::bfyx},                   {}},
    {{{-1, -1, 1024, 512}, data_types::f32, format::bfyx},      {{1,   1, 512}, data_types::f32, format::bfyx},             eltwise_mode::sum,      {AutoBroadcastType::NUMPY},     {{-1,-1,1024,512}, data_types::f32, format::bfyx},              {}},
    {{{-1, -1, 768}, data_types::f32, format::bfyx},            {{768}, data_types::f32, format::bfyx},                     eltwise_mode::sum,      {AutoBroadcastType::NUMPY},     {{-1,-1,768}, data_types::f32, format::bfyx},                   {}},
    // test for output data type of logic and comparison operations
    {{{2, 3, 4, 5}, data_types::f32, format::bfyx},             {{5}, data_types::f32, format::bfyx},                       eltwise_mode::eq,       {AutoBroadcastType::NUMPY},     {{2, 3, 4, 5}, data_types::i8, format::bfyx},                   {}},
    {{{2, 3, 4, 5}, data_types::f16, format::bfyx},             {{5}, data_types::f16, format::bfyx},                       eltwise_mode::ne,       {AutoBroadcastType::NUMPY},     {{2, 3, 4, 5}, data_types::i8, format::bfyx},                   {}},
    {{{2, 3, 4, 5}, data_types::f16, format::bfyx},             {{5}, data_types::f16, format::bfyx},                       eltwise_mode::lt,       {AutoBroadcastType::NUMPY},     {{2, 3, 4, 5}, data_types::i8, format::bfyx},                   {}},
    {{{2, 3, 4, 5}, data_types::i32, format::bfyx},             {{5}, data_types::i32, format::bfyx},                       eltwise_mode::le,       {AutoBroadcastType::NUMPY},     {{2, 3, 4, 5}, data_types::i8, format::bfyx},                   {}},
    {{{2, 3, 4, 5}, data_types::i64, format::bfyx},             {{5}, data_types::i64, format::bfyx},                       eltwise_mode::gt,       {AutoBroadcastType::NUMPY},     {{2, 3, 4, 5}, data_types::i8, format::bfyx},                   {}},
    {{{2, 3, 4, 5}, data_types::u8,  format::bfyx},             {{3}, data_types::u8,  format::bfyx},                       eltwise_mode::ge,       {AutoBroadcastType::PDPD, 1},   {{2, 3, 4, 5}, data_types::i8, format::bfyx},                   {}},
    {{{2, 3, 4, 5}, data_types::i8,  format::bfyx},             {{3}, data_types::i8,  format::bfyx},                       eltwise_mode::logic_and,{AutoBroadcastType::PDPD, 1},   {{2, 3, 4, 5}, data_types::i8, format::bfyx},                   {}},
    {{{2, 3, 4, 5}, data_types::f32, format::bfyx},             {{3}, data_types::f32, format::bfyx},                       eltwise_mode::logic_or, {AutoBroadcastType::PDPD, 1},   {{2, 3, 4, 5}, data_types::i8, format::bfyx},                   {}},
    {{{2, 3, 4, 5}, data_types::f32, format::bfyx},             {{3}, data_types::f32, format::bfyx},                       eltwise_mode::logic_xor,{AutoBroadcastType::PDPD, 1},   {{2, 3, 4, 5}, data_types::i8, format::bfyx},                   {}},
    // test stride
    {{{5, 2, 1, 20}, data_types::f32, format::bfyx},            {{1, 40, 1}, data_types::f32, format::bfyx},                eltwise_mode::sum,      {AutoBroadcastType::NUMPY},     {{5, 2, 1, 5}, data_types::f32, format::bfyx},                  {{1,3,4,2}}},
    {{{2, 3, 40,50}, data_types::f32, format::bfyx},            {{40, 50}, data_types::f32, format::bfyx},                  eltwise_mode::sum,      {AutoBroadcastType::PDPD, -1},  {{2, 3, 5, 10}, data_types::f32, format::bfyx},                 {{1,1,5,8}}},
    {{PartialShape::dynamic(4), data_types::f32, format::bfyx}, {{2, 1, 5}, data_types::f32, format::bfyx},                 eltwise_mode::sum,      {AutoBroadcastType::NUMPY},     {PartialShape::dynamic(4), data_types::f32, format::bfyx},      {{1,1,5,8}}},
    {{PartialShape::dynamic(4), data_types::f32, format::bfyx}, {{2, 1, 5}, data_types::f32, format::bfyx},                 eltwise_mode::sum,      {AutoBroadcastType::PDPD, 1},   {PartialShape::dynamic(4), data_types::f32, format::bfyx},      {{1,1,3,8}}},
}));

}  // shape_infer_tests
