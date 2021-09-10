// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <string>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/space_to_depth_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

std::shared_ptr<opset6::StridedSlice> create_ss(const Output<Node> &data_node,
                                                size_t ndims, int stride,
                                                int axis, int begin) {
  std::vector<int64_t> begin_c(ndims, 0);
  std::vector<int64_t> end_c(ndims, 0);
  std::vector<int64_t> stride_c(ndims, 1);
  begin_c[axis] = begin;
  stride_c[axis] = stride;
  auto begin_node = opset6::Constant::create(ngraph::element::i64,
                                             ngraph::Shape{ndims}, begin_c);
  auto end_node = opset6::Constant::create(ngraph::element::i64,
                                           ngraph::Shape{ndims}, end_c);
  auto stride_node = opset6::Constant::create(ngraph::element::i64,
                                              ngraph::Shape{ndims}, stride_c);
  std::vector<int64_t> begin_mask(ndims, 0);
  std::vector<int64_t> end_mask(ndims, 1);
  auto ss = std::make_shared<opset6::StridedSlice>(
      data_node, begin_node, end_node, stride_node, begin_mask, end_mask);
  return ss;
}

TEST(TransformationTests, SpaceToDepthFusionFromStridedSlice2x2) {
  std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
  {
    auto in = std::make_shared<opset6::Parameter>(element::f32,
                                                  Shape{1, 3, 640, 640});
    auto ss_chain = [&](int begin_dim2, int begin_dim3) {
      auto s0 = create_ss(in, 4, 2, 2, begin_dim2);
      auto s1 = create_ss(s0, 4, 2, 3, begin_dim3);
      return s1;
    };

    auto a = ss_chain(0, 0);
    auto b = ss_chain(0, 1);
    auto c = ss_chain(1, 0);
    auto d = ss_chain(1, 1);

    auto out = std::make_shared<opset6::Concat>(OutputVector{a, b, c, d}, 1);

    f = std::make_shared<Function>(NodeVector{out}, ParameterVector{in});

    pass::Manager m;
    m.register_pass<pass::InitNodeInfo>();
    m.register_pass<pass::SpaceToDepthFusion>();
    m.run_passes(f);
    ASSERT_NO_THROW(check_rt_info(f));
  }

  {
    auto data =
        std::make_shared<opset6::Parameter>(element::f32, Shape{12, 3, 4, 8});
    auto batch_to_space = std::make_shared<opset6::BatchToSpace>(
        data, op::Constant::create(element::i64, Shape{4}, {1, 1, 2, 2}),
        op::Constant::create(element::i64, Shape{4}, {0, 0, 2, 1}),
        op::Constant::create(element::i64, Shape{4}, {1, 2, 1, 14}));

    f_ref = std::make_shared<Function>(NodeVector{batch_to_space},
                                       ParameterVector{data});
  }

  auto res = compare_functions(f, f_ref, true);
  ASSERT_TRUE(res.first) << res.second;
}
