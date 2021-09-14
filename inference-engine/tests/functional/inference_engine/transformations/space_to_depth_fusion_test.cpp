// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/manager.hpp>
#include <queue>
#include <random>
#include <string>
#include <transformations/common_optimizations/space_to_depth_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

static const auto end_max = std::numeric_limits<int64_t>::max();

static std::shared_ptr<opset8::StridedSlice> create_ss(const Output<Node>& data_node,
                                                size_t ndims,
                                                int axis,
                                                int begin,
                                                int stride) {
    std::vector<int64_t> begin_c(ndims, 0);
    std::vector<int64_t> end_c(ndims, 0);
    std::vector<int64_t> stride_c(ndims, 1);
    begin_c[axis] = begin;
    stride_c[axis] = stride;
    auto begin_node = opset8::Constant::create(ngraph::element::i64, ngraph::Shape{ndims}, begin_c);
    auto end_node = opset8::Constant::create(ngraph::element::i64, ngraph::Shape{ndims}, end_c);
    auto stride_node = opset8::Constant::create(ngraph::element::i64, ngraph::Shape{ndims}, stride_c);
    std::vector<int64_t> begin_mask(ndims, 0);
    std::vector<int64_t> end_mask(ndims, 1);
    auto ss =
        std::make_shared<opset8::StridedSlice>(data_node, begin_node, end_node, stride_node, begin_mask, end_mask);
    return ss;
}

struct coordinate : std::vector<int> {
    using base = std::vector<int>;
    int radix;
    coordinate(int ndims, int radix) : base(ndims, 0), radix(radix) {}
    coordinate& operator++() {
        int ndims = size();
        for (int k = ndims - 1; k >= 0; k--) {
            (*this)[k]++;
            if ((*this)[k] < radix)
                break;
            (*this)[k] = 0;
        }
        return *this;
    }
};

static std::shared_ptr<Node> build_ss_chain(const Output<Node>& in,
                                            int block_size,
                                            const std::vector<int>& shuffle = {}) {
    auto shape = in.get_shape();

    OutputVector ss_outputs;

    coordinate begin(shape.size(), block_size);

    do {
        std::shared_ptr<Node> node = in.get_node_shared_ptr();
        for (int k = 2; k < shape.size(); k++)
            node = create_ss(node, k + 1, k, begin[k], block_size);

        ss_outputs.push_back(node);

        ++begin;
    } while (begin[1] == 0);

    if (shuffle.size()) {
        OutputVector after_shuffle;

        for (int i = 0; i < ss_outputs.size(); i++) {
            auto id = shuffle[i % shuffle.size()];
            after_shuffle.push_back(ss_outputs[id]);
        }

        ss_outputs = after_shuffle;
    }

    return std::make_shared<opset8::Concat>(ss_outputs, 1);
}

TEST(TransformationTests, SpaceToDepthFusionFromStridedSlice2x2) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    auto block_size = 2;
    {
        auto in = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 640, 640});
        auto out = build_ss_chain(in, block_size);
        f = std::make_shared<Function>(NodeVector{out}, ParameterVector{in});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::SpaceToDepthFusion>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto in = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 640, 640});
        auto space_to_depth =
            std::make_shared<opset8::SpaceToDepth>(in,
                                                   opset6::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST,
                                                   block_size);

        f_ref = std::make_shared<Function>(NodeVector{space_to_depth}, ParameterVector{in});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, SpaceToDepthFusionFromStridedSlice2x2_Negative) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    auto block_size = 2;
    {
        auto in = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 640, 640});
        auto out = build_ss_chain(in, block_size, {0, 1, 3, 2});  // shuffled order, so should fail
        f = std::make_shared<Function>(NodeVector{out}, ParameterVector{in});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::SpaceToDepthFusion>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto in = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 640, 640});
        auto out = build_ss_chain(in, block_size, {0, 1, 3, 2});  // shuffled order, so should fail
        f_ref = std::make_shared<Function>(NodeVector{out}, ParameterVector{in});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, SpaceToDepthFusionFromStridedSlice3x3) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    auto block_size = 3;

    {
        auto in = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 120, 120});
        auto out = build_ss_chain(in, block_size);
        f = std::make_shared<Function>(NodeVector{out}, ParameterVector{in});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::SpaceToDepthFusion>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto in = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 120, 120});
        auto space_to_depth =
            std::make_shared<opset8::SpaceToDepth>(in,
                                                   opset6::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST,
                                                   block_size);

        f_ref = std::make_shared<Function>(NodeVector{space_to_depth}, ParameterVector{in});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, SpaceToDepthFusionFromStridedSlice2x2x2) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    auto block_size = 2;
    {
        auto in = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 640, 640, 640});

        auto out = build_ss_chain(in, block_size);

        f = std::make_shared<Function>(NodeVector{out}, ParameterVector{in});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::SpaceToDepthFusion>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto in = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 640, 640, 640});
        auto space_to_depth =
            std::make_shared<opset8::SpaceToDepth>(in,
                                                   opset6::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST,
                                                   block_size);

        f_ref = std::make_shared<Function>(NodeVector{space_to_depth}, ParameterVector{in});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, SpaceToDepthFusionFromStridedSlice2x2WithConv) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    std::vector<float> weights(10 * 12 * 3 * 3, 0);

    auto block_size = 2;
    {
        auto in = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 640, 640});
        auto ssconcat = build_ss_chain(in, block_size, {0, 1, 3, 2});

        auto filters = op::Constant::create(element::f32, Shape{10, 12, 3, 3}, weights);
        auto out = std::make_shared<opset8::Convolution>(ssconcat,
                                                         filters,
                                                         Strides{1, 1},
                                                         CoordinateDiff{0, 0},
                                                         CoordinateDiff{0, 0},
                                                         Strides{1, 1});

        f = std::make_shared<Function>(NodeVector{out}, ParameterVector{in});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::SpaceToDepthFusion>();
        m.register_pass<pass::ConstantFolding>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto in = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 640, 640});
        auto space_to_depth =
            std::make_shared<opset8::SpaceToDepth>(in,
                                                   opset6::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST,
                                                   block_size);

        auto filters = op::Constant::create(element::f32, Shape{10, 12, 3, 3}, weights);
        auto out = std::make_shared<opset8::Convolution>(space_to_depth,
                                                         filters,
                                                         Strides{1, 1},
                                                         CoordinateDiff{0, 0},
                                                         CoordinateDiff{0, 0},
                                                         Strides{1, 1});

        f_ref = std::make_shared<Function>(NodeVector{out}, ParameterVector{in});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}
