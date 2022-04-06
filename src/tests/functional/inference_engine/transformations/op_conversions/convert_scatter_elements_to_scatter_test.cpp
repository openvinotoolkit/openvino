// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include <string>
#include <sstream>
#include <fstream>
#include <memory>
#include <queue>
#include <map>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <transformations/op_conversions/convert_scatter_elements_to_scatter.hpp>
#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

#include <ngraph/pass/manager.hpp>

using namespace testing;

std::shared_ptr<ngraph::Function> get_initial_function(const ngraph::PartialShape & data_shape,
                                                       const ngraph::PartialShape & indexes_shape,
                                                       const ngraph::PartialShape & updates_shape,
                                                       const ngraph::PartialShape & broadcast_shape,
                                                       const int64_t & axis) {
    auto data = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, data_shape);
    auto indexes = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, indexes_shape);
    auto updates = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, updates_shape);
    auto axis_const = ngraph::opset3::Constant::create(ngraph::element::i64, {1}, {axis});

    auto broadcast_len = broadcast_shape.rank().get_length();
    if (std::numeric_limits<size_t>::max() < broadcast_len) {
        throw ngraph::ngraph_error("broadcast_len cannot be represented in size_t");
    }

    auto broadcast_shape_param = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64,
        ngraph::Shape{static_cast<size_t>(broadcast_len)});
    auto broadcast = std::make_shared<ngraph::opset3::Broadcast>(indexes, broadcast_shape_param);

    auto scatter = std::make_shared<ngraph::opset3::ScatterElementsUpdate>(data, broadcast, updates, axis_const);

    return std::make_shared<ngraph::Function>(ngraph::NodeVector{scatter}, ngraph::ParameterVector{data, indexes, updates, broadcast_shape_param});
}

std::shared_ptr<ngraph::Function> get_reference_function(const ngraph::PartialShape & data_shape,
                                                         const ngraph::PartialShape & indexes_shape,
                                                         const ngraph::PartialShape & updates_shape,
                                                         const int64_t & axis,
                                                         const ngraph::Shape & reshape_shape = {},
                                                         const std::vector<int64_t> & squeeze_indices = {}) {
    auto data = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, data_shape);
    auto indexes = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, indexes_shape);
    auto updates = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, updates_shape);
    auto axis_const = ngraph::opset3::Constant::create(ngraph::element::i64, {1}, {axis});

    ngraph::Output<ngraph::Node> index_out = indexes->output(0);
    if (!reshape_shape.empty()) {
        index_out = std::make_shared<ngraph::opset3::Reshape>(indexes,
                ngraph::opset3::Constant::create(ngraph::element::i64, {reshape_shape.size()}, reshape_shape), false);
    }

    if (!squeeze_indices.empty()) {
        index_out = std::make_shared<ngraph::opset3::Squeeze>(indexes,
                ngraph::opset3::Constant::create(ngraph::element::i64, {squeeze_indices.size()}, squeeze_indices));
    }

    auto scatter = std::make_shared<ngraph::opset3::ScatterUpdate>(data, index_out, updates, axis_const);

    return std::make_shared<ngraph::Function>(ngraph::NodeVector{scatter}, ngraph::ParameterVector{data, indexes, updates});
}

void test(std::shared_ptr<ngraph::Function> f, std::shared_ptr<ngraph::Function> f_ref) {
    auto unh = std::make_shared<ngraph::pass::UniqueNamesHolder>();
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::InitUniqueNames>(unh);
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    manager.register_pass<ngraph::pass::ConvertScatterElementsToScatter>();
    manager.register_pass<ngraph::pass::CheckUniqueNames>(unh);
    manager.register_pass<ngraph::pass::InjectionPass>([](std::shared_ptr<ngraph::Function> f) {
        check_rt_info(f);
    });
    manager.register_pass<ngraph::pass::ConstantFolding>();
    ASSERT_NO_THROW(manager.run_passes(f));

    auto fc = FunctionsComparator::no_default()
            .enable(FunctionsComparator::NODES)
            .enable(FunctionsComparator::PRECISIONS);
    auto res = fc.compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

void test(std::shared_ptr<ngraph::Function> f) {
    test(f, f);
}

TEST(TransformationTests, ConvertScatterElementsToScatterTestAxis0) {
    test(get_initial_function({1000, 256, 7, 7}, {1000, 1, 1, 1}, {1000, 256, 7, 7}, {1000, 256, 7, 7}, 0),
         get_reference_function({1000, 256, 7, 7}, {1000, 1, 1, 1}, {1000, 256, 7, 7}, 0, {1000}));
}

TEST(TransformationTests, ConvertScatterElementsToScatterTestAxis1) {
    test(get_initial_function({1000, 256, 7, 7}, {256, 1, 1}, {1000, 256, 7, 7}, {1000, 256, 7, 7}, 1),
         get_reference_function({1000, 256, 7, 7}, {256, 1, 1}, {1000, 256, 7, 7}, 1, {256}));
}

TEST(TransformationTests, ConvertScatterElementsToScatterTestNoReshape) {
    test(get_initial_function({1000, 256, 7, 7}, {1}, {1000, 1, 7, 7}, {1000, 1, 7, 7}, 1),
         get_reference_function({1000, 256, 7, 7}, {1}, {1000, 1, 7, 7}, 1));
}

TEST(TransformationTests, ConvertScatterElementsToScatterTestNoReshapeNegAxis) {
    test(get_initial_function({1000, 256, 7, 7}, {1}, {1000, 1, 7, 7}, {1000, 1, 7, 7}, -3),
         get_reference_function({1000, 256, 7, 7}, {1}, {1000, 1, 7, 7}, -3));
}

TEST(TransformationTests, ConvertScatterElementsToScatterTestNegative) {
    test(get_initial_function({1000, 256, 7, 7}, {1000, 256, 1, 1}, {1000, 256, 7, 7}, {1000, 256, 7, 7}, 0));
}

TEST(TransformationTests, ConvertScatterElementsToScatterTestAxis0Dyn) {
    test(get_initial_function({DYN, 256, 7, 7}, {DYN, 1, 1, 1}, {DYN, 256, 7, 7}, {DYN, 256, 7, 7}, 0),
         get_reference_function({DYN, 256, 7, 7}, {DYN, 1, 1, 1}, {DYN, 256, 7, 7}, 0, {}, {1, 2, 3}));
}

TEST(TransformationTests, ConvertScatterElementsToScatterTestAxis1Dyn) {
    test(get_initial_function({1000, DYN, 7, 7}, {DYN, 1, 1}, {1000, DYN, 7, 7}, {1000, DYN, 7, 7}, 1),
         get_reference_function({1000, DYN, 7, 7}, {DYN, 1, 1}, {1000, DYN, 7, 7}, 1, {}, {1, 2}));
}

TEST(TransformationTests, ConvertScatterElementsToScatterTestAxis1NoSqueezeDyn) {
    test(get_initial_function({1000, DYN, 7, 7}, {DYN}, {1000, 256, 7, 7}, {1000, DYN, 7, 7}, 1),
         get_reference_function({1000, DYN, 7, 7}, {DYN}, {1000, 256, 7, 7}, 1));
}

TEST(TransformationTests, ConvertScatterElementsToScatterTestAxis0Neg1Dyn) {
    test(get_initial_function({DYN, 256, 7, 7}, {DYN, DYN, 1, 1}, {DYN, 256, 7, 7}, {DYN, 256, 7, 7}, 0));
}

TEST(TransformationTests, ConvertScatterElementsToScatterTestAxis0Neg2Dyn) {
    test(get_initial_function({DYN, 256, 7, 7}, {DYN, 1, 2, 1}, {DYN, 256, 7, 7}, {DYN, 256, 7, 7}, 0));
}