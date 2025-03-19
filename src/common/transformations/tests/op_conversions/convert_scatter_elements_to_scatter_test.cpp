// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_scatter_elements_to_scatter.hpp"

#include <gtest/gtest.h>

#include <fstream>
#include <map>
#include <memory>
#include <queue>
#include <sstream>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace testing;

std::shared_ptr<ov::Model> get_initial_function(const PartialShape& data_shape,
                                                const PartialShape& indexes_shape,
                                                const PartialShape& updates_shape,
                                                const PartialShape& broadcast_shape,
                                                const int64_t& axis) {
    auto data = std::make_shared<opset3::Parameter>(element::f32, data_shape);
    auto indexes = std::make_shared<opset3::Parameter>(element::i64, indexes_shape);
    auto updates = std::make_shared<opset3::Parameter>(element::f32, updates_shape);
    auto axis_const = opset3::Constant::create(element::i64, {1}, {axis});

    auto broadcast_len = broadcast_shape.rank().get_length();
    if (std::numeric_limits<size_t>::max() < (size_t)broadcast_len) {
        OPENVINO_THROW("broadcast_len cannot be represented in size_t");
    }

    auto broadcast_shape_param =
        std::make_shared<opset3::Parameter>(element::i64, Shape{static_cast<size_t>(broadcast_len)});
    auto broadcast = std::make_shared<opset3::Broadcast>(indexes, broadcast_shape_param);

    auto scatter = std::make_shared<opset3::ScatterElementsUpdate>(data, broadcast, updates, axis_const);

    return std::make_shared<ov::Model>(NodeVector{scatter},
                                       ParameterVector{data, indexes, updates, broadcast_shape_param});
}

std::shared_ptr<ov::Model> get_reference_function(const PartialShape& data_shape,
                                                  const PartialShape& indexes_shape,
                                                  const PartialShape& updates_shape,
                                                  const int64_t& axis,
                                                  const Shape& reshape_shape = {},
                                                  const std::vector<int64_t>& squeeze_indices = {}) {
    auto data = std::make_shared<opset3::Parameter>(element::f32, data_shape);
    auto indexes = std::make_shared<opset3::Parameter>(element::i64, indexes_shape);
    auto updates = std::make_shared<opset3::Parameter>(element::f32, updates_shape);
    auto axis_const = opset3::Constant::create(element::i64, {1}, {axis});

    Output<Node> index_out = indexes->output(0);
    if (!reshape_shape.empty()) {
        index_out = std::make_shared<opset3::Reshape>(
            indexes,
            opset3::Constant::create(element::i64, {reshape_shape.size()}, reshape_shape),
            false);
    }

    if (!squeeze_indices.empty()) {
        index_out = std::make_shared<opset3::Squeeze>(
            indexes,
            opset3::Constant::create(element::i64, {squeeze_indices.size()}, squeeze_indices));
    }

    auto scatter = std::make_shared<opset3::ScatterUpdate>(data, index_out, updates, axis_const);

    return std::make_shared<ov::Model>(NodeVector{scatter}, ParameterVector{data, indexes, updates});
}

void gen_test(std::shared_ptr<ov::Model> f, std::shared_ptr<ov::Model> f_ref) {
    auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();
    pass::Manager manager;
    manager.register_pass<ov::pass::InitUniqueNames>(unh);
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::pass::ConvertScatterElementsToScatter>();
    manager.register_pass<ov::pass::CheckUniqueNames>(unh);
    manager.register_pass<ov::pass::InjectionPass>([](std::shared_ptr<ov::Model> f) {
        check_rt_info(f);
    });
    manager.register_pass<pass::ConstantFolding>();
    OV_ASSERT_NO_THROW(manager.run_passes(f));

    auto fc =
        FunctionsComparator::no_default().enable(FunctionsComparator::NODES).enable(FunctionsComparator::PRECISIONS);
    auto res = fc.compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

void gen_test(std::shared_ptr<ov::Model> f) {
    gen_test(f, f);
}

TEST(TransformationTests, ConvertScatterElementsToScatterTestAxis0) {
    gen_test(get_initial_function({1000, 256, 7, 7}, {1000, 1, 1, 1}, {1000, 256, 7, 7}, {1000, 256, 7, 7}, 0),
             get_reference_function({1000, 256, 7, 7}, {1000, 1, 1, 1}, {1000, 256, 7, 7}, 0, {1000}));
}

TEST(TransformationTests, ConvertScatterElementsToScatterTestAxis1) {
    gen_test(get_initial_function({1000, 256, 7, 7}, {256, 1, 1}, {1000, 256, 7, 7}, {1000, 256, 7, 7}, 1),
             get_reference_function({1000, 256, 7, 7}, {256, 1, 1}, {1000, 256, 7, 7}, 1, {256}));
}

TEST(TransformationTests, ConvertScatterElementsToScatterTestNoReshape) {
    gen_test(get_initial_function({1000, 256, 7, 7}, {1}, {1000, 1, 7, 7}, {1000, 1, 7, 7}, 1),
             get_reference_function({1000, 256, 7, 7}, {1}, {1000, 1, 7, 7}, 1));
}

TEST(TransformationTests, ConvertScatterElementsToScatterTestNoReshapeNegAxis) {
    gen_test(get_initial_function({1000, 256, 7, 7}, {1}, {1000, 1, 7, 7}, {1000, 1, 7, 7}, -3),
             get_reference_function({1000, 256, 7, 7}, {1}, {1000, 1, 7, 7}, -3));
}

TEST(TransformationTests, ConvertScatterElementsToScatterTestNegative) {
    gen_test(get_initial_function({1000, 256, 7, 7}, {1000, 256, 1, 1}, {1000, 256, 7, 7}, {1000, 256, 7, 7}, 0));
}

TEST(TransformationTests, ConvertScatterElementsToScatterTestAxis0Dyn) {
    gen_test(get_initial_function({DYN, 256, 7, 7}, {DYN, 1, 1, 1}, {DYN, 256, 7, 7}, {DYN, 256, 7, 7}, 0),
             get_reference_function({DYN, 256, 7, 7}, {DYN, 1, 1, 1}, {DYN, 256, 7, 7}, 0, {}, {1, 2, 3}));
}

TEST(TransformationTests, ConvertScatterElementsToScatterTestAxis1Dyn) {
    gen_test(get_initial_function({1000, DYN, 7, 7}, {DYN, 1, 1}, {1000, DYN, 7, 7}, {1000, DYN, 7, 7}, 1),
             get_reference_function({1000, DYN, 7, 7}, {DYN, 1, 1}, {1000, DYN, 7, 7}, 1, {}, {1, 2}));
}

TEST(TransformationTests, ConvertScatterElementsToScatterTestAxis1NoSqueezeDyn) {
    gen_test(get_initial_function({1000, DYN, 7, 7}, {DYN}, {1000, 256, 7, 7}, {1000, DYN, 7, 7}, 1),
             get_reference_function({1000, DYN, 7, 7}, {DYN}, {1000, 256, 7, 7}, 1));
}

TEST(TransformationTests, ConvertScatterElementsToScatterTestAxis0Neg1Dyn) {
    gen_test(get_initial_function({DYN, 256, 7, 7}, {DYN, DYN, 1, 1}, {DYN, 256, 7, 7}, {DYN, 256, 7, 7}, 0));
}

TEST(TransformationTests, ConvertScatterElementsToScatterTestAxis0Neg2Dyn) {
    gen_test(get_initial_function({DYN, 256, 7, 7}, {DYN, 1, 2, 1}, {DYN, 256, 7, 7}, {DYN, 256, 7, 7}, 0));
}
