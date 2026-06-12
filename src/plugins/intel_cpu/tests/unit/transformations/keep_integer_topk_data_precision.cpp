// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/convert_precision.hpp"
#include "transformations/cpu_opset/convert_to_cpu_specific_opset.hpp"

namespace {

class KeepIntegerTopKDataPrecisionTest : public testing::TestWithParam<ov::element::Type> {};

std::shared_ptr<ov::op::v11::TopK> make_topk(const ov::Output<ov::Node>& data) {
    auto k = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {3});
    return std::make_shared<ov::op::v11::TopK>(data,
                                               k,
                                               0,
                                               ov::op::TopKMode::MAX,
                                               ov::op::TopKSortType::SORT_VALUES,
                                               ov::element::i32,
                                               false);
}

std::shared_ptr<ov::op::v0::Constant> make_data_constant(const ov::element::Type& precision) {
    if (precision == ov::element::i64) {
        return ov::op::v0::Constant::create(precision, ov::Shape{5}, std::vector<int64_t>{1, 2, 3, 4, 5});
    }

    return ov::op::v0::Constant::create(precision, ov::Shape{5}, std::vector<uint64_t>{1, 2, 3, 4, 5});
}

void run_convert_precision(const std::shared_ptr<ov::Model>& model) {
    ov::pass::Manager manager;
    manager.register_pass<ov::intel_cpu::KeepIntegerTopKDataPrecision>();
    manager.register_pass<ov::pass::ConvertPrecision>(precisions_map{{ov::element::i64, ov::element::i32},
                                                                     {ov::element::u64, ov::element::i32}},
                                                      type_to_fuse_map{},
                                                      false,
                                                      false);
    manager.run_passes(model);
}

TEST_P(KeepIntegerTopKDataPrecisionTest, TopKOnlyParameterKeepsTopKDataPrecision) {
    const auto precision = GetParam();
    auto data = std::make_shared<ov::op::v0::Parameter>(precision, ov::Shape{5});
    auto topk = make_topk(data);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{topk->output(0), topk->output(1)},
                                             ov::ParameterVector{data});

    run_convert_precision(model);

    EXPECT_EQ(topk->get_input_element_type(0), precision);
    EXPECT_EQ(topk->get_output_element_type(0), precision);
}

TEST_P(KeepIntegerTopKDataPrecisionTest, ConstantKeepsTopKDataPrecision) {
    const auto precision = GetParam();
    auto data = make_data_constant(precision);
    auto topk = make_topk(data);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{topk->output(0), topk->output(1)},
                                             ov::ParameterVector{});

    run_convert_precision(model);

    EXPECT_EQ(topk->get_input_element_type(0), precision);
    EXPECT_EQ(topk->get_output_element_type(0), precision);
}

TEST_P(KeepIntegerTopKDataPrecisionTest, SharedParameterKeepsTopKDataPrecisionAndConvertsOtherConsumers) {
    const auto precision = GetParam();
    auto data = std::make_shared<ov::op::v0::Parameter>(precision, ov::Shape{5});
    auto topk = make_topk(data);
    auto abs = std::make_shared<ov::op::v0::Abs>(data);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{topk->output(0), topk->output(1), abs->output(0)},
                                             ov::ParameterVector{data});

    run_convert_precision(model);

    EXPECT_EQ(topk->get_input_element_type(0), precision);
    EXPECT_EQ(topk->get_output_element_type(0), precision);

    const auto abs_input = abs->input_value(0);
    EXPECT_EQ(abs_input.get_element_type(), ov::element::i32);
    ASSERT_TRUE(ov::is_type<ov::op::v0::Convert>(abs_input.get_node_shared_ptr()));
    EXPECT_EQ(abs_input.get_node_shared_ptr()->get_input_element_type(0), precision);

    const auto& results = model->get_results();
    ASSERT_EQ(results.size(), 3);
    EXPECT_EQ(results[2]->get_input_element_type(0), precision);
}

INSTANTIATE_TEST_SUITE_P(smoke_SharedSource,
                         KeepIntegerTopKDataPrecisionTest,
                         testing::Values(ov::element::i64, ov::element::u64));

}  // namespace
