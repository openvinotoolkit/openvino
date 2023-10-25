// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common/graph_utils.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_models/builders.hpp"
#include "transformations/gather_sinking_transpose.hpp"
#include "transformations/init_node_info.hpp"

using namespace ov::opset10;
using namespace ov::intel_gna;

namespace gather_transpose_merge_test {

namespace {

inline std::vector<size_t> make_indexes(size_t size) {
    std::vector<size_t> indexes(size);
    std::iota(indexes.begin(), indexes.end(), 0);
    std::reverse(indexes.begin(), indexes.end());
    return indexes;
}

inline std::vector<size_t> make_transpose_order(const std::vector<size_t>& input_shape) {
    std::vector<size_t> transpose_order;
    switch (input_shape.size()) {
    case 2:
        transpose_order = {1, 0};
        break;
    case 3:
        transpose_order = {0, 2, 1};
        break;
    case 4:
        transpose_order = {0, 2, 3, 1};
        break;
    default:
        break;
    }

    return transpose_order;
}

}  // namespace

typedef std::tuple<ov::element::Type,  // Net precision
                   ov::Shape           // Input shape
                   >
    GatherTransposeMergeTestParams;

class GatherTransposeMergeBase : public ov::test::TestsCommon,
                                 public ::testing::WithParamInterface<GatherTransposeMergeTestParams> {
public:
    static std::string get_test_name(const testing::TestParamInfo<GatherTransposeMergeTestParams>& obj) {
        std::vector<size_t> input_shape;
        ov::element::Type net_type;
        std::tie(net_type, input_shape) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << net_type << "_";
        result << "Shape=" << ov::test::utils::vec2str(input_shape);

        return result.str();
    }
    void SetUp() override {
        std::tie(m_net_type, m_input_shape) = this->GetParam();
    }

    virtual void init_test_model(){};
    virtual void init_ref_model(){};

    virtual void Validate() {
        ov::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::Serialize>("test_before.xml", "test_before.bin");
        m.register_pass<ov::intel_gna::pass::GatherSinkingTranspose>();
        m.register_pass<ov::pass::Serialize>("test_after.xml", "test_after.bin");
        m.run_passes(m_model);

        const FunctionsComparator func_comparator =
            FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
        FunctionsComparator::Result result = func_comparator(m_model, m_model_ref);

        EXPECT_TRUE(result.valid) << result.message;
    }
    virtual void Run() {
        SetUp();
        init_test_model();
        init_ref_model();
        Validate();
    }

protected:
    std::shared_ptr<ov::Model> m_model, m_model_ref;
    ov::element::Type m_net_type;
    ov::Shape m_input_shape, m_output_shape;
    ov::AxisVector m_gather_ids_ref;
};

class TransposeGatherTest : public GatherTransposeMergeBase {
public:
    void init_test_model() override {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(m_net_type, ov::Shape(m_input_shape))};
        const size_t input_shape_size = ov::shape_size(params[0]->get_shape());

        std::vector<size_t> transpose_order = make_transpose_order(m_input_shape);
        auto transpose_const =
            std::make_shared<Constant>(ov::element::i8, ov::Shape{transpose_order.size()}, transpose_order);
        auto transpose_node = std::make_shared<Transpose>(params[0], transpose_const);

        ov::Shape shape_in = {1, input_shape_size};
        auto reshape_in_const = std::make_shared<Constant>(ov::element::i64, ov::Shape{shape_in.size()}, shape_in);
        auto reshape_in_node = std::make_shared<Reshape>(transpose_node, reshape_in_const, false);

        const std::vector<size_t> gather_ids = make_indexes(ov::shape_size(m_input_shape));
        auto gather_const_ids = Constant::create(ov::element::i64, ov::Shape{gather_ids.size()}, gather_ids);
        const size_t gather_axis = 1;
        auto gather_const_axis = Constant::create(ov::element::i64, ov::Shape{}, {gather_axis});
        auto gather_node = std::make_shared<Gather>(reshape_in_node, gather_const_ids, gather_const_axis);

        ov::Shape shape_out = transpose_node->get_output_shape(0);
        auto reshape_out_const = std::make_shared<Constant>(ov::element::i64, ov::Shape{shape_out.size()}, shape_out);
        auto reshape_out_node = std::make_shared<Reshape>(gather_node, reshape_out_const, false);

        ov::ResultVector results{std::make_shared<Result>(reshape_out_node)};
        m_model = std::make_shared<ov::Model>(results, params, "concat");
        // save for the ref model
        const ov::AxisVector transpose_ids =
            graph_utils::make_gather_indexes_from_transpose_axes(m_input_shape, transpose_order);
        m_gather_ids_ref = graph_utils::combine_gather_indexes(gather_ids, transpose_ids);
        m_output_shape = shape_out;
    }

    void init_ref_model() override {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(m_net_type, ov::Shape(m_input_shape))};
        const size_t input_shape_size = ov::shape_size(params[0]->get_shape());

        ov::Shape shape_in = {1, input_shape_size};
        auto reshape_in_const = std::make_shared<Constant>(ov::element::i64, ov::Shape{shape_in.size()}, shape_in);
        auto reshape_in_node = std::make_shared<Reshape>(params[0], reshape_in_const, false);

        const std::vector<size_t> gather_ids = m_gather_ids_ref;
        auto gather_const_ids = Constant::create(ov::element::i64, ov::Shape{gather_ids.size()}, gather_ids);
        const size_t gather_axis = 1;
        auto gather_const_axis = Constant::create(ov::element::i64, ov::Shape{}, {gather_axis});
        auto gather_node = std::make_shared<Gather>(reshape_in_node, gather_const_ids, gather_const_axis);

        ov::Shape shape_out = m_output_shape;
        auto reshape_out_const = std::make_shared<Constant>(ov::element::i64, ov::Shape{shape_out.size()}, shape_out);
        auto reshape_out_node = std::make_shared<Reshape>(gather_node, reshape_out_const, false);

        ov::ResultVector results{std::make_shared<Result>(reshape_out_node)};
        m_model_ref = std::make_shared<ov::Model>(results, params, "concat");
    }
};

class GatherTransposeTest : public GatherTransposeMergeBase {
public:
    void init_test_model() override {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(m_net_type, ov::Shape(m_input_shape))};
        const size_t input_shape_size = ov::shape_size(params[0]->get_shape());

        ov::Shape shape_in = {1, input_shape_size};
        auto reshape_in_const = std::make_shared<Constant>(ov::element::i64, ov::Shape{shape_in.size()}, shape_in);
        auto reshape_in_node = std::make_shared<Reshape>(params[0], reshape_in_const, false);

        const std::vector<size_t> gather_ids = make_indexes(ov::shape_size(m_input_shape));
        auto gather_const_ids = Constant::create(ov::element::i64, ov::Shape{gather_ids.size()}, gather_ids);
        const size_t gather_axis = 1;
        auto gather_const_axis = Constant::create(ov::element::i64, ov::Shape{}, {gather_axis});
        auto gather_node = std::make_shared<Gather>(reshape_in_node, gather_const_ids, gather_const_axis);

        ov::Shape shape_middle = m_input_shape;
        auto reshape_middle_const =
            std::make_shared<Constant>(ov::element::i64, ov::Shape{shape_middle.size()}, shape_middle);
        auto reshape_middle_node = std::make_shared<Reshape>(gather_node, reshape_middle_const, false);

        std::vector<size_t> transpose_order = make_transpose_order(m_input_shape);
        auto transpose_const =
            std::make_shared<Constant>(ov::element::i8, ov::Shape{transpose_order.size()}, transpose_order);
        auto transpose_node = std::make_shared<Transpose>(reshape_middle_node, transpose_const);

        ov::ResultVector results{std::make_shared<Result>(transpose_node)};
        m_model = std::make_shared<ov::Model>(results, params, "transpose_gather_test_model");

        // save values for the ref model
        const ov::AxisVector transpose_ids =
            graph_utils::make_gather_indexes_from_transpose_axes(m_input_shape, transpose_order);
        m_gather_ids_ref = graph_utils::combine_gather_indexes(gather_ids, transpose_ids);
        m_output_shape = transpose_node->get_output_shape(0);
    }

    void init_ref_model() override {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(m_net_type, ov::Shape(m_input_shape))};
        const size_t input_shape_size = ov::shape_size(params[0]->get_shape());

        ov::Shape shape_in = {1, input_shape_size};
        auto reshape_in_const = std::make_shared<Constant>(ov::element::i64, ov::Shape{shape_in.size()}, shape_in);
        auto reshape_in_node = std::make_shared<Reshape>(params[0], reshape_in_const, false);

        const std::vector<size_t> gather_ids = m_gather_ids_ref;
        auto gather_const_ids = Constant::create(ov::element::i64, ov::Shape{gather_ids.size()}, gather_ids);
        const size_t gather_axis = 1;
        auto gather_const_axis = Constant::create(ov::element::i64, ov::Shape{}, {gather_axis});
        auto gather_node = std::make_shared<Gather>(reshape_in_node, gather_const_ids, gather_const_axis);

        ov::Shape shape_out = m_output_shape;
        auto reshape_out_const = std::make_shared<Constant>(ov::element::i64, ov::Shape{shape_out.size()}, shape_out);
        auto reshape_out_node = std::make_shared<Reshape>(gather_node, reshape_out_const, false);

        ov::ResultVector results{std::make_shared<Result>(reshape_out_node)};
        m_model_ref = std::make_shared<ov::Model>(results, params, "transpose_gather_test_model");
    }
};

TEST_P(TransposeGatherTest, CompareWithRefs) {
    Run();
}

TEST_P(GatherTransposeTest, CompareWithRefs) {
    Run();
}

ov::element::TypeVector input_precisions = {ov::element::f16, ov::element::f32};

const std::vector<ov::Shape> input_shapes = {{16, 64}, {1, 16, 64}, {1, 8, 16, 64}};

INSTANTIATE_TEST_SUITE_P(smoke_merge_transpose_gather,
                         TransposeGatherTest,
                         ::testing::Combine(::testing::ValuesIn(input_precisions), ::testing::ValuesIn(input_shapes)),
                         TransposeGatherTest::get_test_name);

INSTANTIATE_TEST_SUITE_P(smoke_merge_transpose_gather,
                         GatherTransposeTest,
                         ::testing::Combine(::testing::ValuesIn(input_precisions), ::testing::ValuesIn(input_shapes)),
                         GatherTransposeTest::get_test_name);

}  // namespace gather_transpose_merge_test
