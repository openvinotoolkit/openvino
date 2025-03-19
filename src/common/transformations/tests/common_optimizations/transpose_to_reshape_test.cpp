// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/transpose_to_reshape.hpp"

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
#include "transformations/common_optimizations/transpose_sinking.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

using namespace testing;
using namespace ov;
using namespace std;

using InputShape = PartialShape;
using TransposeOrder = std::vector<int64_t>;

struct ReferenceParams {
    bool no_changes = false;
    bool is_empty = false;
    std::vector<int64_t> reshape_value;

    ReferenceParams() = default;

    explicit ReferenceParams(bool no_changes, bool is_empty) : no_changes(no_changes), is_empty(is_empty) {}

    explicit ReferenceParams(const std::vector<int64_t>& reshape_value) : reshape_value(reshape_value) {}
};

class TransposeToReshapeTests
    : public ov::test::TestsCommon,
      public testing::WithParamInterface<std::tuple<InputShape, TransposeOrder, ReferenceParams>> {
public:
    std::shared_ptr<ov::Model> f, f_ref;

    void SetUp() override {
        const auto& input_shape = std::get<0>(GetParam());
        const auto& transpose_order = std::get<1>(GetParam());
        const auto& reference_params = std::get<2>(GetParam());

        f = get_initial_function(input_shape, transpose_order);
        f_ref = get_reference_function(input_shape, transpose_order, reference_params);
    }

private:
    std::shared_ptr<ov::Model> get_initial_function(const PartialShape& input_shape,
                                                    const std::vector<int64_t>& transpose_order) {
        auto data = std::make_shared<opset3::Parameter>(element::f32, input_shape);
        auto order_const = opset3::Constant::create(element::i64, Shape{transpose_order.size()}, transpose_order);
        auto transpose = std::make_shared<opset3::Transpose>(data, order_const);

        // WA to test cases with transpose elimination
        auto relu = std::make_shared<opset3::Relu>(transpose);

        return std::make_shared<ov::Model>(NodeVector{relu}, ParameterVector{data});
    }

    std::shared_ptr<ov::Model> get_reference_function(const PartialShape& input_shape,
                                                      const std::vector<int64_t>& transpose_order,
                                                      const ReferenceParams& params) {
        if (params.no_changes) {
            return get_initial_function(input_shape, transpose_order);
        }

        auto data = std::make_shared<opset3::Parameter>(element::f32, input_shape);

        Output<Node> reshape_dims, last(data);
        if (!params.reshape_value.empty()) {
            reshape_dims =
                opset3::Constant::create(element::i64, Shape{params.reshape_value.size()}, params.reshape_value);
        } else {
            auto shape_of = std::make_shared<opset3::ShapeOf>(data);
            reshape_dims = std::make_shared<opset3::Gather>(
                shape_of,
                opset3::Constant::create(element::i64, Shape{transpose_order.size()}, transpose_order),
                opset3::Constant::create(element::i64, Shape{1}, {0}));
        }

        if (!params.is_empty) {
            last = std::make_shared<opset3::Reshape>(last, reshape_dims, true);
        }

        last = std::make_shared<opset3::Relu>(last);

        return std::make_shared<ov::Model>(NodeVector{last.get_node_shared_ptr()}, ParameterVector{data});
    }
};

TEST_P(TransposeToReshapeTests, CompareFunctions) {
    auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();
    pass::Manager m;
    m.register_pass<ov::pass::InitUniqueNames>(unh);
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::pass::TransposeToReshape>();
    m.register_pass<ov::pass::CheckUniqueNames>(unh);
    m.run_passes(f);
    f->validate_nodes_and_infer_types();
    OV_ASSERT_NO_THROW(check_rt_info(f));

    auto fc =
        FunctionsComparator::no_default().enable(FunctionsComparator::NODES).enable(FunctionsComparator::PRECISIONS);
    auto res = fc.compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

#define SAME_FUNCTION   ReferenceParams(true, false)
#define EMPTY_FUNCTION  ReferenceParams(false, true)
#define SHAPE_OF_GATHER ReferenceParams()

INSTANTIATE_TEST_SUITE_P(
    KeepTranspose,
    TransposeToReshapeTests,
    testing::Values(std::make_tuple(InputShape{1, 3, 64, 64}, TransposeOrder{0, 1, 3, 2}, SAME_FUNCTION),
                    std::make_tuple(InputShape{1, 3, 1, 64}, TransposeOrder{2, 0, 3, 1}, SAME_FUNCTION),
                    std::make_tuple(InputShape{1, 3, 1, 3}, TransposeOrder{3, 0, 2, 1}, SAME_FUNCTION),
                    std::make_tuple(InputShape{DYN, 2, 64, 1}, TransposeOrder{1, 0, 3, 2}, SAME_FUNCTION),
                    std::make_tuple(InputShape{DYN, 3}, TransposeOrder{1, 0}, SAME_FUNCTION),
                    std::make_tuple(InputShape{DYN, DYN, 1}, TransposeOrder{2, 1, 0}, SAME_FUNCTION),
                    std::make_tuple(InputShape{DYN, DYN}, TransposeOrder{1, 0}, SAME_FUNCTION)));

INSTANTIATE_TEST_SUITE_P(
    EliminateTranspose,
    TransposeToReshapeTests,
    testing::Values(std::make_tuple(InputShape{1, 3, 64, 64}, TransposeOrder{0, 1, 2, 3}, EMPTY_FUNCTION),
                    std::make_tuple(InputShape{1, 1, 1}, TransposeOrder{2, 0, 1}, EMPTY_FUNCTION),
                    std::make_tuple(InputShape{DYN, DYN}, TransposeOrder{0, 1}, EMPTY_FUNCTION)));

INSTANTIATE_TEST_SUITE_P(
    ReshapeWithConstant,
    TransposeToReshapeTests,
    testing::Values(
        std::make_tuple(InputShape{1, 3, 64, 1}, TransposeOrder{0, 1, 3, 2}, ReferenceParams({1, 3, 1, 64})),
        std::make_tuple(InputShape{1, 3, 1, 64}, TransposeOrder{1, 0, 3, 2}, ReferenceParams({3, 1, 64, 1})),
        std::make_tuple(InputShape{DYN, DYN, 1}, TransposeOrder{0, 2, 1}, ReferenceParams({0, 1, -1})),
        std::make_tuple(InputShape{1, 1, DYN}, TransposeOrder{2, 1, 0}, ReferenceParams({-1, 0, 1})),
        std::make_tuple(InputShape{DYN, 1, 64, 1}, TransposeOrder{1, 0, 3, 2}, ReferenceParams({1, -1, 1, 64}))));

INSTANTIATE_TEST_SUITE_P(
    ReshapeWithGather,
    TransposeToReshapeTests,
    testing::Values(std::make_tuple(InputShape{DYN, 1, DYN, 1}, TransposeOrder{1, 0, 3, 2}, SHAPE_OF_GATHER),
                    std::make_tuple(InputShape{1, DYN, DYN, DYN}, TransposeOrder{1, 2, 3, 0}, SHAPE_OF_GATHER)));

#undef SAME_FUNCTION
#undef EMPTY_FUNCTION
#undef SHAPE_OF_GATHER

TEST(TransformationTests, replace_transpose_with_reshape) {
    auto check_usecase = [](const PartialShape& shape,
                            const std::vector<int64_t>& perm_val,
                            bool i32,
                            bool multiout,
                            size_t num) {
        static size_t id = 0;
        auto casename = string("usecase #") + to_string(++id);

        shared_ptr<Node> perm;
        if (i32) {
            std::vector<int32_t> perm_val_i32(perm_val.size());
            std::transform(perm_val.begin(), perm_val.end(), perm_val_i32.begin(), [](int64_t x) {
                return (int32_t)x;
            });
            perm = op::v0::Constant::create<int32_t>(element::i32, Shape{perm_val.size()}, perm_val_i32);
        } else {
            perm = op::v0::Constant::create<int64_t>(element::i64, Shape{perm_val.size()}, perm_val);
        }
        auto param = make_shared<op::v0::Parameter>(element::f32, shape);
        shared_ptr<Node> A1;
        if (multiout) {
            shared_ptr<Node> k;
            auto last_dim = shape.rank().get_length() - 1;
            if (shape[last_dim].is_dynamic()) {
                k = make_shared<op::v1::Gather>(make_shared<op::v3::ShapeOf>(param),
                                                op::v0::Constant::create(element::i64, {}, {last_dim}),
                                                op::v0::Constant::create(element::i64, {}, {0}));
            } else {
                k = make_shared<op::v0::Constant>(element::i64,
                                                  Shape{},
                                                  std::vector<int64_t>{shape[last_dim].get_length()});
            }
            A1 = make_shared<op::v1::TopK>(param, k, last_dim, op::v1::TopK::Mode::MAX, op::v1::TopK::SortType::NONE);
        } else {
            A1 = make_shared<op::v0::Abs>(param);
        }
        auto transpose = make_shared<op::v1::Transpose>((multiout ? A1->output(0) : A1), perm);
        auto transpose1 = make_shared<op::v0::Abs>(transpose);
        auto baseline_f = make_shared<Model>(transpose1, ParameterVector{param});
        auto optimized_f = baseline_f->clone();

        auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();
        pass::Manager m;
        m.register_pass<ov::pass::InitUniqueNames>(unh);
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::Validate>();
        m.register_pass<ov::pass::TransposeToReshape>();
        m.register_pass<ov::pass::CheckUniqueNames>(unh);
        m.run_passes(optimized_f);

        auto ps = baseline_f->get_results()[0]->get_output_partial_shape(0);
        auto ps_r = optimized_f->get_results()[0]->get_output_partial_shape(0);
        EXPECT_TRUE(ps.rank().is_static() && ps_r.rank().is_static()) << casename;
        ASSERT_EQ(ps.rank().get_length(), ps_r.rank().get_length()) << casename;

        ASSERT_EQ(count_ops_of_type<op::v1::Transpose>(baseline_f), 1);
        ASSERT_EQ(count_ops_of_type<op::v1::Reshape>(baseline_f), 0);
        ASSERT_EQ(count_ops_of_type<op::v1::Transpose>(optimized_f), num);
        ASSERT_EQ(count_ops_of_type<op::v1::Reshape>(optimized_f), (num ? 0 : 1));
    };

    for (auto& i32 : {true, false})
        for (auto& multiout : {true, false}) {
            check_usecase(Shape{1, 3}, vector<int64_t>{1, 0}, i32, multiout, 0);
            check_usecase(Shape{2, 3, 1}, vector<int64_t>{2, 0, 1}, i32, multiout, 0);
            check_usecase(Shape{10, 20, 1, 1}, vector<int64_t>{0, 2, 3, 1}, i32, multiout, 0);
            check_usecase(Shape{10, 1, 1, 20}, vector<int64_t>{0, 3, 1, 2}, i32, multiout, 0);
            check_usecase(Shape{10, 20, 1, 2}, vector<int64_t>{0, 2, 1, 3}, i32, multiout, 0);
            check_usecase(Shape{10, 1, 1, 1, 20}, vector<int64_t>{0, 4, 1, 2, 3}, i32, multiout, 0);
            check_usecase(Shape{10, 20, 1, 1, 1}, vector<int64_t>{0, 2, 3, 4, 1}, i32, multiout, 0);
            check_usecase(Shape{10, 1, 1, 1, 1}, vector<int64_t>{1, 4, 2, 3, 0}, i32, multiout, 0);
            check_usecase(Shape{10, 1, 1, 1, 1}, vector<int64_t>{4, 2, 0, 1, 3}, i32, multiout, 0);
            check_usecase(Shape{10, 20, 1, 2}, vector<int64_t>{0, 2, 3, 1}, i32, multiout, 1);
            check_usecase(Shape{10, 20, 1, 2}, vector<int64_t>{0, 3, 1, 2}, i32, multiout, 1);
            check_usecase(Shape{10, 20}, vector<int64_t>{1, 0}, i32, multiout, 1);

            check_usecase(PartialShape{Dimension::dynamic(), 20, 1, 1},
                          vector<int64_t>{
                              0,
                              2,
                              3,
                              1,
                          },
                          i32,
                          multiout,
                          0);
            check_usecase(PartialShape{Dimension::dynamic(), Dimension::dynamic(), 20, 1, 1},
                          vector<int64_t>{0, 1, 3, 2, 4},
                          i32,
                          multiout,
                          0);
            check_usecase(PartialShape{Dimension::dynamic(), Dimension::dynamic(), 20, 1, 1},
                          vector<int64_t>{0, 2, 1, 4, 3},
                          i32,
                          multiout,
                          1);
        }
}
