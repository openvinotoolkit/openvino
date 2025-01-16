// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/group_normalization.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/serialize.hpp"
#include "transformations/common_optimizations/group_normalization_fusion.hpp"
#include "transformations/init_node_info.hpp"

using namespace testing;
using namespace ov;

template <element::Type_t T_act_elem,
          element::Type_t T_gn_gamma_elem = T_act_elem,
          element::Type_t T_gn_beta_elem = T_act_elem,
          element::Type_t T_in_gamma_elem = T_act_elem,
          element::Type_t T_in_beta_elem = T_act_elem>
class GroupNormalizationFusionTestsFixture
    : public ::testing::TestWithParam<
          std::tuple<bool, PartialShape, Shape, Shape, Shape, Shape, unsigned long long, float>> {
public:
    static constexpr element::Type_t T_act_elem_t = T_act_elem;
    static constexpr element::Type_t T_gn_gamma_elem_t = T_gn_gamma_elem;
    static constexpr element::Type_t T_gn_beta_elem_t = T_gn_beta_elem;
    static constexpr element::Type_t T_in_gamma_elem_t = T_in_gamma_elem;
    static constexpr element::Type_t T_in_beta_elem_t = T_in_beta_elem;

    typedef typename ov::element_type_traits<T_act_elem_t>::value_type T_act_store_t;
    typedef typename ov::element_type_traits<T_gn_gamma_elem_t>::value_type T_gn_gamma_store_t;
    typedef typename ov::element_type_traits<T_gn_beta_elem_t>::value_type T_gn_beta_store_t;
    typedef typename ov::element_type_traits<T_in_gamma_elem_t>::value_type T_in_gamma_store_t;
    typedef typename ov::element_type_traits<T_in_beta_elem_t>::value_type T_in_beta_store_t;

    void TestBody() override {
        auto params = GetParam();
        auto positive_test = std::get<0>(params);
        auto data_shape = std::get<1>(params);
        ASSERT_TRUE(data_shape[1].is_static());
        auto num_channels = static_cast<unsigned long long>(data_shape[1].get_max_length());
        auto instance_norm_gamma_shape = std::get<2>(params);
        auto instance_norm_beta_shape = std::get<3>(params);
        auto group_norm_gamma_shape = std::get<4>(params);
        auto group_norm_beta_shape = std::get<5>(params);
        auto num_groups = std::get<6>(params);
        auto epsilon = std::get<7>(params);

        if (positive_test) {
            if ((instance_norm_gamma_shape != Shape{}) && (shape_size(instance_norm_gamma_shape) != num_groups))
                FAIL() << "Unexpected shape of instance norm beta - expected either empty shape (which means that it "
                          "will not "
                          "be put in the graph) or shape with exactly num_groups elements that can be merged with the "
                          "result "
                          "of MVN.";

            if ((instance_norm_beta_shape != Shape{}) && (shape_size(instance_norm_beta_shape) != num_groups))
                FAIL() << "Unexpected shape of instance norm beta - expected either empty shape (which means that it "
                          "will not "
                          "be put in the graph) or shape with exactly num_groups elements that can be merged with the "
                          "result "
                          "of MVN.";

            if ((group_norm_gamma_shape != Shape{}) && (shape_size(group_norm_gamma_shape) != num_channels))
                FAIL()
                    << "Unexpected shape of group norm gamma - expected either empty shape (which means that it will "
                       "not "
                       "be put in the graph) or shape with exactly num_channels elements that can be merged with the "
                       "result "
                       "of instance norm.";

            if ((group_norm_beta_shape != Shape{}) && (shape_size(group_norm_gamma_shape) != num_channels))
                FAIL()
                    << "Unexpected shape of group norm beta - expected either empty shape (which means that it will "
                       "not "
                       "be put in the graph) or shape with exactly num_channels elements that can be merged with the "
                       "result "
                       "of instance norm.";
        }
        auto instance_norm_gamma_present = (instance_norm_gamma_shape != Shape{});
        auto instance_norm_beta_present = (instance_norm_beta_shape != Shape{});
        auto group_norm_beta_present = (group_norm_beta_shape != Shape{});
        auto group_norm_gamma_present = (group_norm_gamma_shape != Shape{});

        if (positive_test) {
            instance_norm_gamma_present =
                instance_norm_gamma_present && (shape_size(instance_norm_gamma_shape) == num_groups);
            instance_norm_beta_present =
                instance_norm_beta_present && (shape_size(instance_norm_beta_shape) == num_groups);
            group_norm_beta_present = group_norm_beta_present && (shape_size(group_norm_beta_shape) == num_channels);
            group_norm_gamma_present = group_norm_gamma_present && (shape_size(group_norm_gamma_shape) == num_channels);
        }

        auto instance_norm_gamma_vals = std::vector<T_in_gamma_store_t>();
        if (instance_norm_gamma_present)
            instance_norm_gamma_vals =
                test::utils::generateVector<T_in_gamma_elem_t>(shape_size(instance_norm_gamma_shape));

        auto instance_norm_beta_vals = std::vector<T_in_beta_store_t>();
        if (instance_norm_beta_present)
            instance_norm_beta_vals =
                test::utils::generateVector<T_in_beta_elem_t>(shape_size(instance_norm_beta_shape));

        auto group_norm_gamma_vals = std::vector<T_gn_gamma_store_t>();
        if (group_norm_gamma_present)
            group_norm_gamma_vals = test::utils::generateVector<T_gn_gamma_elem_t>(shape_size(group_norm_gamma_shape));

        auto group_norm_beta_vals = std::vector<T_in_beta_store_t>();
        if (group_norm_beta_present)
            group_norm_beta_vals = test::utils::generateVector<T_in_beta_elem_t>(shape_size(group_norm_beta_shape));

        std::shared_ptr<Model> model(nullptr), model_ref(nullptr);
        {
            auto input = std::make_shared<op::v0::Parameter>(T_act_elem_t, data_shape);
            auto pre_mvn_shape_const = op::v0::Constant::create<long long>(element::i64,
                                                                           Shape{3},
                                                                           {0, static_cast<long long>(num_groups), -1});
            auto pre_mvn_reshape = std::make_shared<ov::op::v1::Reshape>(input, pre_mvn_shape_const, true);

            auto mvn_axes_const = op::v0::Constant::create<long long>(element::i64, Shape{1}, {1});
            auto mvn = std::make_shared<op::v6::MVN>(pre_mvn_reshape,
                                                     mvn_axes_const,
                                                     true,
                                                     epsilon,
                                                     op::MVNEpsMode::INSIDE_SQRT);

            std::shared_ptr<Node> opt_instance_norm_gamma_multiply = mvn;
            if (instance_norm_gamma_present) {
                auto instance_norm_gamma_const =
                    op::v0::Constant::create(T_in_gamma_elem_t, instance_norm_gamma_shape, instance_norm_gamma_vals);
                opt_instance_norm_gamma_multiply = std::make_shared<op::v1::Multiply>(mvn, instance_norm_gamma_const);
            }

            std::shared_ptr<ov::Node> opt_instance_norm_beta_add = opt_instance_norm_gamma_multiply;
            if (instance_norm_beta_present) {
                auto instance_norm_beta_const =
                    op::v0::Constant::create(T_in_beta_elem_t, instance_norm_beta_shape, instance_norm_beta_vals);
                opt_instance_norm_beta_add =
                    std::make_shared<ov::op::v1::Add>(opt_instance_norm_gamma_multiply, instance_norm_beta_const);
            }

            auto post_instance_norm_shape = std::make_shared<ov::op::v0::ShapeOf>(input);

            auto post_instance_norm_reshape =
                std::make_shared<op::v1::Reshape>(opt_instance_norm_beta_add, post_instance_norm_shape, true);

            std::shared_ptr<ov::Node> opt_group_norm_gamma_multiply = post_instance_norm_reshape;
            if (group_norm_gamma_present) {
                auto group_norm_gamma_const =
                    op::v0::Constant::create(T_gn_gamma_elem_t, group_norm_gamma_shape, group_norm_gamma_vals);
                opt_group_norm_gamma_multiply =
                    std::make_shared<op::v1::Multiply>(post_instance_norm_reshape, group_norm_gamma_const);
            }

            std::shared_ptr<ov::Node> opt_group_norm_beta_add = opt_group_norm_gamma_multiply;
            if (group_norm_beta_present) {
                auto group_norm_beta_const =
                    op::v0::Constant::create(T_gn_beta_elem_t, group_norm_beta_shape, group_norm_beta_vals);
                opt_group_norm_beta_add =
                    std::make_shared<op::v1::Add>(opt_group_norm_gamma_multiply, group_norm_beta_const);
            }

            model = std::make_shared<Model>(NodeVector{opt_group_norm_beta_add}, ParameterVector{input});

            pass::Manager m;
            m.register_pass<ov::pass::GroupNormalizationFusion>();
            OV_ASSERT_NO_THROW(m.run_passes(model));
        }

        if (positive_test) {
            auto input = std::make_shared<ov::op::v0::Parameter>(T_act_elem_t, data_shape);

            std::shared_ptr<ov::Node> group_norm_beta_1d = nullptr;
            std::shared_ptr<ov::Node> group_norm_gamma_1d = nullptr;

            if (instance_norm_gamma_present) {
                if (!group_norm_gamma_present)
                    group_norm_gamma_vals = std::vector<T_gn_gamma_store_t>(num_channels, 1);
                auto group_norm_gamma_corr_vals = group_norm_gamma_vals;
                for (auto i = 0; i < group_norm_gamma_corr_vals.size(); i++)
                    group_norm_gamma_corr_vals[i] /= instance_norm_gamma_vals[i % num_groups];
                group_norm_gamma_1d =
                    op::v0::Constant::create(T_gn_gamma_elem_t, Shape{num_channels}, group_norm_gamma_corr_vals);
                if (instance_norm_beta_present) {
                    if (!group_norm_beta_present)
                        group_norm_beta_vals = std::vector<T_gn_beta_store_t>(num_channels, 0);
                    auto group_norm_beta_corr_vals = group_norm_beta_vals;
                    for (auto i = 0; i < group_norm_beta_corr_vals.size(); i++)
                        group_norm_beta_corr_vals[i] -=
                            (group_norm_gamma_corr_vals[i] * instance_norm_beta_vals[i % num_groups]) /
                            instance_norm_gamma_vals[i % num_groups];
                    group_norm_beta_1d =
                        op::v0::Constant::create(T_gn_beta_elem_t, Shape{num_channels}, group_norm_beta_corr_vals);
                }
            } else {
                if (instance_norm_beta_present) {
                    if (!group_norm_beta_present)
                        group_norm_beta_vals = std::vector<T_gn_beta_store_t>(num_channels, 0);
                    auto group_norm_beta_corr_vals = group_norm_beta_vals;
                    for (auto i = 0; i < group_norm_beta_corr_vals.size(); i++)
                        group_norm_beta_corr_vals[i] -=
                            group_norm_gamma_vals[i] * instance_norm_beta_vals[i % num_groups];
                    group_norm_beta_1d =
                        op::v0::Constant::create(T_gn_beta_elem_t, Shape{num_channels}, group_norm_beta_corr_vals);
                }
            }

            if (group_norm_gamma_present) {
                if (group_norm_gamma_1d == nullptr) {
                    group_norm_gamma_1d =
                        op::v0::Constant::create(T_gn_gamma_elem_t, Shape{num_channels}, group_norm_gamma_vals);
                }
            } else {
                group_norm_gamma_1d = op::v0::Constant::create(T_gn_gamma_elem_t,
                                                               Shape{num_channels},
                                                               std::vector<T_gn_gamma_store_t>(num_channels, 1));
            }

            if (group_norm_beta_present) {
                if (group_norm_beta_1d == nullptr) {
                    group_norm_beta_1d =
                        op::v0::Constant::create(T_gn_beta_elem_t, Shape{num_channels}, group_norm_beta_vals);
                }
            } else {
                group_norm_beta_1d = op::v0::Constant::create(T_gn_beta_elem_t,
                                                              Shape{num_channels},
                                                              std::vector<T_gn_beta_store_t>(num_channels, 0));
            }

            auto group_norm = std::make_shared<ov::op::v12::GroupNormalization>(input,
                                                                                group_norm_gamma_1d,
                                                                                group_norm_beta_1d,
                                                                                num_groups,
                                                                                epsilon);

            model_ref = std::make_shared<Model>(NodeVector{group_norm}, ParameterVector{input});
        }

        if (positive_test) {
            ASSERT_EQ(count_ops_of_type<ov::op::v12::GroupNormalization>(model), 1);
            auto fc = FunctionsComparator::no_default().enable(FunctionsComparator::ACCURACY);
            auto res = fc.compare(model, model_ref);
            ASSERT_TRUE(res.valid) << res.message;
        } else {
            ASSERT_EQ(count_ops_of_type<ov::op::v12::GroupNormalization>(model), 0);
        }
    }
};

class GroupNormalizationFusionTestsFixture_f16 : public GroupNormalizationFusionTestsFixture<element::Type_t::f16> {};
class GroupNormalizationFusionTestsFixture_bf16 : public GroupNormalizationFusionTestsFixture<element::Type_t::bf16> {};
class GroupNormalizationFusionTestsFixture_f32 : public GroupNormalizationFusionTestsFixture<element::Type_t::f32> {};
class GroupNormalizationFusionTestsFixture_u8 : public GroupNormalizationFusionTestsFixture<element::Type_t::u8> {};
class GroupNormalizationFusionTestsFixture_u16 : public GroupNormalizationFusionTestsFixture<element::Type_t::u16> {};
class GroupNormalizationFusionTestsFixture_u32 : public GroupNormalizationFusionTestsFixture<element::Type_t::u32> {};
class GroupNormalizationFusionTestsFixture_u64 : public GroupNormalizationFusionTestsFixture<element::Type_t::u64> {};
class GroupNormalizationFusionTestsFixture_i8 : public GroupNormalizationFusionTestsFixture<element::Type_t::i8> {};
class GroupNormalizationFusionTestsFixture_i16 : public GroupNormalizationFusionTestsFixture<element::Type_t::i16> {};
class GroupNormalizationFusionTestsFixture_i32 : public GroupNormalizationFusionTestsFixture<element::Type_t::i32> {};
class GroupNormalizationFusionTestsFixture_i64 : public GroupNormalizationFusionTestsFixture<element::Type_t::i64> {};
class GroupNormalizationFusionTestsFixture_f8e4m3
    : public GroupNormalizationFusionTestsFixture<element::Type_t::f8e4m3> {};
class GroupNormalizationFusionTestsFixture_f8e5m2
    : public GroupNormalizationFusionTestsFixture<element::Type_t::f8e5m2> {};
class GroupNormalizationFusionTestsFixture_f4e2m1
    : public GroupNormalizationFusionTestsFixture<element::Type_t::f4e2m1> {};
class GroupNormalizationFusionTestsFixture_f8e8m0
    : public GroupNormalizationFusionTestsFixture<element::Type_t::f8e8m0> {};

TEST_P(GroupNormalizationFusionTestsFixture_f16, GroupNormalizationFusionTests_f16) {
    GroupNormalizationFusionTestsFixture_f16::TestBody();
}

TEST_P(GroupNormalizationFusionTestsFixture_bf16, GroupNormalizationFusionTests_bf16) {
    GroupNormalizationFusionTestsFixture_bf16::TestBody();
}

TEST_P(GroupNormalizationFusionTestsFixture_f32, GroupNormalizationFusionTests_f32) {
    GroupNormalizationFusionTestsFixture_f32::TestBody();
}

TEST_P(GroupNormalizationFusionTestsFixture_u8, GroupNormalizationFusionTests_u8) {
    GroupNormalizationFusionTestsFixture_u8::TestBody();
}

TEST_P(GroupNormalizationFusionTestsFixture_u16, GroupNormalizationFusionTests_u16) {
    GroupNormalizationFusionTestsFixture_u16::TestBody();
}

TEST_P(GroupNormalizationFusionTestsFixture_u32, GroupNormalizationFusionTests_u32) {
    GroupNormalizationFusionTestsFixture_u32::TestBody();
}

TEST_P(GroupNormalizationFusionTestsFixture_u64, GroupNormalizationFusionTests_u64) {
    GroupNormalizationFusionTestsFixture_u64::TestBody();
}

TEST_P(GroupNormalizationFusionTestsFixture_i8, GroupNormalizationFusionTests_i8) {
    GroupNormalizationFusionTestsFixture_i8::TestBody();
}

TEST_P(GroupNormalizationFusionTestsFixture_i16, GroupNormalizationFusionTests_i16) {
    GroupNormalizationFusionTestsFixture_i16::TestBody();
}

TEST_P(GroupNormalizationFusionTestsFixture_i32, GroupNormalizationFusionTests_i32) {
    GroupNormalizationFusionTestsFixture_i32::TestBody();
}

TEST_P(GroupNormalizationFusionTestsFixture_i64, GroupNormalizationFusionTests_i64) {
    GroupNormalizationFusionTestsFixture_i64::TestBody();
}

TEST_P(GroupNormalizationFusionTestsFixture_f8e4m3, GroupNormalizationFusionTests_f8e4m3) {
    GroupNormalizationFusionTestsFixture_f8e4m3::TestBody();
}

TEST_P(GroupNormalizationFusionTestsFixture_f8e5m2, GroupNormalizationFusionTests_f8e5m2) {
    GroupNormalizationFusionTestsFixture_f8e5m2::TestBody();
}

TEST_P(GroupNormalizationFusionTestsFixture_f4e2m1, GroupNormalizationFusionTests_f4e2m1) {
    GroupNormalizationFusionTestsFixture_f4e2m1::TestBody();
}

TEST_P(GroupNormalizationFusionTestsFixture_f8e8m0, GroupNormalizationFusionTests_f8e8m0) {
    GroupNormalizationFusionTestsFixture_f8e8m0::TestBody();
}

using RawValuesContainer = std::tuple<PartialShape, Shape, Shape, Shape, Shape, unsigned long long, float>;
using ValuesContainerWithPositiveTestFlag =
    std::tuple<bool, PartialShape, Shape, Shape, Shape, Shape, unsigned long long, float>;

std::vector<RawValuesContainer> valid_vals = {
    std::make_tuple(Shape{1, 320}, Shape{}, Shape{}, Shape{320}, Shape{320}, 1, 1e-5f),
    std::make_tuple(Shape{1, 320, 2, 2},
                    Shape{1, 1, 1},
                    Shape{1, 1, 1},
                    Shape{320, 1, 1},
                    Shape{1, 320, 1, 1},
                    1,
                    1e-5f),
    std::make_tuple(Shape{1, 320, 2, 2},
                    Shape{1, 320, 1},
                    Shape{1, 320, 1},
                    Shape{320, 1, 1},
                    Shape{320, 1, 1},
                    320,
                    1e-5f),
    std::make_tuple(PartialShape{Dimension::dynamic(), 320, Dimension::dynamic(), Dimension::dynamic()},
                    Shape{1, 320, 1},
                    Shape{1, 320, 1},
                    Shape{320, 1, 1},
                    Shape{320, 1, 1},
                    320,
                    1e-5f),
    std::make_tuple(PartialShape{Dimension::dynamic(), 320},
                    Shape{32, 1},
                    Shape{32, 1},
                    Shape{320},
                    Shape{320},
                    32,
                    1e-5f),
    std::make_tuple(PartialShape{1, 320, Dimension::dynamic()},
                    Shape{32, 1},
                    Shape{32, 1},
                    Shape{320, 1},
                    Shape{320, 1},
                    32,
                    1e-5f),
    std::make_tuple(PartialShape{1, 320, 2, Dimension::dynamic()},
                    Shape{1, 32, 1},
                    Shape{1, 32, 1},
                    Shape{320, 1, 1},
                    Shape{320, 1, 1},
                    32,
                    1e-5f),
    std::make_tuple(Shape{2, 320, 4, 8}, Shape{}, Shape{}, Shape{320, 1, 1}, Shape{1, 320, 1, 1}, 32, 1e-5f),
    std::make_tuple(PartialShape{1, 512, Dimension::dynamic(), Dimension::dynamic()},
                    Shape{},
                    Shape{1, 128, 1},
                    Shape{1, 512, 1, 1},
                    Shape{512, 1, 1},
                    128,
                    1e-6f),
    std::make_tuple(Shape{1, 512, 2, 2},
                    Shape{1, 64, 1},
                    Shape{},
                    Shape{1, 512, 1, 1},
                    Shape{1, 512, 1, 1},
                    64,
                    1e-6f)};

auto invalid_vals = ::testing::Values(
    std::make_tuple(false, Shape{1, 320}, Shape{}, Shape{}, Shape{}, Shape{}, 1, 1e-5f),
    std::make_tuple(false,
                    Shape{1, 320, 2, 2},
                    Shape{1, 1, 1},
                    Shape{1, 1, 1},
                    Shape{1, 1, 1},
                    Shape{1, 1, 1, 1},
                    1,
                    1e-5f),
    std::make_tuple(false, Shape{1, 320, 2, 2}, Shape{}, Shape{}, Shape{320, 1, 1}, Shape{}, 1, 1e-5f),
    std::make_tuple(false, Shape{1, 320, 2, 2}, Shape{}, Shape{}, Shape{}, Shape{1, 320, 1, 1}, 1, 1e-5f),
    std::make_tuple(false,
                    Shape{1, 320, 2, 2},
                    Shape{1, 1, 1},
                    Shape{1, 32, 1},
                    Shape{320, 1, 1},
                    Shape{320, 1, 1},
                    32,
                    1e-5f),
    std::make_tuple(false,
                    Shape{1, 320, 2, 2},
                    Shape{1, 32, 1},
                    Shape{1, 1, 1},
                    Shape{320, 1, 1},
                    Shape{320, 1, 1},
                    32,
                    1e-5f),
    std::make_tuple(false,
                    PartialShape{Dimension::dynamic(), 512, Dimension::dynamic(), Dimension::dynamic()},
                    Shape{},
                    Shape{},
                    Shape{1, 512, 1, 1},
                    Shape{1, 512, 1, 1},
                    100,
                    1e-6f));

std::vector<ValuesContainerWithPositiveTestFlag> add_positive_test_flag_to_vals(
    const bool positive_test,
    const std::vector<RawValuesContainer>& vals) {
    std::vector<ValuesContainerWithPositiveTestFlag> res;
    for (const RawValuesContainer& t : vals) {
        auto new_val = std::tuple_cat(std::tuple<bool>(positive_test), t);
        res.push_back(new_val);
    }
    return res;
}

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionPositiveTests_f16,
                         GroupNormalizationFusionTestsFixture_f16,
                         ::testing::ValuesIn(add_positive_test_flag_to_vals(true, valid_vals)));

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionNegativeTestsInvalidVals_f16,
                         GroupNormalizationFusionTestsFixture_f16,
                         invalid_vals);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionPositiveTests_bf16,
                         GroupNormalizationFusionTestsFixture_bf16,
                         ::testing::ValuesIn(add_positive_test_flag_to_vals(true, valid_vals)));

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionNegativeTestsInvalidVals_bf16,
                         GroupNormalizationFusionTestsFixture_bf16,
                         invalid_vals);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionPositiveTests_f32,
                         GroupNormalizationFusionTestsFixture_f32,
                         ::testing::ValuesIn(add_positive_test_flag_to_vals(true, valid_vals)));

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionNegativeTests_f32,
                         GroupNormalizationFusionTestsFixture_f32,
                         invalid_vals);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionNegativeTestsValidVals_u8,
                         GroupNormalizationFusionTestsFixture_u8,
                         ::testing::ValuesIn(add_positive_test_flag_to_vals(false, valid_vals)));

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionNegativeTestsInvalidVals_u8,
                         GroupNormalizationFusionTestsFixture_u8,
                         invalid_vals);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionNegativeTestsValidVals_u16,
                         GroupNormalizationFusionTestsFixture_u16,
                         ::testing::ValuesIn(add_positive_test_flag_to_vals(false, valid_vals)));

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionNegativeTestsInvalidVals_u16,
                         GroupNormalizationFusionTestsFixture_u16,
                         invalid_vals);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionNegativeTestsValidVals_u32,
                         GroupNormalizationFusionTestsFixture_u32,
                         ::testing::ValuesIn(add_positive_test_flag_to_vals(false, valid_vals)));

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionNegativeTestsInvalidVals_u32,
                         GroupNormalizationFusionTestsFixture_u32,
                         invalid_vals);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionNegativeTestsValidVals_u64,
                         GroupNormalizationFusionTestsFixture_u64,
                         ::testing::ValuesIn(add_positive_test_flag_to_vals(false, valid_vals)));

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionNegativeTestsInvalidVals_u64,
                         GroupNormalizationFusionTestsFixture_u64,
                         invalid_vals);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionNegativeTestsValidVals_i8,
                         GroupNormalizationFusionTestsFixture_i8,
                         ::testing::ValuesIn(add_positive_test_flag_to_vals(false, valid_vals)));

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionNegativeTestsInvalidVals_i8,
                         GroupNormalizationFusionTestsFixture_i8,
                         invalid_vals);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionNegativeTestsValidVals_i16,
                         GroupNormalizationFusionTestsFixture_i16,
                         ::testing::ValuesIn(add_positive_test_flag_to_vals(false, valid_vals)));

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionNegativeTestsInvalidVals_i16,
                         GroupNormalizationFusionTestsFixture_i16,
                         invalid_vals);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionNegativeTestsValidVals_i32,
                         GroupNormalizationFusionTestsFixture_i32,
                         ::testing::ValuesIn(add_positive_test_flag_to_vals(false, valid_vals)));

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionNegativeTestsInvalidVals_i32,
                         GroupNormalizationFusionTestsFixture_i32,
                         invalid_vals);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionNegativeTestsValidVals_i64,
                         GroupNormalizationFusionTestsFixture_i64,
                         ::testing::ValuesIn(add_positive_test_flag_to_vals(false, valid_vals)));

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionNegativeTestsInvalidVals_i64,
                         GroupNormalizationFusionTestsFixture_i64,
                         invalid_vals);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionNegativeTestsValidVals_f8e4m3,
                         GroupNormalizationFusionTestsFixture_f8e4m3,
                         ::testing::ValuesIn(add_positive_test_flag_to_vals(false, valid_vals)));

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionNegativeTestsInvalidVals_f8e4m3,
                         GroupNormalizationFusionTestsFixture_f8e4m3,
                         invalid_vals);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionNegativeTestsValidVals_f8e5m2,
                         GroupNormalizationFusionTestsFixture_f8e5m2,
                         ::testing::ValuesIn(add_positive_test_flag_to_vals(false, valid_vals)));

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionNegativeTestsInvalidVals_f8e5m2,
                         GroupNormalizationFusionTestsFixture_f8e5m2,
                         invalid_vals);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionNegativeTestsValidVals_f4e2m1,
                         GroupNormalizationFusionTestsFixture_f4e2m1,
                         ::testing::ValuesIn(add_positive_test_flag_to_vals(false, valid_vals)));

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionNegativeTestsInvalidVals_f4e2m1,
                         GroupNormalizationFusionTestsFixture_f4e2m1,
                         invalid_vals);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionNegativeTestsValidVals_f8e8m0,
                         GroupNormalizationFusionTestsFixture_f8e8m0,
                         ::testing::ValuesIn(add_positive_test_flag_to_vals(false, valid_vals)));

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionNegativeTestsInvalidVals_f8e8m0,
                         GroupNormalizationFusionTestsFixture_f8e8m0,
                         invalid_vals);