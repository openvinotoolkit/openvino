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

class GroupNormalizationFusionValueParametrizedTestsFixture
    : public ::testing::TestWithParam<
          std::tuple<bool, PartialShape, Shape, Shape, Shape, Shape, unsigned long long, float>> {};

TEST_P(GroupNormalizationFusionValueParametrizedTestsFixture, GroupNormalizationFusionTestValueParametrizedTests) {
    auto params = GetParam();
    typedef ov::float16 T_act_t;
    constexpr auto T_act_elem_t = element::from<T_act_t>();
    typedef ov::element_type_traits<T_act_elem_t>::value_type T_act_store_t;
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
            FAIL()
                << "Unexpected shape of instance norm beta - expected either empty shape (which means that it will not "
                   "be put in the graph) or shape with exactly num_groups elements that can be merged with the result "
                   "of MVN.";

        if ((instance_norm_beta_shape != Shape{}) && (shape_size(instance_norm_beta_shape) != num_groups))
            FAIL()
                << "Unexpected shape of instance norm beta - expected either empty shape (which means that it will not "
                   "be put in the graph) or shape with exactly num_groups elements that can be merged with the result "
                   "of MVN.";

        if ((group_norm_gamma_shape != Shape{}) && (shape_size(group_norm_gamma_shape) != num_channels))
            FAIL()
                << "Unexpected shape of group norm gamma - expected either empty shape (which means that it will not "
                   "be put in the graph) or shape with exactly num_channels elements that can be merged with the "
                   "result "
                   "of instance norm.";

        if ((group_norm_beta_shape != Shape{}) && (shape_size(group_norm_gamma_shape) != num_channels))
            FAIL() << "Unexpected shape of group norm beta - expected either empty shape (which means that it will not "
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
        instance_norm_beta_present = instance_norm_beta_present && (shape_size(instance_norm_beta_shape) == num_groups);
        group_norm_beta_present = group_norm_beta_present && (shape_size(group_norm_beta_shape) == num_channels);
        group_norm_gamma_present = group_norm_gamma_present && (shape_size(group_norm_gamma_shape) == num_channels);
    }

    auto instance_norm_gamma_vals = std::vector<T_act_store_t>();
    if (instance_norm_gamma_present)
        instance_norm_gamma_vals = test::utils::generateVector<T_act_elem_t>(shape_size(instance_norm_gamma_shape));

    auto instance_norm_beta_vals = std::vector<T_act_store_t>();
    if (instance_norm_beta_present)
        instance_norm_beta_vals = test::utils::generateVector<T_act_elem_t>(shape_size(instance_norm_beta_shape));

    auto group_norm_gamma_vals = std::vector<T_act_store_t>();
    if (group_norm_gamma_present)
        group_norm_gamma_vals = test::utils::generateVector<T_act_elem_t>(shape_size(group_norm_gamma_shape));

    auto group_norm_beta_vals = std::vector<T_act_store_t>();
    if (group_norm_beta_present)
        group_norm_beta_vals = test::utils::generateVector<T_act_elem_t>(shape_size(group_norm_beta_shape));

    std::shared_ptr<Model> model(nullptr), model_ref(nullptr);
    {
        auto input = std::make_shared<op::v0::Parameter>(T_act_elem_t, data_shape);
        auto pre_mvn_shape_const =
            op::v0::Constant::create<long long>(element::i64, Shape{3}, {0, static_cast<long long>(num_groups), -1});
        auto pre_mvn_reshape = std::make_shared<ov::op::v1::Reshape>(input, pre_mvn_shape_const, true);

        auto mvn_axes_const = op::v0::Constant::create<long long>(element::i64, Shape{1}, {1});
        auto mvn =
            std::make_shared<op::v6::MVN>(pre_mvn_reshape, mvn_axes_const, true, epsilon, op::MVNEpsMode::INSIDE_SQRT);

        std::shared_ptr<Node> opt_instance_norm_gamma_multiply = mvn;
        if (instance_norm_gamma_present) {
            auto instance_norm_gamma_const =
                op::v0::Constant::create(T_act_elem_t, instance_norm_gamma_shape, instance_norm_gamma_vals);
            opt_instance_norm_gamma_multiply = std::make_shared<op::v1::Multiply>(mvn, instance_norm_gamma_const);
        }

        std::shared_ptr<ov::Node> opt_instance_norm_beta_add = opt_instance_norm_gamma_multiply;
        if (instance_norm_beta_present) {
            auto instance_norm_beta_const =
                op::v0::Constant::create(T_act_elem_t, instance_norm_beta_shape, instance_norm_beta_vals);
            opt_instance_norm_beta_add =
                std::make_shared<ov::op::v1::Add>(opt_instance_norm_gamma_multiply, instance_norm_beta_const);
        }

        auto post_instance_norm_shape = std::make_shared<ov::op::v0::ShapeOf>(input);

        auto post_instance_norm_reshape =
            std::make_shared<op::v1::Reshape>(opt_instance_norm_beta_add, post_instance_norm_shape, true);

        std::shared_ptr<ov::Node> opt_group_norm_gamma_multiply = post_instance_norm_reshape;
        if (group_norm_gamma_present) {
            auto group_norm_gamma_const =
                op::v0::Constant::create(T_act_elem_t, group_norm_gamma_shape, group_norm_gamma_vals);
            opt_group_norm_gamma_multiply =
                std::make_shared<op::v1::Multiply>(post_instance_norm_reshape, group_norm_gamma_const);
        }

        std::shared_ptr<ov::Node> opt_group_norm_beta_add = opt_group_norm_gamma_multiply;
        if (group_norm_beta_present) {
            auto group_norm_beta_const =
                op::v0::Constant::create(T_act_elem_t, group_norm_beta_shape, group_norm_beta_vals);
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
                group_norm_gamma_vals = std::vector<T_act_store_t>(num_channels, 1);
            auto group_norm_gamma_corr_vals = group_norm_gamma_vals;
            for (auto i = 0; i < group_norm_gamma_corr_vals.size(); i++)
                group_norm_gamma_corr_vals[i] /= instance_norm_gamma_vals[i % num_groups];
            group_norm_gamma_1d =
                op::v0::Constant::create(T_act_elem_t, Shape{num_channels}, group_norm_gamma_corr_vals);
            if (instance_norm_beta_present) {
                if (!group_norm_beta_present)
                    group_norm_beta_vals = std::vector<T_act_store_t>(num_channels, 0);
                auto group_norm_beta_corr_vals = group_norm_beta_vals;
                for (auto i = 0; i < group_norm_beta_corr_vals.size(); i++)
                    group_norm_beta_corr_vals[i] -=
                        (group_norm_gamma_corr_vals[i] * instance_norm_beta_vals[i % num_groups]) /
                        instance_norm_gamma_vals[i % num_groups];
                group_norm_beta_1d =
                    op::v0::Constant::create(T_act_elem_t, Shape{num_channels}, group_norm_beta_corr_vals);
            }
        } else {
            if (instance_norm_beta_present) {
                if (!group_norm_beta_present)
                    group_norm_beta_vals = std::vector<T_act_store_t>(num_channels, 0);
                auto group_norm_beta_corr_vals = group_norm_beta_vals;
                for (auto i = 0; i < group_norm_beta_corr_vals.size(); i++)
                    group_norm_beta_corr_vals[i] -= group_norm_gamma_vals[i] * instance_norm_beta_vals[i % num_groups];
                group_norm_beta_1d =
                    op::v0::Constant::create(T_act_elem_t, Shape{num_channels}, group_norm_beta_corr_vals);
            }
        }

        if (group_norm_gamma_present) {
            if (group_norm_gamma_1d == nullptr) {
                group_norm_gamma_1d =
                    op::v0::Constant::create(T_act_elem_t, Shape{num_channels}, group_norm_gamma_vals);
            }
        } else {
            group_norm_gamma_1d = op::v0::Constant::create(T_act_elem_t,
                                                           Shape{num_channels},
                                                           std::vector<T_act_store_t>(num_channels, 1));
        }

        if (group_norm_beta_present) {
            if (group_norm_beta_1d == nullptr) {
                group_norm_beta_1d = op::v0::Constant::create(T_act_elem_t, Shape{num_channels}, group_norm_beta_vals);
            }
        } else {
            group_norm_beta_1d = op::v0::Constant::create(T_act_elem_t,
                                                          Shape{num_channels},
                                                          std::vector<T_act_store_t>(num_channels, 0));
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

INSTANTIATE_TEST_SUITE_P(
    GroupNormalizationFusionValueParametrizedPositiveTests,
    GroupNormalizationFusionValueParametrizedTestsFixture,
    ::testing::Values(
        std::make_tuple(true, Shape{1, 320}, Shape{}, Shape{}, Shape{320}, Shape{320}, 1, 1e-5f),
        std::make_tuple(true,
                        Shape{1, 320, 2, 2},
                        Shape{1, 1, 1},
                        Shape{1, 1, 1},
                        Shape{320, 1, 1},
                        Shape{1, 320, 1, 1},
                        1,
                        1e-5f),
        std::make_tuple(true,
                        Shape{1, 320, 2, 2},
                        Shape{1, 320, 1},
                        Shape{1, 320, 1},
                        Shape{320, 1, 1},
                        Shape{320, 1, 1},
                        320,
                        1e-5f),
        std::make_tuple(true,
                        PartialShape{Dimension::dynamic(), 320, Dimension::dynamic(), Dimension::dynamic()},
                        Shape{1, 320, 1},
                        Shape{1, 320, 1},
                        Shape{320, 1, 1},
                        Shape{320, 1, 1},
                        320,
                        1e-5f),
        std::make_tuple(true,
                        PartialShape{Dimension::dynamic(), 320},
                        Shape{32, 1},
                        Shape{32, 1},
                        Shape{320},
                        Shape{320},
                        32,
                        1e-5f),
        std::make_tuple(true,
                        PartialShape{1, 320, Dimension::dynamic()},
                        Shape{32, 1},
                        Shape{32, 1},
                        Shape{320, 1},
                        Shape{320, 1},
                        32,
                        1e-5f),
        std::make_tuple(true,
                        PartialShape{1, 320, 2, Dimension::dynamic()},
                        Shape{1, 32, 1},
                        Shape{1, 32, 1},
                        Shape{320, 1, 1},
                        Shape{320, 1, 1},
                        32,
                        1e-5f),
        std::make_tuple(true, Shape{2, 320, 4, 8}, Shape{}, Shape{}, Shape{320, 1, 1}, Shape{1, 320, 1, 1}, 32, 1e-5f),
        std::make_tuple(true,
                        PartialShape{1, 512, Dimension::dynamic(), Dimension::dynamic()},
                        Shape{},
                        Shape{1, 128, 1},
                        Shape{1, 512, 1, 1},
                        Shape{512, 1, 1},
                        128,
                        1e-6f),
        std::make_tuple(true,
                        Shape{1, 512, 2, 2},
                        Shape{1, 64, 1},
                        Shape{},
                        Shape{1, 512, 1, 1},
                        Shape{1, 512, 1, 1},
                        64,
                        1e-6f)));

INSTANTIATE_TEST_SUITE_P(
    GroupNormalizationFusionValueParametrizedNegativeTests,
    GroupNormalizationFusionValueParametrizedTestsFixture,
    ::testing::Values(
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
                        1e-6f)));

template <bool positive_test,
          typename T_act,
          typename T_gn_gamma = T_act,
          typename T_gn_beta = T_act,
          typename T_in_gamma = T_act,
          typename T_in_beta = T_act>
class GroupNormalizationFusionTestMultiType {
public:
    constexpr static bool positive_test = positive_test;
    using T_act_t = T_act;
    using T_gn_gamma_t = T_gn_gamma;
    using T_gn_beta_t = T_gn_beta;
    using T_in_gamma_t = T_in_gamma;
    using T_in_beta_t = T_in_beta;
};

template <typename T>
class GroupNormalizationFusionTypeParametrizedTestsFixture : public ::testing::Test {};

using GroupNormalizationFusionPositiveTestTypes =
    ::testing::Types<GroupNormalizationFusionTestMultiType<true, float>,
                     GroupNormalizationFusionTestMultiType<true, ov::float16>,
                     GroupNormalizationFusionTestMultiType<true, ov::bfloat16>>;

using GroupNormalizationFusionNegativeTestTypes =
    ::testing::Types<GroupNormalizationFusionTestMultiType<false, int8_t>,
                     GroupNormalizationFusionTestMultiType<false, int16_t>,
                     GroupNormalizationFusionTestMultiType<false, int32_t>,
                     GroupNormalizationFusionTestMultiType<false, int64_t>,
                     GroupNormalizationFusionTestMultiType<false, uint8_t>,
                     GroupNormalizationFusionTestMultiType<false, uint16_t>,
                     GroupNormalizationFusionTestMultiType<false, uint32_t>,
                     GroupNormalizationFusionTestMultiType<false, uint64_t>,
                     GroupNormalizationFusionTestMultiType<false, ov::float8_e4m3>,
                     GroupNormalizationFusionTestMultiType<false, ov::float8_e5m2>,
                     GroupNormalizationFusionTestMultiType<false, ov::float4_e2m1>,
                     GroupNormalizationFusionTestMultiType<false, ov::float8_e8m0>>;

TYPED_TEST_SUITE_P(GroupNormalizationFusionTypeParametrizedTestsFixture);

TYPED_TEST_P(GroupNormalizationFusionTypeParametrizedTestsFixture, GroupNormalizationFusionTypeParametrizedTests) {
    constexpr bool positive_test = TypeParam::positive_test;

    typedef TypeParam::T_act_t T_act_t;
    typedef TypeParam::T_gn_gamma_t T_gn_gamma_t;
    typedef TypeParam::T_gn_beta_t T_gn_beta_t;
    typedef TypeParam::T_in_gamma_t T_in_gamma_t;
    typedef TypeParam::T_in_beta_t T_in_beta_t;

    constexpr auto T_act_elem_t = element::from<T_act_t>();
    constexpr auto T_gn_gamma_elem_t = element::from<T_gn_gamma_t>();
    constexpr auto T_gn_beta_elem_t = element::from<T_gn_beta_t>();
    constexpr auto T_in_gamma_elem_t = element::from<T_in_gamma_t>();
    constexpr auto T_in_beta_elem_t = element::from<T_in_beta_t>();

    typedef ov::element_type_traits<T_act_elem_t>::value_type T_act_store_t;
    typedef ov::element_type_traits<T_gn_gamma_elem_t>::value_type T_gn_gamma_store_t;
    typedef ov::element_type_traits<T_gn_beta_elem_t>::value_type T_gn_beta_store_t;
    typedef ov::element_type_traits<T_in_gamma_elem_t>::value_type T_in_gamma_store_t;
    typedef ov::element_type_traits<T_in_beta_elem_t>::value_type T_in_beta_store_t;

    auto data_shape = Shape{1, 320, 2, 2};
    auto instance_norm_gamma_shape = Shape{1, 32, 1};
    auto instance_norm_beta_shape = Shape{1, 32, 1};
    auto group_norm_gamma_shape = Shape{1, 320, 1, 1};
    auto group_norm_beta_shape = Shape{1, 320, 1, 1};

    auto num_channels = 320ull;
    auto num_groups = 32;
    auto epsilon = 1e-5f;

    auto instance_norm_gamma_vals =
        test::utils::generateVector<T_in_gamma_elem_t>(shape_size(instance_norm_gamma_shape));
    auto instance_norm_beta_vals = test::utils::generateVector<T_in_beta_elem_t>(shape_size(instance_norm_beta_shape));
    auto group_norm_gamma_vals = test::utils::generateVector<T_gn_gamma_elem_t>(shape_size(group_norm_gamma_shape));
    auto group_norm_beta_vals = test::utils::generateVector<T_gn_beta_elem_t>(shape_size(group_norm_beta_shape));

    std::shared_ptr<Model> model(nullptr), model_ref(nullptr);
    {
        auto input = std::make_shared<op::v0::Parameter>(T_act_elem_t, data_shape);
        auto pre_mvn_shape_const =
            op::v0::Constant::create<long long>(element::i64, Shape{3}, {0, static_cast<long long>(num_groups), -1});
        auto pre_mvn_reshape = std::make_shared<ov::op::v1::Reshape>(input, pre_mvn_shape_const, true);

        auto mvn_axes_const = op::v0::Constant::create<long long>(element::i64, Shape{1}, {1});
        auto mvn =
            std::make_shared<op::v6::MVN>(pre_mvn_reshape, mvn_axes_const, true, epsilon, op::MVNEpsMode::INSIDE_SQRT);

        auto instance_norm_gamma_const =
            op::v0::Constant::create(T_in_gamma_elem_t, instance_norm_gamma_shape, instance_norm_gamma_vals);
        auto instance_norm_gamma_multiply = std::make_shared<op::v1::Multiply>(mvn, instance_norm_gamma_const);

        auto instance_norm_beta_const =
            op::v0::Constant::create(T_in_beta_elem_t, instance_norm_beta_shape, instance_norm_beta_vals);
        auto instance_norm_beta_add =
            std::make_shared<ov::op::v1::Add>(instance_norm_gamma_multiply, instance_norm_beta_const);

        auto post_instance_norm_shape = std::make_shared<ov::op::v0::ShapeOf>(input);

        auto post_instance_norm_reshape =
            std::make_shared<op::v1::Reshape>(instance_norm_beta_add, post_instance_norm_shape, true);

        auto group_norm_gamma_const =
            op::v0::Constant::create(T_gn_gamma_elem_t, group_norm_gamma_shape, group_norm_gamma_vals);
        auto group_norm_gamma_multiply =
            std::make_shared<op::v1::Multiply>(post_instance_norm_reshape, group_norm_gamma_const);

        auto group_norm_beta_const =
            op::v0::Constant::create(T_gn_beta_elem_t, group_norm_beta_shape, group_norm_beta_vals);
        auto group_norm_beta_add = std::make_shared<op::v1::Add>(group_norm_gamma_multiply, group_norm_beta_const);

        model = std::make_shared<Model>(NodeVector{group_norm_beta_add}, ParameterVector{input});

        pass::Manager m;
        m.register_pass<ov::pass::GroupNormalizationFusion>();
        OV_ASSERT_NO_THROW(m.run_passes(model));
    }

    if (positive_test) {
        auto input = std::make_shared<ov::op::v0::Parameter>(T_act_elem_t, data_shape);

        auto group_norm_gamma_corr_vals = group_norm_gamma_vals;

        for (auto i = 0; i < group_norm_gamma_corr_vals.size(); i++)
            group_norm_gamma_corr_vals[i] /= instance_norm_gamma_vals[i % num_groups];

        auto group_norm_gamma_1d =
            op::v0::Constant::create(T_gn_gamma_elem_t, Shape{num_channels}, group_norm_gamma_corr_vals);

        auto group_norm_beta_corr_vals = group_norm_beta_vals;
        for (auto i = 0; i < group_norm_beta_corr_vals.size(); i++)
            group_norm_beta_corr_vals[i] -= (group_norm_gamma_corr_vals[i] * instance_norm_beta_vals[i % num_groups]) /
                                            instance_norm_gamma_vals[i % num_groups];
        auto group_norm_beta_1d =
            op::v0::Constant::create(T_gn_beta_elem_t, Shape{num_channels}, group_norm_beta_corr_vals);

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

REGISTER_TYPED_TEST_SUITE_P(GroupNormalizationFusionTypeParametrizedTestsFixture,
                            GroupNormalizationFusionTypeParametrizedTests);

INSTANTIATE_TYPED_TEST_SUITE_P(GroupNormalizationFusionTypeParametrizedPositiveTests,
                               GroupNormalizationFusionTypeParametrizedTestsFixture,
                               GroupNormalizationFusionPositiveTestTypes);

INSTANTIATE_TYPED_TEST_SUITE_P(GroupNormalizationFusionTypeParametrizedNegativeTests,
                               GroupNormalizationFusionTypeParametrizedTestsFixture,
                               GroupNormalizationFusionNegativeTestTypes);