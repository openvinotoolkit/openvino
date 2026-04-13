// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/group_normalization.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "transformations/common_optimizations/group_normalization_fusion.hpp"

using namespace testing;

namespace v1 = ov::op::v1;
namespace ov {
namespace test {

using GroupNormalizationFusionTransformationTestValues =
    std::tuple<PartialShape,   // (partial) shape of input/output tensor (all dims except channel can be dynamic)
               Shape,          // shape of optional instance norm gamma tensor (or empty shape if not used)
               Shape,          // shape of optional instance norm beta tensor (or empty shape if not used)
               Shape,          // shape of group norm gamma tensor
               Shape,          // shape of group norm beta tensor
               int64_t,        // number of groups
               double,         // epsilon
               element::Type,  // input/output tensor element type
               bool>;          // whether it's a positive test that should run reference model or a negative test

using GroupNormalizationFusionTestBaseValues =
    std::tuple<PartialShape,  // (partial) shape of input/output tensor (all dims except channel can be dynamic)
               Shape,         // shape of optional instance norm gamma tensor (or empty shape if not used)
               Shape,         // shape of optional instance norm beta tensor (or empty shape if not used)
               Shape,         // shape of group norm gamma tensor
               Shape,         // shape of group norm beta tensor
               int64_t,       // number of groups
               double>;       // epsilon

class GroupNormalizationFusionTestBase {
protected:
    element::Type elem_type;
    int64_t num_channels;
    bool instance_norm_gamma_present;
    bool instance_norm_beta_present;

    std::shared_ptr<op::v0::Constant> instance_norm_gamma_const;
    std::shared_ptr<op::v0::Constant> instance_norm_beta_const;
    std::shared_ptr<op::v0::Constant> group_norm_gamma_const;
    std::shared_ptr<op::v0::Constant> group_norm_beta_const;

    PartialShape data_shape;
    Shape instance_norm_gamma_shape;
    Shape instance_norm_beta_shape;
    Shape group_norm_gamma_shape;
    Shape group_norm_beta_shape;
    int64_t num_groups;
    double epsilon;

    virtual void read_test_parameters() = 0;

    void generate_weights_init_values() {
        if (instance_norm_gamma_present) {
            auto instanceNormGammaTensor = utils::create_and_fill_tensor(elem_type,
                                                                         instance_norm_gamma_shape,
                                                                         utils::InputGenerateData(1, 10, 1, 2));
            instance_norm_gamma_const = std::make_shared<op::v0::Constant>(instanceNormGammaTensor);
        }
        if (instance_norm_beta_present) {
            auto instanceNormBetaTensor = utils::create_and_fill_tensor(elem_type,
                                                                        instance_norm_beta_shape,
                                                                        utils::InputGenerateData(1, 10, 1, 3));
            instance_norm_beta_const = std::make_shared<op::v0::Constant>(instanceNormBetaTensor);
        }

        auto groupNormGammaTensor =
            utils::create_and_fill_tensor(elem_type, group_norm_gamma_shape, utils::InputGenerateData(1, 10, 1, 1));
        group_norm_gamma_const = std::make_shared<op::v0::Constant>(groupNormGammaTensor);

        auto groupNormBetaTensor =
            utils::create_and_fill_tensor(elem_type, group_norm_beta_shape, utils::InputGenerateData(1, 10, 1, 11));
        group_norm_beta_const = std::make_shared<op::v0::Constant>(groupNormBetaTensor);
    }

    std::shared_ptr<Model> create_model() {
        auto input = std::make_shared<op::v0::Parameter>(elem_type, data_shape);
        auto pre_mvn_shape_const = op::v0::Constant::create<long long>(element::i64, Shape{3}, {0, num_groups, -1});
        auto pre_mvn_reshape = std::make_shared<op::v1::Reshape>(input, pre_mvn_shape_const, true);

        auto mvn_axes_const = op::v0::Constant::create<long long>(element::i64, Shape{1}, {2});
        auto mvn = std::make_shared<op::v6::MVN>(pre_mvn_reshape,
                                                 mvn_axes_const,
                                                 true,
                                                 static_cast<float>(epsilon),
                                                 op::MVNEpsMode::INSIDE_SQRT);

        std::shared_ptr<Node> opt_instance_norm_gamma_multiply = mvn;
        if (instance_norm_gamma_present)
            opt_instance_norm_gamma_multiply = std::make_shared<op::v1::Multiply>(mvn, instance_norm_gamma_const);

        std::shared_ptr<Node> opt_instance_norm_beta_add = opt_instance_norm_gamma_multiply;
        if (instance_norm_beta_present)
            opt_instance_norm_beta_add =
                std::make_shared<op::v1::Add>(opt_instance_norm_gamma_multiply, instance_norm_beta_const);

        auto post_instance_norm_shape = std::make_shared<op::v0::ShapeOf>(input);
        auto post_instance_norm_reshape =
            std::make_shared<op::v1::Reshape>(opt_instance_norm_beta_add, post_instance_norm_shape, true);
        auto group_norm_gamma_multiply =
            std::make_shared<op::v1::Multiply>(post_instance_norm_reshape, group_norm_gamma_const);
        auto group_norm_beta_add = std::make_shared<op::v1::Add>(group_norm_gamma_multiply, group_norm_beta_const);

        return std::make_shared<Model>(OutputVector{group_norm_beta_add}, ParameterVector{input});
    }
};

class GroupNormalizationFusionTransformationTestsF
    : public GroupNormalizationFusionTestBase,
      public TransformationTestsF,
      public testing::WithParamInterface<GroupNormalizationFusionTransformationTestValues> {
public:
    static std::string getTestCaseName(
        const testing::TestParamInfo<GroupNormalizationFusionTransformationTestValues>& obj) {
        const auto& params = obj.param;

        const auto& data_shape = std::get<0>(params);
        const auto& instance_norm_gamma_shape = std::get<1>(params);
        const auto& instance_norm_beta_shape = std::get<2>(params);
        const auto& group_norm_gamma_shape = std::get<3>(params);
        const auto& group_norm_beta_shape = std::get<4>(params);
        const auto& num_groups = std::get<5>(params);
        const auto& epsilon = std::get<6>(params);
        const auto& elem_type = std::get<7>(params);
        const auto& positive_test = std::get<8>(params);

        std::ostringstream results;

        results << "T=" << elem_type << "_";
        results << "Input=" << utils::partialShape2str({data_shape}) << "_";
        results << "InstNormGamma=" << utils::partialShape2str({instance_norm_gamma_shape}) << "_";
        results << "InstNormBeta=" << utils::partialShape2str({instance_norm_beta_shape}) << "_";
        results << "GroupNormGamma=" << utils::partialShape2str({group_norm_gamma_shape}) << "_";
        results << "GroupNormBeta=" << utils::partialShape2str({group_norm_beta_shape}) << "_";
        results << "NumGroups=" << num_groups << "_";
        results << "Epsilon=" << epsilon << "_";
        results << "PositiveTest=" << std::boolalpha << positive_test;

        return results.str();
    }

    void run() {
        read_test_parameters();
        generate_weights_init_values();
        model = create_model();
        manager.register_pass<pass::GroupNormalizationFusion>();

        if (positive_test) {
            model_ref = create_ref_model();
        } else {
            ASSERT_EQ(count_ops_of_type<op::v12::GroupNormalization>(model), 0);
            test_skipped = true;
        }
    }

protected:
    bool positive_test;

    void read_test_parameters() override {
        const auto& params = GetParam();

        data_shape = std::get<0>(params);
        if (!data_shape.rank().is_static())
            throw std::runtime_error("Rank of input tensor has to be static!");
        if (data_shape.rank().get_max_length() < 2)
            throw std::runtime_error("Expected at least two dimensions in input tensor!");
        if (!data_shape[1].is_static())
            throw std::runtime_error("Channel dimension in input tensor has to be static!");
        num_channels = data_shape[1].get_max_length();
        instance_norm_gamma_shape = std::get<1>(params);
        instance_norm_beta_shape = std::get<2>(params);
        group_norm_gamma_shape = std::get<3>(params);
        group_norm_beta_shape = std::get<4>(params);
        num_groups = std::get<5>(params);
        if (num_groups < 1)
            throw std::runtime_error("Number of groups has to be positive!");
        epsilon = std::get<6>(params);
        elem_type = std::get<7>(params);
        positive_test = std::get<8>(params);

        instance_norm_gamma_present = (instance_norm_gamma_shape != Shape{});
        instance_norm_beta_present = (instance_norm_beta_shape != Shape{});

        if (positive_test) {
            if ((instance_norm_gamma_shape != Shape{}) &&
                (shape_size(instance_norm_gamma_shape) != static_cast<size_t>(num_groups)))
                throw std::runtime_error("Shape of instance norm gamma has to either be empty or contain "
                                         "exactly <num_groups> elements");
            if ((instance_norm_beta_shape != Shape{}) &&
                (shape_size(instance_norm_beta_shape) != static_cast<size_t>(num_groups)))
                throw std::runtime_error("Shape of instance norm beta has to either be empty shape or contain "
                                         "exactly <num_groups> elements");
            if (shape_size(group_norm_gamma_shape) != static_cast<size_t>(num_channels))
                throw std::runtime_error("Shape of group norm gamma has to contain exactly <num_channels> elements");
            if (shape_size(group_norm_beta_shape) != static_cast<size_t>(num_channels))
                throw std::runtime_error("Shape of group norm beta has to contain exactly <num_channels> elements");

            instance_norm_gamma_present = instance_norm_gamma_present &&
                                          (shape_size(instance_norm_gamma_shape) == static_cast<size_t>(num_groups));
            instance_norm_beta_present =
                instance_norm_beta_present && (shape_size(instance_norm_beta_shape) == static_cast<size_t>(num_groups));
        }
    }

    std::shared_ptr<Model> create_ref_model() {
        auto input = std::make_shared<op::v0::Parameter>(elem_type, data_shape);

        auto shape_1d_const = op::v0::Constant::create(element::i64, Shape{1}, {1});
        auto gather_axis_const = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto gather_indices_vals = std::vector<int64_t>();
        for (auto i = 0ll; i < num_groups; i++)
            gather_indices_vals.insert(gather_indices_vals.end(), num_channels / num_groups, i);
        auto gather_indices_const =
            op::v0::Constant::create(element::i64, Shape{static_cast<size_t>(num_channels)}, gather_indices_vals);

        std::shared_ptr<Node> group_norm_gamma_1d = std::make_shared<op::v0::Squeeze>(group_norm_gamma_const);
        std::shared_ptr<Node> group_norm_beta_1d = std::make_shared<op::v0::Squeeze>(group_norm_beta_const);

        if (instance_norm_beta_present) {
            std::shared_ptr<Node> instance_norm_beta_1d = nullptr;

            if (shape_size(instance_norm_beta_shape) == 1) {
                instance_norm_beta_1d =
                    std::make_shared<op::v1::Reshape>(instance_norm_beta_const, shape_1d_const, true);
            } else {
                instance_norm_beta_1d = std::make_shared<op::v0::Squeeze>(instance_norm_beta_const);
            }

            instance_norm_beta_1d =
                std::make_shared<op::v8::Gather>(instance_norm_beta_1d, gather_indices_const, gather_axis_const);

            auto group_norm_beta_corr_multiply =
                std::make_shared<v1::Multiply>(group_norm_gamma_1d, instance_norm_beta_1d);
            group_norm_beta_1d = std::make_shared<v1::Add>(group_norm_beta_corr_multiply, group_norm_beta_1d);
        }

        if (instance_norm_gamma_present) {
            std::shared_ptr<Node> instance_norm_gamma_1d = nullptr;

            if (shape_size(instance_norm_gamma_shape) == 1) {
                instance_norm_gamma_1d =
                    std::make_shared<op::v1::Reshape>(instance_norm_gamma_const, shape_1d_const, true);
            } else {
                instance_norm_gamma_1d = std::make_shared<op::v0::Squeeze>(instance_norm_gamma_const);
            }

            instance_norm_gamma_1d =
                std::make_shared<op::v8::Gather>(instance_norm_gamma_1d, gather_indices_const, gather_axis_const);
            group_norm_gamma_1d = std::make_shared<v1::Multiply>(group_norm_gamma_1d, instance_norm_gamma_1d);
        }

        auto group_norm = std::make_shared<op::v12::GroupNormalization>(input,
                                                                        group_norm_gamma_1d,
                                                                        group_norm_beta_1d,
                                                                        num_groups,
                                                                        epsilon);

        return std::make_shared<Model>(OutputVector{group_norm}, ParameterVector{input});
    }
};

TEST_P(GroupNormalizationFusionTransformationTestsF, GroupNormalizationFusionTransformationTests) {
    GroupNormalizationFusionTransformationTestsF::run();
}

std::vector<GroupNormalizationFusionTestBaseValues> valid_vals = {
    std::make_tuple(PartialShape{1, 320}, Shape{}, Shape{}, Shape{320}, Shape{320}, 1, 1e-5),
    std::make_tuple(PartialShape{1, 320, 2, 2},
                    Shape{1, 1, 1},
                    Shape{1, 1, 1},
                    Shape{320, 1, 1},
                    Shape{1, 320, 1, 1},
                    1,
                    1e-5),
    std::make_tuple(PartialShape{5, 320, 2, 2, 2},
                    Shape{1, 320, 1},
                    Shape{1, 320, 1},
                    Shape{320, 1, 1, 1},
                    Shape{320, 1, 1, 1},
                    320,
                    1e-5),
    std::make_tuple(
        PartialShape{Dimension::dynamic(), 320, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()},
        Shape{1, 320, 1},
        Shape{1, 320, 1},
        Shape{320, 1, 1, 1},
        Shape{320, 1, 1, 1},
        320,
        1e-5),
    std::make_tuple(PartialShape{3, 320}, Shape{32, 1}, Shape{32, 1}, Shape{320}, Shape{320}, 32, 1e-5),
    std::make_tuple(PartialShape{2, 9, 4, 5, 6},
                    Shape{3, 1},
                    Shape{3, 1},
                    Shape{1, 9, 1, 1, 1},
                    Shape{1, 9, 1, 1, 1},
                    3,
                    1e-5),
    std::make_tuple(PartialShape{1, 320, 2, 4},
                    Shape{1, 32, 1},
                    Shape{1, 32, 1},
                    Shape{320, 1, 1},
                    Shape{320, 1, 1},
                    32,
                    1e-5),
    std::make_tuple(PartialShape{8, 320, 4, 8}, Shape{}, Shape{}, Shape{320, 1, 1}, Shape{1, 320, 1, 1}, 32, 1e-5),
    std::make_tuple(PartialShape{1, 512, 4, 8},
                    Shape{},
                    Shape{1, 128, 1},
                    Shape{1, 512, 1, 1},
                    Shape{512, 1, 1},
                    128,
                    1e-6),
    std::make_tuple(PartialShape{1, 192, 2, 2},
                    Shape{1, 64, 1},
                    Shape{},
                    Shape{1, 192, 1, 1},
                    Shape{1, 192, 1, 1},
                    64,
                    1e-6)};

std::vector<GroupNormalizationFusionTestBaseValues> invalid_vals = {
    std::make_tuple(PartialShape{1, 320}, Shape{}, Shape{}, Shape{}, Shape{}, 1, 1e-5),
    std::make_tuple(PartialShape{1, 320, 2, 2},
                    Shape{1, 1, 1},
                    Shape{1, 1, 1},
                    Shape{1, 1, 1},
                    Shape{1, 1, 1, 1},
                    1,
                    1e-5),
    std::make_tuple(PartialShape{1, 320, 2, 2}, Shape{}, Shape{}, Shape{320, 1, 1}, Shape{}, 1, 1e-5),
    std::make_tuple(PartialShape{1, 320, 2, 2}, Shape{}, Shape{}, Shape{}, Shape{1, 320, 1, 1}, 1, 1e-5),
    std::make_tuple(PartialShape{1, 320, 2, 2},
                    Shape{1, 1, 1},
                    Shape{1, 32, 1},
                    Shape{320, 1, 1},
                    Shape{320, 1, 1},
                    32,
                    1e-5),
    std::make_tuple(PartialShape{1, 320, 2, 2},
                    Shape{1, 32, 1},
                    Shape{1, 1, 1},
                    Shape{320, 1, 1},
                    Shape{320, 1, 1},
                    32,
                    1e-5),
    std::make_tuple(PartialShape{Dimension::dynamic(), 512, Dimension::dynamic(), Dimension::dynamic()},
                    Shape{},
                    Shape{},
                    Shape{1, 512, 1, 1},
                    Shape{1, 512, 1, 1},
                    100,
                    1e-6)};

using GroupNormalizationFusionTransformationTestAdditionalValues =
    std::tuple<element::Type,  // input/output tensor element type
               bool>;          // whether it's a positive test that should run reference model or a negative test

template <typename... T_old_vals, typename... T_added_vals>
std::vector<std::tuple<T_old_vals..., T_added_vals...>> expand_vals(std::vector<std::tuple<T_old_vals...>> old_vals,
                                                                    std::tuple<T_added_vals...> added_vals) {
    std::vector<std::tuple<T_old_vals..., T_added_vals...>> res;
    for (const std::tuple<T_old_vals...>& t : old_vals) {
        auto new_tuple = std::tuple_cat(t, added_vals);
        res.push_back(new_tuple);
    }
    return res;
}

INSTANTIATE_TEST_SUITE_P(
    GroupNormalizationFusionTransformationPositiveTests_f32,
    GroupNormalizationFusionTransformationTestsF,
    ValuesIn(expand_vals(valid_vals, GroupNormalizationFusionTransformationTestAdditionalValues(element::f32, true))),
    GroupNormalizationFusionTransformationTestsF::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    GroupNormalizationFusionTransformationPositiveTests_f16,
    GroupNormalizationFusionTransformationTestsF,
    ValuesIn(expand_vals(valid_vals, GroupNormalizationFusionTransformationTestAdditionalValues(element::f16, true))),
    GroupNormalizationFusionTransformationTestsF::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    GroupNormalizationFusionTransformationPositiveTests_bf16,
    GroupNormalizationFusionTransformationTestsF,
    ValuesIn(expand_vals(valid_vals, GroupNormalizationFusionTransformationTestAdditionalValues(element::bf16, true))),
    GroupNormalizationFusionTransformationTestsF::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionTransformationNegativeTests_f32,
                         GroupNormalizationFusionTransformationTestsF,
                         ValuesIn(expand_vals(invalid_vals,
                                              GroupNormalizationFusionTransformationTestAdditionalValues(element::f32,
                                                                                                         false))),
                         GroupNormalizationFusionTransformationTestsF::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionTransformationNegativeTests_f16,
                         GroupNormalizationFusionTransformationTestsF,
                         ValuesIn(expand_vals(invalid_vals,
                                              GroupNormalizationFusionTransformationTestAdditionalValues(element::f16,
                                                                                                         false))),
                         GroupNormalizationFusionTransformationTestsF::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionTransformationNegativeTests_bf16,
                         GroupNormalizationFusionTransformationTestsF,
                         ValuesIn(expand_vals(invalid_vals,
                                              GroupNormalizationFusionTransformationTestAdditionalValues(element::bf16,
                                                                                                         false))),
                         GroupNormalizationFusionTransformationTestsF::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    GroupNormalizationFusionTransformationNegativeTestsValidVals_u8,
    GroupNormalizationFusionTransformationTestsF,
    ValuesIn(expand_vals(valid_vals, GroupNormalizationFusionTransformationTestAdditionalValues(element::u8, false))),
    GroupNormalizationFusionTransformationTestsF::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    GroupNormalizationFusionTransformationNegativeTestsValidVals_u16,
    GroupNormalizationFusionTransformationTestsF,
    ValuesIn(expand_vals(valid_vals, GroupNormalizationFusionTransformationTestAdditionalValues(element::u16, false))),
    GroupNormalizationFusionTransformationTestsF::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    GroupNormalizationFusionTransformationNegativeTestsValidVals_u32,
    GroupNormalizationFusionTransformationTestsF,
    ValuesIn(expand_vals(valid_vals, GroupNormalizationFusionTransformationTestAdditionalValues(element::u32, false))),
    GroupNormalizationFusionTransformationTestsF::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    GroupNormalizationFusionTransformationNegativeTestsValidVals_u64,
    GroupNormalizationFusionTransformationTestsF,
    ValuesIn(expand_vals(valid_vals, GroupNormalizationFusionTransformationTestAdditionalValues(element::u64, false))),
    GroupNormalizationFusionTransformationTestsF::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    GroupNormalizationFusionTransformationNegativeTestsValidVals_i8,
    GroupNormalizationFusionTransformationTestsF,
    ValuesIn(expand_vals(valid_vals, GroupNormalizationFusionTransformationTestAdditionalValues(element::i8, false))),
    GroupNormalizationFusionTransformationTestsF::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    GroupNormalizationFusionTransformationNegativeTestsValidVals_i16,
    GroupNormalizationFusionTransformationTestsF,
    ValuesIn(expand_vals(valid_vals, GroupNormalizationFusionTransformationTestAdditionalValues(element::i16, false))),
    GroupNormalizationFusionTransformationTestsF::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    GroupNormalizationFusionTransformationNegativeTestsValidVals_i32,
    GroupNormalizationFusionTransformationTestsF,
    ValuesIn(expand_vals(valid_vals, GroupNormalizationFusionTransformationTestAdditionalValues(element::i32, false))),
    GroupNormalizationFusionTransformationTestsF::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    GroupNormalizationFusionTransformationNegativeTestsValidVals_f8e5m2,
    GroupNormalizationFusionTransformationTestsF,
    ValuesIn(expand_vals(valid_vals,
                         GroupNormalizationFusionTransformationTestAdditionalValues(element::f8e5m2, false))),
    GroupNormalizationFusionTransformationTestsF::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    GroupNormalizationFusionTransformationNegativeTestsValidVals_f4e2m1,
    GroupNormalizationFusionTransformationTestsF,
    ValuesIn(expand_vals(valid_vals,
                         GroupNormalizationFusionTransformationTestAdditionalValues(element::f4e2m1, false))),
    GroupNormalizationFusionTransformationTestsF::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    GroupNormalizationFusionTransformationNegativeTestsValidVals_f8e8m0,
    GroupNormalizationFusionTransformationTestsF,
    ValuesIn(expand_vals(valid_vals,
                         GroupNormalizationFusionTransformationTestAdditionalValues(element::f8e8m0, false))),
    GroupNormalizationFusionTransformationTestsF::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    GroupNormalizationFusionTransformationNegativeTestsInvalidVals_u8,
    GroupNormalizationFusionTransformationTestsF,
    ValuesIn(expand_vals(invalid_vals, GroupNormalizationFusionTransformationTestAdditionalValues(element::u8, false))),
    GroupNormalizationFusionTransformationTestsF::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionTransformationNegativeTestsInvalidVals_u16,
                         GroupNormalizationFusionTransformationTestsF,
                         ValuesIn(expand_vals(invalid_vals,
                                              GroupNormalizationFusionTransformationTestAdditionalValues(element::u16,
                                                                                                         false))),
                         GroupNormalizationFusionTransformationTestsF::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionTransformationNegativeTestsInvalidVals_u32,
                         GroupNormalizationFusionTransformationTestsF,
                         ValuesIn(expand_vals(invalid_vals,
                                              GroupNormalizationFusionTransformationTestAdditionalValues(element::u32,
                                                                                                         false))),
                         GroupNormalizationFusionTransformationTestsF::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionTransformationNegativeTestsInvalidVals_u64,
                         GroupNormalizationFusionTransformationTestsF,
                         ValuesIn(expand_vals(invalid_vals,
                                              GroupNormalizationFusionTransformationTestAdditionalValues(element::u64,
                                                                                                         false))),
                         GroupNormalizationFusionTransformationTestsF::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    GroupNormalizationFusionTransformationNegativeTestsInvalidVals_i8,
    GroupNormalizationFusionTransformationTestsF,
    ValuesIn(expand_vals(invalid_vals, GroupNormalizationFusionTransformationTestAdditionalValues(element::i8, false))),
    GroupNormalizationFusionTransformationTestsF::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionTransformationNegativeTestsInvalidVals_i16,
                         GroupNormalizationFusionTransformationTestsF,
                         ValuesIn(expand_vals(invalid_vals,
                                              GroupNormalizationFusionTransformationTestAdditionalValues(element::i16,
                                                                                                         false))),
                         GroupNormalizationFusionTransformationTestsF::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionTransformationNegativeTestsInvalidVals_i32,
                         GroupNormalizationFusionTransformationTestsF,
                         ValuesIn(expand_vals(invalid_vals,
                                              GroupNormalizationFusionTransformationTestAdditionalValues(element::i32,
                                                                                                         false))),
                         GroupNormalizationFusionTransformationTestsF::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    GroupNormalizationFusionTransformationNegativeTestsInalidVals_f8e5m2,
    GroupNormalizationFusionTransformationTestsF,
    ValuesIn(expand_vals(invalid_vals,
                         GroupNormalizationFusionTransformationTestAdditionalValues(element::f8e5m2, false))),
    GroupNormalizationFusionTransformationTestsF::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    GroupNormalizationFusionTransformationNegativeTestsInvalidVals_f4e2m1,
    GroupNormalizationFusionTransformationTestsF,
    ValuesIn(expand_vals(invalid_vals,
                         GroupNormalizationFusionTransformationTestAdditionalValues(element::f4e2m1, false))),
    GroupNormalizationFusionTransformationTestsF::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    GroupNormalizationFusionTransformationNegativeTestsInvalidVals_f8e8m0,
    GroupNormalizationFusionTransformationTestsF,
    ValuesIn(expand_vals(invalid_vals,
                         GroupNormalizationFusionTransformationTestAdditionalValues(element::f8e8m0, false))),
    GroupNormalizationFusionTransformationTestsF::getTestCaseName);

// 4D InstanceNormalization MVN Pattern Tests
// Pattern: Input -> Reshape{N,G,-1,1} -> MVN(axes={2,3}) -> [Mul] -> [Add] -> Reshape(original)
// This pattern appears when GroupNormalization is decomposed via InstanceNormalization
// and reshaped directly to 4D with a trailing unit dimension before MVN.

class GroupNormalizationFusion4DTestsF
    : public GroupNormalizationFusionTestBase,
      public TransformationTestsF,
      public testing::WithParamInterface<GroupNormalizationFusionTransformationTestValues> {
public:
    static std::string getTestCaseName(
        const testing::TestParamInfo<GroupNormalizationFusionTransformationTestValues>& obj) {
        const auto& params = obj.param;

        const auto& data_shape = std::get<0>(params);
        const auto& instance_norm_gamma_shape = std::get<1>(params);
        const auto& instance_norm_beta_shape = std::get<2>(params);
        const auto& group_norm_gamma_shape = std::get<3>(params);
        const auto& group_norm_beta_shape = std::get<4>(params);
        const auto& num_groups = std::get<5>(params);
        const auto& epsilon = std::get<6>(params);
        const auto& elem_type = std::get<7>(params);
        const auto& positive_test = std::get<8>(params);

        std::ostringstream results;

        results << "T=" << elem_type << "_";
        results << "Input=" << utils::partialShape2str({data_shape}) << "_";
        results << "InstNormGamma=" << utils::partialShape2str({instance_norm_gamma_shape}) << "_";
        results << "InstNormBeta=" << utils::partialShape2str({instance_norm_beta_shape}) << "_";
        results << "GroupNormGamma=" << utils::partialShape2str({group_norm_gamma_shape}) << "_";
        results << "GroupNormBeta=" << utils::partialShape2str({group_norm_beta_shape}) << "_";
        results << "NumGroups=" << num_groups << "_";
        results << "Epsilon=" << epsilon << "_";
        results << "PositiveTest=" << std::boolalpha << positive_test;

        return results.str();
    }

    void run() {
        read_test_parameters();
        generate_weights_init_values();
        model = create_model_4d();
        manager.register_pass<pass::GroupNormalizationFusion>();

        if (positive_test) {
            model_ref = create_ref_model();
        } else {
            ASSERT_EQ(count_ops_of_type<op::v12::GroupNormalization>(model), 0);
            test_skipped = true;
        }
    }

protected:
    bool positive_test;

    void read_test_parameters() override {
        const auto& params = GetParam();

        data_shape = std::get<0>(params);
        if (!data_shape.rank().is_static())
            throw std::runtime_error("Rank of input tensor has to be static!");
        if (data_shape.rank().get_max_length() < 2)
            throw std::runtime_error("Expected at least two dimensions in input tensor!");
        if (!data_shape[1].is_static())
            throw std::runtime_error("Channel dimension in input tensor has to be static!");
        num_channels = data_shape[1].get_max_length();
        instance_norm_gamma_shape = std::get<1>(params);
        instance_norm_beta_shape = std::get<2>(params);
        group_norm_gamma_shape = std::get<3>(params);
        group_norm_beta_shape = std::get<4>(params);
        num_groups = std::get<5>(params);
        if (num_groups < 1)
            throw std::runtime_error("Number of groups has to be positive!");
        epsilon = std::get<6>(params);
        elem_type = std::get<7>(params);
        positive_test = std::get<8>(params);

        instance_norm_gamma_present = (instance_norm_gamma_shape != Shape{});
        instance_norm_beta_present = (instance_norm_beta_shape != Shape{});

        if (positive_test) {
            if ((instance_norm_gamma_shape != Shape{}) &&
                (shape_size(instance_norm_gamma_shape) != static_cast<size_t>(num_groups)))
                throw std::runtime_error("Shape of instance norm gamma has to either be empty or contain "
                                         "exactly <num_groups> elements");
            if ((instance_norm_beta_shape != Shape{}) &&
                (shape_size(instance_norm_beta_shape) != static_cast<size_t>(num_groups)))
                throw std::runtime_error("Shape of instance norm beta has to either be empty shape or contain "
                                         "exactly <num_groups> elements");
            if (shape_size(group_norm_gamma_shape) != static_cast<size_t>(num_channels))
                throw std::runtime_error("Shape of group norm gamma has to contain exactly <num_channels> elements");
            if (shape_size(group_norm_beta_shape) != static_cast<size_t>(num_channels))
                throw std::runtime_error("Shape of group norm beta has to contain exactly <num_channels> elements");

            instance_norm_gamma_present = instance_norm_gamma_present &&
                                          (shape_size(instance_norm_gamma_shape) == static_cast<size_t>(num_groups));
            instance_norm_beta_present =
                instance_norm_beta_present && (shape_size(instance_norm_beta_shape) == static_cast<size_t>(num_groups));
        }
    }

    // Creates the 4D InstanceNorm pattern model
    std::shared_ptr<Model> create_model_4d() {
        auto input = std::make_shared<op::v0::Parameter>(elem_type, data_shape);

        // Reshape directly to 4D: {0, G, -1, 1}
        auto pre_mvn_shape_const_4d =
            op::v0::Constant::create<long long>(element::i64, Shape{4}, {0, num_groups, -1, 1});
        auto pre_mvn_reshape_4d = std::make_shared<op::v1::Reshape>(input, pre_mvn_shape_const_4d, true);

        // MVN with axes={2,3}
        auto mvn_axes_const = op::v0::Constant::create<long long>(element::i64, Shape{2}, {2, 3});
        auto mvn = std::make_shared<op::v6::MVN>(pre_mvn_reshape_4d,
                                                 mvn_axes_const,
                                                 true,
                                                 static_cast<float>(epsilon),
                                                 op::MVNEpsMode::INSIDE_SQRT);

        std::shared_ptr<Node> opt_instance_norm_gamma_multiply = mvn;
        if (instance_norm_gamma_present) {
            opt_instance_norm_gamma_multiply = std::make_shared<op::v1::Multiply>(mvn, instance_norm_gamma_const);
        }

        std::shared_ptr<Node> opt_instance_norm_beta_add = opt_instance_norm_gamma_multiply;
        if (instance_norm_beta_present) {
            opt_instance_norm_beta_add =
                std::make_shared<op::v1::Add>(opt_instance_norm_gamma_multiply, instance_norm_beta_const);
        }

        // Reshape back to original shape
        auto post_instance_norm_shape = std::make_shared<op::v0::ShapeOf>(input);
        auto post_instance_norm_reshape =
            std::make_shared<op::v1::Reshape>(opt_instance_norm_beta_add, post_instance_norm_shape, true);

        auto group_norm_gamma_multiply =
            std::make_shared<op::v1::Multiply>(post_instance_norm_reshape, group_norm_gamma_const);
        auto group_norm_beta_add = std::make_shared<op::v1::Add>(group_norm_gamma_multiply, group_norm_beta_const);

        return std::make_shared<Model>(OutputVector{group_norm_beta_add}, ParameterVector{input});
    }

    // Reference model - same as the 3D tests
    std::shared_ptr<Model> create_ref_model() {
        auto input = std::make_shared<op::v0::Parameter>(elem_type, data_shape);

        auto shape_1d_const = op::v0::Constant::create(element::i64, Shape{1}, {1});
        auto gather_axis_const = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto gather_indices_vals = std::vector<int64_t>();
        for (auto i = 0ll; i < num_groups; i++)
            gather_indices_vals.insert(gather_indices_vals.end(), num_channels / num_groups, i);
        auto gather_indices_const =
            op::v0::Constant::create(element::i64, Shape{static_cast<size_t>(num_channels)}, gather_indices_vals);

        std::shared_ptr<Node> group_norm_gamma_1d = std::make_shared<op::v0::Squeeze>(group_norm_gamma_const);
        std::shared_ptr<Node> group_norm_beta_1d = std::make_shared<op::v0::Squeeze>(group_norm_beta_const);

        if (instance_norm_beta_present) {
            std::shared_ptr<Node> instance_norm_beta_1d = nullptr;

            if (shape_size(instance_norm_beta_shape) == 1) {
                instance_norm_beta_1d =
                    std::make_shared<op::v1::Reshape>(instance_norm_beta_const, shape_1d_const, true);
            } else {
                instance_norm_beta_1d = std::make_shared<op::v0::Squeeze>(instance_norm_beta_const);
            }

            instance_norm_beta_1d =
                std::make_shared<op::v8::Gather>(instance_norm_beta_1d, gather_indices_const, gather_axis_const);

            auto group_norm_beta_corr_multiply =
                std::make_shared<ov::op::v1::Multiply>(group_norm_gamma_1d, instance_norm_beta_1d);
            group_norm_beta_1d = std::make_shared<ov::op::v1::Add>(group_norm_beta_corr_multiply, group_norm_beta_1d);
        }

        if (instance_norm_gamma_present) {
            std::shared_ptr<Node> instance_norm_gamma_1d = nullptr;

            if (shape_size(instance_norm_gamma_shape) == 1) {
                instance_norm_gamma_1d =
                    std::make_shared<op::v1::Reshape>(instance_norm_gamma_const, shape_1d_const, true);
            } else {
                instance_norm_gamma_1d = std::make_shared<op::v0::Squeeze>(instance_norm_gamma_const);
            }

            instance_norm_gamma_1d =
                std::make_shared<op::v8::Gather>(instance_norm_gamma_1d, gather_indices_const, gather_axis_const);
            group_norm_gamma_1d = std::make_shared<ov::op::v1::Multiply>(group_norm_gamma_1d, instance_norm_gamma_1d);
        }

        auto group_norm = std::make_shared<op::v12::GroupNormalization>(input,
                                                                        group_norm_gamma_1d,
                                                                        group_norm_beta_1d,
                                                                        num_groups,
                                                                        epsilon);

        return std::make_shared<Model>(OutputVector{group_norm}, ParameterVector{input});
    }
};

TEST_P(GroupNormalizationFusion4DTestsF, GroupNormalizationFusion4DTests) {
    GroupNormalizationFusion4DTestsF::run();
}

std::vector<GroupNormalizationFusionTestBaseValues> valid_vals_4d = {
    std::make_tuple(PartialShape{1, 320}, Shape{}, Shape{}, Shape{320}, Shape{320}, 1, 1e-5),
    std::make_tuple(PartialShape{1, 320}, Shape{1, 32, 1, 1}, Shape{1, 32, 1, 1}, Shape{320}, Shape{320}, 32, 1e-5),
    std::make_tuple(PartialShape{1, 64, 8, 8},
                    Shape{1, 8, 1, 1},
                    Shape{1, 8, 1, 1},
                    Shape{64, 1, 1},
                    Shape{64, 1, 1},
                    8,
                    1e-5),
    std::make_tuple(PartialShape{2, 128, 16, 16}, Shape{}, Shape{}, Shape{128, 1, 1}, Shape{128, 1, 1}, 32, 1e-5),
    std::make_tuple(PartialShape{1, 320, 64, 64}, Shape{}, Shape{}, Shape{320, 1, 1}, Shape{1, 320, 1, 1}, 32, 1e-5),
    std::make_tuple(PartialShape{1, 512, 32, 32}, Shape{}, Shape{}, Shape{512, 1, 1}, Shape{1, 512, 1, 1}, 32, 1e-6),
    std::make_tuple(PartialShape{4, 512, 64, 64},
                    Shape{1, 32, 1, 1},
                    Shape{1, 32, 1, 1},
                    Shape{512, 1, 1},
                    Shape{512, 1, 1},
                    32,
                    1e-6),
};

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusion4DPositiveTests_f32,
                         GroupNormalizationFusion4DTestsF,
                         ValuesIn(expand_vals(valid_vals_4d,
                                              GroupNormalizationFusionTransformationTestAdditionalValues(element::f32,
                                                                                                         true))),
                         GroupNormalizationFusion4DTestsF::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusion4DPositiveTests_f16,
                         GroupNormalizationFusion4DTestsF,
                         ValuesIn(expand_vals(valid_vals_4d,
                                              GroupNormalizationFusionTransformationTestAdditionalValues(element::f16,
                                                                                                         true))),
                         GroupNormalizationFusion4DTestsF::getTestCaseName);

// 4D InstanceNorm pattern with concrete shape values (vs special markers like {0, G, -1, 1})
// Some frameworks resolve shapes during optimization and emit concrete dimension values.
class GroupNormalizationFusion4DConcreteValuesTestsF
    : public GroupNormalizationFusionTestBase,
      public TransformationTestsF,
      public testing::WithParamInterface<GroupNormalizationFusionTransformationTestValues> {
public:
    static std::string getTestCaseName(
        const testing::TestParamInfo<GroupNormalizationFusionTransformationTestValues>& obj) {
        const auto& params = obj.param;

        const auto& data_shape = std::get<0>(params);
        const auto& instance_norm_gamma_shape = std::get<1>(params);
        const auto& instance_norm_beta_shape = std::get<2>(params);
        const auto& group_norm_gamma_shape = std::get<3>(params);
        const auto& group_norm_beta_shape = std::get<4>(params);
        const auto& num_groups = std::get<5>(params);
        const auto& epsilon = std::get<6>(params);
        const auto& elem_type = std::get<7>(params);
        const auto& positive_test = std::get<8>(params);

        std::ostringstream results;

        results << "T=" << elem_type << "_";
        results << "Input=" << utils::partialShape2str({data_shape}) << "_";
        results << "InstNormGamma=" << utils::partialShape2str({instance_norm_gamma_shape}) << "_";
        results << "InstNormBeta=" << utils::partialShape2str({instance_norm_beta_shape}) << "_";
        results << "GroupNormGamma=" << utils::partialShape2str({group_norm_gamma_shape}) << "_";
        results << "GroupNormBeta=" << utils::partialShape2str({group_norm_beta_shape}) << "_";
        results << "NumGroups=" << num_groups << "_";
        results << "Epsilon=" << epsilon << "_";
        results << "PositiveTest=" << std::boolalpha << positive_test;

        return results.str();
    }

    void run() {
        read_test_parameters();
        generate_weights_init_values();
        model = create_model_4d_concrete();
        manager.register_pass<pass::GroupNormalizationFusion>();

        if (positive_test) {
            model_ref = create_ref_model();
        } else {
            ASSERT_EQ(count_ops_of_type<op::v12::GroupNormalization>(model), 0);
            test_skipped = true;
        }
    }

protected:
    bool positive_test;

    void read_test_parameters() override {
        const auto& params = GetParam();

        data_shape = std::get<0>(params);
        if (!data_shape.rank().is_static())
            throw std::runtime_error("Rank of input tensor has to be static!");
        if (data_shape.rank().get_max_length() < 2)
            throw std::runtime_error("Expected at least two dimensions in input tensor!");
        if (!data_shape[1].is_static())
            throw std::runtime_error("Channel dimension in input tensor has to be static!");
        num_channels = data_shape[1].get_max_length();
        instance_norm_gamma_shape = std::get<1>(params);
        instance_norm_beta_shape = std::get<2>(params);
        group_norm_gamma_shape = std::get<3>(params);
        group_norm_beta_shape = std::get<4>(params);
        num_groups = std::get<5>(params);
        if (num_groups < 1)
            throw std::runtime_error("Number of groups has to be positive!");
        epsilon = std::get<6>(params);
        elem_type = std::get<7>(params);
        positive_test = std::get<8>(params);

        instance_norm_gamma_present = (instance_norm_gamma_shape != Shape{});
        instance_norm_beta_present = (instance_norm_beta_shape != Shape{});

        if (positive_test) {
            if ((instance_norm_gamma_shape != Shape{}) &&
                (shape_size(instance_norm_gamma_shape) != static_cast<size_t>(num_groups)))
                throw std::runtime_error("Shape of instance norm gamma has to either be empty or contain "
                                         "exactly <num_groups> elements");
            if ((instance_norm_beta_shape != Shape{}) &&
                (shape_size(instance_norm_beta_shape) != static_cast<size_t>(num_groups)))
                throw std::runtime_error("Shape of instance norm beta has to either be empty shape or contain "
                                         "exactly <num_groups> elements");
            if (shape_size(group_norm_gamma_shape) != static_cast<size_t>(num_channels))
                throw std::runtime_error("Shape of group norm gamma has to contain exactly <num_channels> elements");
            if (shape_size(group_norm_beta_shape) != static_cast<size_t>(num_channels))
                throw std::runtime_error("Shape of group norm beta has to contain exactly <num_channels> elements");

            instance_norm_gamma_present = instance_norm_gamma_present &&
                                          (shape_size(instance_norm_gamma_shape) == static_cast<size_t>(num_groups));
            instance_norm_beta_present =
                instance_norm_beta_present && (shape_size(instance_norm_beta_shape) == static_cast<size_t>(num_groups));
        }
    }

    // Creates the 4D InstanceNorm pattern model with concrete shape values
    std::shared_ptr<Model> create_model_4d_concrete() {
        if (!data_shape.is_static())
            throw std::runtime_error("Data shape must be static for concrete values test!");

        auto static_shape = data_shape.to_shape();
        auto input = std::make_shared<op::v0::Parameter>(elem_type, static_shape);

        // Compute concrete shape values
        int64_t batch = static_cast<int64_t>(static_shape[0]);
        int64_t merged_spatial = 1;
        for (size_t i = 1; i < static_shape.size(); i++) {
            if (i == 1)
                merged_spatial = static_cast<int64_t>(static_shape[i]) / num_groups;
            else
                merged_spatial *= static_cast<int64_t>(static_shape[i]);
        }

        // Reshape directly to 4D with concrete values: {batch, num_groups, merged_spatial, 1}
        auto pre_mvn_shape_const_4d =
            op::v0::Constant::create<long long>(element::i64, Shape{4}, {batch, num_groups, merged_spatial, 1});
        auto pre_mvn_reshape_4d = std::make_shared<op::v1::Reshape>(input, pre_mvn_shape_const_4d, false);

        // MVN with axes={2,3}
        auto mvn_axes_const = op::v0::Constant::create<long long>(element::i64, Shape{2}, {2, 3});
        auto mvn = std::make_shared<op::v6::MVN>(pre_mvn_reshape_4d,
                                                 mvn_axes_const,
                                                 true,
                                                 static_cast<float>(epsilon),
                                                 op::MVNEpsMode::INSIDE_SQRT);

        std::shared_ptr<Node> opt_instance_norm_gamma_multiply = mvn;
        if (instance_norm_gamma_present) {
            opt_instance_norm_gamma_multiply = std::make_shared<op::v1::Multiply>(mvn, instance_norm_gamma_const);
        }

        std::shared_ptr<Node> opt_instance_norm_beta_add = opt_instance_norm_gamma_multiply;
        if (instance_norm_beta_present) {
            opt_instance_norm_beta_add =
                std::make_shared<op::v1::Add>(opt_instance_norm_gamma_multiply, instance_norm_beta_const);
        }

        // Reshape back to original shape with concrete values
        std::vector<long long> orig_shape_vals;
        for (auto dim : static_shape)
            orig_shape_vals.push_back(static_cast<long long>(dim));
        auto original_shape_const =
            op::v0::Constant::create<long long>(element::i64, Shape{static_shape.size()}, orig_shape_vals);
        auto post_instance_norm_reshape =
            std::make_shared<op::v1::Reshape>(opt_instance_norm_beta_add, original_shape_const, false);

        auto group_norm_gamma_multiply =
            std::make_shared<op::v1::Multiply>(post_instance_norm_reshape, group_norm_gamma_const);
        auto group_norm_beta_add = std::make_shared<op::v1::Add>(group_norm_gamma_multiply, group_norm_beta_const);

        return std::make_shared<Model>(OutputVector{group_norm_beta_add}, ParameterVector{input});
    }

    std::shared_ptr<Model> create_ref_model() {
        auto input = std::make_shared<op::v0::Parameter>(elem_type, data_shape);

        auto shape_1d_const = op::v0::Constant::create(element::i64, Shape{1}, {1});
        auto gather_axis_const = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto gather_indices_vals = std::vector<int64_t>();
        for (auto i = 0ll; i < num_groups; i++)
            gather_indices_vals.insert(gather_indices_vals.end(), num_channels / num_groups, i);
        auto gather_indices_const =
            op::v0::Constant::create(element::i64, Shape{static_cast<size_t>(num_channels)}, gather_indices_vals);

        std::shared_ptr<Node> group_norm_gamma_1d = std::make_shared<op::v0::Squeeze>(group_norm_gamma_const);
        std::shared_ptr<Node> group_norm_beta_1d = std::make_shared<op::v0::Squeeze>(group_norm_beta_const);

        if (instance_norm_beta_present) {
            std::shared_ptr<Node> instance_norm_beta_1d = nullptr;

            if (shape_size(instance_norm_beta_shape) == 1) {
                instance_norm_beta_1d =
                    std::make_shared<op::v1::Reshape>(instance_norm_beta_const, shape_1d_const, true);
            } else {
                instance_norm_beta_1d = std::make_shared<op::v0::Squeeze>(instance_norm_beta_const);
            }

            instance_norm_beta_1d =
                std::make_shared<op::v8::Gather>(instance_norm_beta_1d, gather_indices_const, gather_axis_const);

            auto group_norm_beta_corr_multiply =
                std::make_shared<ov::op::v1::Multiply>(group_norm_gamma_1d, instance_norm_beta_1d);
            group_norm_beta_1d = std::make_shared<ov::op::v1::Add>(group_norm_beta_corr_multiply, group_norm_beta_1d);
        }

        if (instance_norm_gamma_present) {
            std::shared_ptr<Node> instance_norm_gamma_1d = nullptr;

            if (shape_size(instance_norm_gamma_shape) == 1) {
                instance_norm_gamma_1d =
                    std::make_shared<op::v1::Reshape>(instance_norm_gamma_const, shape_1d_const, true);
            } else {
                instance_norm_gamma_1d = std::make_shared<op::v0::Squeeze>(instance_norm_gamma_const);
            }

            instance_norm_gamma_1d =
                std::make_shared<op::v8::Gather>(instance_norm_gamma_1d, gather_indices_const, gather_axis_const);
            group_norm_gamma_1d = std::make_shared<ov::op::v1::Multiply>(group_norm_gamma_1d, instance_norm_gamma_1d);
        }

        auto group_norm = std::make_shared<op::v12::GroupNormalization>(input,
                                                                        group_norm_gamma_1d,
                                                                        group_norm_beta_1d,
                                                                        num_groups,
                                                                        epsilon);

        return std::make_shared<Model>(OutputVector{group_norm}, ParameterVector{input});
    }
};

TEST_P(GroupNormalizationFusion4DConcreteValuesTestsF, GroupNormalizationFusion4DConcreteValuesTests) {
    GroupNormalizationFusion4DConcreteValuesTestsF::run();
}

std::vector<GroupNormalizationFusionTestBaseValues> valid_vals_4d_concrete = {
    std::make_tuple(PartialShape{2, 64, 8, 8}, Shape{}, Shape{}, Shape{64, 1, 1}, Shape{64, 1, 1}, 8, 1e-5),
    std::make_tuple(PartialShape{1, 320, 64, 64}, Shape{}, Shape{}, Shape{320, 1, 1}, Shape{1, 320, 1, 1}, 32, 1e-5),
    std::make_tuple(PartialShape{4, 512, 64, 64}, Shape{}, Shape{}, Shape{512, 1, 1}, Shape{1, 512, 1, 1}, 32, 1e-6),
};

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusion4DConcreteValuesPositiveTests_f32,
                         GroupNormalizationFusion4DConcreteValuesTestsF,
                         ValuesIn(expand_vals(valid_vals_4d_concrete,
                                              GroupNormalizationFusionTransformationTestAdditionalValues(element::f32,
                                                                                                         true))),
                         GroupNormalizationFusion4DConcreteValuesTestsF::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusion4DConcreteValuesPositiveTests_f16,
                         GroupNormalizationFusion4DConcreteValuesTestsF,
                         ValuesIn(expand_vals(valid_vals_4d_concrete,
                                              GroupNormalizationFusionTransformationTestAdditionalValues(element::f16,
                                                                                                         true))),
                         GroupNormalizationFusion4DConcreteValuesTestsF::getTestCaseName);

// Standalone negative tests for 4D-specific edge cases that require custom model construction.
// When model_ref is not set, TransformationTestsF::TearDown clones the model, runs the pass,
// and verifies the model is unchanged — confirming the fusion does NOT fire.
class GroupNormalizationFusion4DNegativeEdgeCasesF : public TransformationTestsF {};

// Helper: builds a 4D InstanceNorm-style pattern with customizable MVN axes and 4D trailing dimension.
// Returns a Model whose output is the final Add node (beta_add).
static std::shared_ptr<Model> build_4d_pattern_model(const PartialShape& data_shape,
                                                     int64_t num_groups,
                                                     const std::vector<long long>& mvn_axes_vals,
                                                     long long trailing_dim,
                                                     float epsilon = 1e-5f,
                                                     element::Type elem_type = element::f32) {
    const auto num_channels = data_shape[1].get_max_length();

    auto input = std::make_shared<op::v0::Parameter>(elem_type, data_shape);

    // Reshape directly to 4D: {0, G, -1, trailing_dim}
    auto pre_mvn_shape_4d =
        op::v0::Constant::create<long long>(element::i64, Shape{4}, {0, num_groups, -1, trailing_dim});
    auto reshape_4d = std::make_shared<v1::Reshape>(input, pre_mvn_shape_4d, true);

    // MVN with custom axes
    auto mvn_axes = op::v0::Constant::create<long long>(element::i64, Shape{mvn_axes_vals.size()}, mvn_axes_vals);
    auto mvn = std::make_shared<op::v6::MVN>(reshape_4d, mvn_axes, true, epsilon, op::MVNEpsMode::INSIDE_SQRT);

    // Reshape back to original shape
    auto post_shape = std::make_shared<op::v0::ShapeOf>(input);
    auto post_reshape = std::make_shared<v1::Reshape>(mvn, post_shape, true);

    // Group norm gamma/beta
    auto gamma_const = op::v0::Constant::create(elem_type,
                                                Shape{static_cast<size_t>(num_channels), 1, 1},
                                                std::vector<float>(num_channels, 1.0f));
    auto gamma_mul = std::make_shared<op::v1::Multiply>(post_reshape, gamma_const);
    auto beta_const = op::v0::Constant::create(elem_type,
                                               Shape{1, static_cast<size_t>(num_channels), 1, 1},
                                               std::vector<float>(num_channels, 0.0f));
    auto beta_add = std::make_shared<op::v1::Add>(gamma_mul, beta_const);

    return std::make_shared<Model>(OutputVector{beta_add}, ParameterVector{input});
}

// 4D pattern with wrong MVN axes: {1, 2} instead of {2, 3}
TEST_F(GroupNormalizationFusion4DNegativeEdgeCasesF, WrongMVNAxes) {
    model = build_4d_pattern_model(PartialShape{1, 320, 64, 64},
                                   /*num_groups=*/32,
                                   /*mvn_axes_vals=*/{1, 2},
                                   /*trailing_dim=*/1);
    manager.register_pass<pass::GroupNormalizationFusion>();
}

// 4D pattern with single MVN axis {2} — requires {2, 3} for 4D pattern
TEST_F(GroupNormalizationFusion4DNegativeEdgeCasesF, SingleMVNAxisWith4DReshape) {
    model = build_4d_pattern_model(PartialShape{1, 320, 64, 64},
                                   /*num_groups=*/32,
                                   /*mvn_axes_vals=*/{2},
                                   /*trailing_dim=*/1);
    manager.register_pass<pass::GroupNormalizationFusion>();
}

// 4D pattern with trailing dimension != 1 (e.g., 2)
TEST_F(GroupNormalizationFusion4DNegativeEdgeCasesF, TrailingDimNotOne) {
    model = build_4d_pattern_model(PartialShape{1, 320, 64, 64},
                                   /*num_groups=*/32,
                                   /*mvn_axes_vals=*/{2, 3},
                                   /*trailing_dim=*/2);
    manager.register_pass<pass::GroupNormalizationFusion>();
}

// 4D pattern with MVN axes {0, 1} — normalizing over wrong dimensions
TEST_F(GroupNormalizationFusion4DNegativeEdgeCasesF, MVNAxesOverBatchAndGroups) {
    model = build_4d_pattern_model(PartialShape{1, 320, 64, 64},
                                   /*num_groups=*/32,
                                   /*mvn_axes_vals=*/{0, 1},
                                   /*trailing_dim=*/1);
    manager.register_pass<pass::GroupNormalizationFusion>();
}

// 4D pattern with three MVN axes {1, 2, 3} — too many axes for 4D pattern
TEST_F(GroupNormalizationFusion4DNegativeEdgeCasesF, ThreeMVNAxes) {
    model = build_4d_pattern_model(PartialShape{1, 320, 64, 64},
                                   /*num_groups=*/32,
                                   /*mvn_axes_vals=*/{1, 2, 3},
                                   /*trailing_dim=*/1);
    manager.register_pass<pass::GroupNormalizationFusion>();
}

}  // namespace test
}  // namespace ov
