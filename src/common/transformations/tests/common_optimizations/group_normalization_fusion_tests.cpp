// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "shared_test_classes/subgraph/group_normalization_fusion.hpp"
#include "transformations/common_optimizations/group_normalization_fusion.hpp"

using namespace testing;

namespace ov {
namespace test {

void core_configuration(SubgraphBaseTest* test) {}

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

        return std::make_shared<Model>(NodeVector{group_norm}, ParameterVector{input});
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

}  // namespace test
}  // namespace ov