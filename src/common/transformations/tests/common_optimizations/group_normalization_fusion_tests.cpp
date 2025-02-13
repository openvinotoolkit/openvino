// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "shared_test_classes/subgraph/group_normalization_fusion.hpp"
#include "transformations/common_optimizations/group_normalization_fusion.hpp"

using namespace testing;
using namespace ov;

using GroupNormalizationFusionSubgraphTestValues =
    std::tuple<PartialShape,        // (partial) shape of input/output tensor (all dims except channel can be dynamic)
               Shape,               // shape of optional instance norm gamma tensor (or empty shape if not used)
               Shape,               // shape of optional instance norm beta tensor (or empty shape if not used)
               Shape,               // shape of group norm gamma tensor
               Shape,               // shape of group norm beta tensor
               unsigned long long,  // number of groups
               float,               // epsilon
               bool>;               // whether it's a positive test that should run reference model or a negative test

template <element::Type_t T_elem_t>
class GroupNormalizationFusionTransformationTestsF
    : public ov::test::GroupNormalizationFusionTestBase<T_elem_t>,
      public testing::TestWithParam<GroupNormalizationFusionSubgraphTestValues> {
public:
    static constexpr element::Type T_elem = T_elem_t;
    static std::string getTestCaseName(const testing::TestParamInfo<GroupNormalizationFusionSubgraphTestValues>& obj) {
        const auto& params = obj.param;

        const auto& data_shape = std::get<0>(params);
        const auto& instance_norm_gamma_shape = std::get<1>(params);
        const auto& instance_norm_beta_shape = std::get<2>(params);
        const auto& group_norm_gamma_shape = std::get<3>(params);
        const auto& group_norm_beta_shape = std::get<4>(params);
        const auto& num_groups = std::get<5>(params);
        const auto& epsilon = std::get<6>(params);
        const auto& positive_test = std::get<7>(params);

        std::ostringstream results;

        results << "T=" << T_elem_t << "_";
        results << "Input=" << ov::test::utils::partialShape2str({data_shape}) << "_";
        results << "InstNormGamma=" << ov::test::utils::partialShape2str({instance_norm_gamma_shape}) << "_";
        results << "InstNormBeta=" << ov::test::utils::partialShape2str({instance_norm_beta_shape}) << "_";
        results << "GroupNormGamma=" << ov::test::utils::partialShape2str({group_norm_gamma_shape}) << "_";
        results << "GroupNormBeta=" << ov::test::utils::partialShape2str({group_norm_beta_shape}) << "_";
        results << "NumGroups=" << num_groups << "_";
        results << "Epsilon=" << epsilon << "_";
        results << "PositiveTest=" << std::boolalpha << positive_test << "_";

        return results.str();
    }

    void run() {
        read_test_parameters();
        this->generate_weights_init_values();
        model = this->create_model();

        manager = ov::pass::Manager();
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::GroupNormalizationFusion>();
        OV_ASSERT_NO_THROW(manager.run_passes(model));

        if (positiveTest) {
            model_ref = create_ref_model();

            manager_ref = ov::pass::Manager();
            manager_ref.register_pass<ov::pass::InitNodeInfo>();
            OV_ASSERT_NO_THROW(manager_ref.run_passes(model_ref));

            const auto& f_parameters = model->get_parameters();
            const auto& f_ref_parameters = model_ref->get_parameters();
            ASSERT_EQ(f_parameters.size(), f_ref_parameters.size());
            ASSERT_EQ(f_parameters.size(), 1);
            ASSERT_EQ(f_parameters[0]->outputs().size(), f_ref_parameters[0]->outputs().size());
            ASSERT_EQ(f_parameters[0]->outputs().size(), 1);
            ASSERT_EQ(f_parameters[0]->get_element_type(), f_ref_parameters[0]->get_element_type());
            ASSERT_EQ(f_parameters[0]->get_element_type(), T_elem);

            const auto& f_results = model->get_results();
            const auto& f_ref_results = model_ref->get_results();
            ASSERT_EQ(f_results.size(), f_ref_results.size());
            ASSERT_EQ(f_results.size(), 1);
            ASSERT_EQ(f_results[0]->outputs().size(), f_ref_results[0]->outputs().size());
            ASSERT_EQ(f_results[0]->outputs().size(), 1);
            ASSERT_EQ(f_results[0]->inputs().size(), f_ref_results[0]->inputs().size());
            ASSERT_EQ(f_results[0]->inputs().size(), 1);
            ASSERT_EQ(f_results[0]->get_element_type(), f_ref_results[0]->get_element_type());
            ASSERT_EQ(f_results[0]->get_element_type(), T_elem);
            ASSERT_EQ(f_results[0]->get_output_partial_shape(0), f_ref_results[0]->get_output_partial_shape(0));
            ASSERT_EQ(f_results[0]->get_output_partial_shape(0), f_parameters[0]->get_output_partial_shape(0));
            ASSERT_EQ(f_ref_results[0]->get_output_partial_shape(0), f_ref_parameters[0]->get_output_partial_shape(0));
            ASSERT_EQ(f_ref_results[0]->get_output_partial_shape(0), this->dataShape);

            const auto& gn_node = f_results[0]->get_input_node_shared_ptr(0);
            const auto& gn_ref_node = f_ref_results[0]->get_input_node_shared_ptr(0);
            ASSERT_TRUE(ov::is_type<ov::op::v12::GroupNormalization>(gn_node));
            ASSERT_TRUE(ov::is_type<ov::op::v12::GroupNormalization>(gn_ref_node));
            ASSERT_EQ(gn_node->inputs().size(), gn_ref_node->inputs().size());
            ASSERT_EQ(gn_node->inputs().size(), 3);
            ASSERT_EQ(gn_node->get_input_partial_shape(0), gn_ref_node->get_input_partial_shape(0));
            ASSERT_EQ(gn_node->get_input_partial_shape(0), this->dataShape);
            ASSERT_EQ(shape_size(gn_node->get_input_shape(1)), shape_size(gn_ref_node->get_input_shape(1)));
            ASSERT_EQ(shape_size(gn_node->get_input_shape(1)), this->numChannels);
            ASSERT_EQ(shape_size(gn_node->get_input_shape(2)), shape_size(gn_ref_node->get_input_shape(2)));
            ASSERT_EQ(shape_size(gn_node->get_input_shape(2)), this->numChannels);

            const auto& gn_node_casted = ov::as_type_ptr<ov::op::v12::GroupNormalization>(gn_node);
            const auto& gn_ref_node_casted = ov::as_type_ptr<ov::op::v12::GroupNormalization>(gn_ref_node);
            ASSERT_EQ(gn_node_casted->get_epsilon(), gn_ref_node_casted->get_epsilon());
            ASSERT_EQ(gn_node_casted->get_epsilon(), this->epsilon);
            ASSERT_EQ(gn_node_casted->get_num_groups(), gn_ref_node_casted->get_num_groups());
            ASSERT_EQ(gn_node_casted->get_num_groups(), this->numGroups);
        } else {
            ASSERT_EQ(count_ops_of_type<ov::op::v12::GroupNormalization>(model), 0);
        }
    }

protected:
    bool positiveTest;
    ov::pass::Manager manager;
    ov::pass::Manager manager_ref;
    std::shared_ptr<ov::Model> model;
    std::shared_ptr<ov::Model> model_ref;

    void read_test_parameters() override {
        const auto& params = GetParam();

        this->dataShape = std::get<0>(params);
        if (!this->dataShape.rank().is_static())
            throw std::runtime_error("Rank of input tensor has to be static!");
        if (this->dataShape.rank().get_max_length() < 2)
            throw std::runtime_error("Expected at least two dimensions in input tensor!");
        if (!this->dataShape[1].is_static())
            throw std::runtime_error("Channel dimension in input tensor has to be static!");

        this->numChannels = static_cast<size_t>(this->dataShape[1].get_max_length());
        this->instanceNormGammaShape = std::get<1>(params);
        this->instanceNormBetaShape = std::get<2>(params);
        this->groupNormGammaShape = std::get<3>(params);
        this->groupNormBetaShape = std::get<4>(params);
        this->numGroups = std::get<5>(params);
        this->epsilon = std::get<6>(params);
        positiveTest = std::get<7>(params);

        this->instanceNormGammaPresent = (this->instanceNormGammaShape != Shape{});
        this->instanceNormBetaPresent = (this->instanceNormBetaShape != Shape{});

        if (positiveTest) {
            if ((this->instanceNormGammaShape != Shape{}) &&
                (shape_size(this->instanceNormGammaShape) != this->numGroups))
                throw std::runtime_error("Shape of instance norm gamma has to either be empty or contain "
                                         "exactly <numGroups> elements");
            if ((this->instanceNormBetaShape != Shape{}) &&
                (shape_size(this->instanceNormBetaShape) != this->numGroups))
                throw std::runtime_error("Shape of instance norm beta has to either be empty shape or contain "
                                         "exactly <numGroups> elements");
            if (shape_size(this->groupNormGammaShape) != this->numChannels)
                throw std::runtime_error("Shape of group norm gamma has to contain exactly <numChannels> elements");
            if (shape_size(this->groupNormBetaShape) != this->numChannels)
                throw std::runtime_error("Shape of group norm beta has to contain exactly <numChannels> elements");

            this->instanceNormGammaPresent =
                this->instanceNormGammaPresent && (shape_size(this->instanceNormGammaShape) == this->numGroups);
            this->instanceNormBetaPresent =
                this->instanceNormBetaPresent && (shape_size(this->instanceNormBetaShape) == this->numGroups);
        }
    }

    std::shared_ptr<ov::Model> create_ref_model() {
        auto input = std::make_shared<ov::op::v0::Parameter>(T_elem, this->dataShape);

        auto group_norm_beta_corr_vals = this->groupNormBetaVals;
        if (this->instanceNormBetaPresent)
            for (auto i = 0ull; i < group_norm_beta_corr_vals.size(); i++)
                group_norm_beta_corr_vals[i] =
                    this->groupNormGammaVals[i] *
                        this->instanceNormBetaVals[i / (this->numChannels / this->numGroups)] +
                    this->groupNormBetaVals[i];
        auto group_norm_beta_1d = op::v0::Constant::create(T_elem, Shape{this->numChannels}, group_norm_beta_corr_vals);

        auto group_norm_gamma_corr_vals = this->groupNormGammaVals;
        if (this->instanceNormGammaPresent)
            for (auto i = 0ull; i < group_norm_gamma_corr_vals.size(); i++)
                group_norm_gamma_corr_vals[i] = this->groupNormGammaVals[i] *
                                                this->instanceNormGammaVals[i / (this->numChannels / this->numGroups)];
        auto group_norm_gamma_1d =
            op::v0::Constant::create(T_elem, Shape{this->numChannels}, group_norm_gamma_corr_vals);

        auto group_norm = std::make_shared<ov::op::v12::GroupNormalization>(input,
                                                                            group_norm_gamma_1d,
                                                                            group_norm_beta_1d,
                                                                            this->numGroups,
                                                                            this->epsilon);

        return std::make_shared<Model>(NodeVector{group_norm}, ParameterVector{input});
    }
};

class GroupNormalizationFusionTransformationTestsF_f32
    : public GroupNormalizationFusionTransformationTestsF<element::Type_t::f32> {};
class GroupNormalizationFusionTransformationTestsF_f16
    : public GroupNormalizationFusionTransformationTestsF<element::Type_t::f16> {};
class GroupNormalizationFusionTransformationTestsF_bf16
    : public GroupNormalizationFusionTransformationTestsF<element::Type_t::bf16> {};
class GroupNormalizationFusionTransformationTestsF_u8
    : public GroupNormalizationFusionTransformationTestsF<element::Type_t::u8> {};
class GroupNormalizationFusionTransformationTestsF_u16
    : public GroupNormalizationFusionTransformationTestsF<element::Type_t::u16> {};
class GroupNormalizationFusionTransformationTestsF_u32
    : public GroupNormalizationFusionTransformationTestsF<element::Type_t::u32> {};
class GroupNormalizationFusionTransformationTestsF_u64
    : public GroupNormalizationFusionTransformationTestsF<element::Type_t::u64> {};
class GroupNormalizationFusionTransformationTestsF_i8
    : public GroupNormalizationFusionTransformationTestsF<element::Type_t::i8> {};
class GroupNormalizationFusionTransformationTestsF_i16
    : public GroupNormalizationFusionTransformationTestsF<element::Type_t::i16> {};
class GroupNormalizationFusionTransformationTestsF_i32
    : public GroupNormalizationFusionTransformationTestsF<element::Type_t::i32> {};
class GroupNormalizationFusionTransformationTestsF_i64
    : public GroupNormalizationFusionTransformationTestsF<element::Type_t::i64> {};
class GroupNormalizationFusionTransformationTestsF_f8e4m3
    : public GroupNormalizationFusionTransformationTestsF<element::Type_t::f8e4m3> {};
class GroupNormalizationFusionTransformationTestsF_f8e5m2
    : public GroupNormalizationFusionTransformationTestsF<element::Type_t::f8e5m2> {};
class GroupNormalizationFusionTransformationTestsF_f4e2m1
    : public GroupNormalizationFusionTransformationTestsF<element::Type_t::f4e2m1> {};
class GroupNormalizationFusionTransformationTestsF_f8e8m0
    : public GroupNormalizationFusionTransformationTestsF<element::Type_t::f8e8m0> {};

TEST_P(GroupNormalizationFusionTransformationTestsF_f32, GroupNormalizationFusionTransformationTests_f32) {
    GroupNormalizationFusionTransformationTestsF_f32::run();
}

TEST_P(GroupNormalizationFusionTransformationTestsF_f16, GroupNormalizationFusionTransformationTests_f16) {
    GroupNormalizationFusionTransformationTestsF_f16::run();
}

TEST_P(GroupNormalizationFusionTransformationTestsF_bf16, GroupNormalizationFusionTransformationTests_bf16) {
    GroupNormalizationFusionTransformationTestsF_bf16::run();
}

TEST_P(GroupNormalizationFusionTransformationTestsF_u8, GroupNormalizationFusionTransformationTests_u8) {
    GroupNormalizationFusionTransformationTestsF_u8::run();
}

TEST_P(GroupNormalizationFusionTransformationTestsF_u16, GroupNormalizationFusionTransformationTests_u16) {
    GroupNormalizationFusionTransformationTestsF_u16::run();
}

TEST_P(GroupNormalizationFusionTransformationTestsF_u32, GroupNormalizationFusionTransformationTests_u32) {
    GroupNormalizationFusionTransformationTestsF_u32::run();
}

TEST_P(GroupNormalizationFusionTransformationTestsF_u64, GroupNormalizationFusionTransformationTests_u64) {
    GroupNormalizationFusionTransformationTestsF_u64::run();
}

TEST_P(GroupNormalizationFusionTransformationTestsF_i8, GroupNormalizationFusionTransformationTests_i8) {
    GroupNormalizationFusionTransformationTestsF_i8::run();
}

TEST_P(GroupNormalizationFusionTransformationTestsF_i16, GroupNormalizationFusionTransformationTests_i16) {
    GroupNormalizationFusionTransformationTestsF_i16::run();
}

TEST_P(GroupNormalizationFusionTransformationTestsF_i32, GroupNormalizationFusionTransformationTests_i32) {
    GroupNormalizationFusionTransformationTestsF_i32::run();
}

TEST_P(GroupNormalizationFusionTransformationTestsF_i64, GroupNormalizationFusionTransformationTests_i64) {
    GroupNormalizationFusionTransformationTestsF_i64::run();
}

TEST_P(GroupNormalizationFusionTransformationTestsF_f8e4m3, GroupNormalizationFusionTransformationTests_f8e4m3) {
    GroupNormalizationFusionTransformationTestsF_f8e4m3::run();
}

TEST_P(GroupNormalizationFusionTransformationTestsF_f8e5m2, GroupNormalizationFusionTransformationTests_f8e5m2) {
    GroupNormalizationFusionTransformationTestsF_f8e5m2::run();
}

TEST_P(GroupNormalizationFusionTransformationTestsF_f4e2m1, GroupNormalizationFusionTransformationTests_f4e2m1) {
    GroupNormalizationFusionTransformationTestsF_f4e2m1::run();
}

TEST_P(GroupNormalizationFusionTransformationTestsF_f8e8m0, GroupNormalizationFusionTransformationTests_f8e8m0) {
    GroupNormalizationFusionTransformationTestsF_f8e8m0::run();
}

using GroupNormalizationFusionSubgraphTestAdditionalValues =
    std::tuple<bool>;  // whether it's a positive test that should run reference model or a negative test

std::vector<ov::test::GroupNormalizationFusionTestBaseValues> valid_vals = {
    std::make_tuple(PartialShape{1, 320}, Shape{}, Shape{}, Shape{320}, Shape{320}, 1, 1e-5f),
    std::make_tuple(PartialShape{1, 320, 2, 2},
                    Shape{1, 1, 1},
                    Shape{1, 1, 1},
                    Shape{320, 1, 1},
                    Shape{1, 320, 1, 1},
                    1,
                    1e-5f),
    std::make_tuple(PartialShape{5, 320, 2, 2, 2},
                    Shape{1, 320, 1},
                    Shape{1, 320, 1},
                    Shape{320, 1, 1, 1},
                    Shape{320, 1, 1, 1},
                    320,
                    1e-5f),
    std::make_tuple(
        PartialShape{Dimension::dynamic(), 320, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()},
        Shape{1, 320, 1},
        Shape{1, 320, 1},
        Shape{320, 1, 1, 1},
        Shape{320, 1, 1, 1},
        320,
        1e-5f),
    std::make_tuple(PartialShape{3, 320}, Shape{32, 1}, Shape{32, 1}, Shape{320}, Shape{320}, 32, 1e-5f),
    std::make_tuple(PartialShape{2, 9, 4, 5, 6},
                    Shape{3, 1},
                    Shape{3, 1},
                    Shape{1, 9, 1, 1, 1},
                    Shape{1, 9, 1, 1, 1},
                    3,
                    1e-5f),
    std::make_tuple(PartialShape{1, 320, 2, 4},
                    Shape{1, 32, 1},
                    Shape{1, 32, 1},
                    Shape{320, 1, 1},
                    Shape{320, 1, 1},
                    32,
                    1e-5f),
    std::make_tuple(PartialShape{8, 320, 4, 8}, Shape{}, Shape{}, Shape{320, 1, 1}, Shape{1, 320, 1, 1}, 32, 1e-5f),
    std::make_tuple(PartialShape{1, 512, 4, 8},
                    Shape{},
                    Shape{1, 128, 1},
                    Shape{1, 512, 1, 1},
                    Shape{512, 1, 1},
                    128,
                    1e-6f),
    std::make_tuple(PartialShape{1, 192, 2, 2},
                    Shape{1, 64, 1},
                    Shape{},
                    Shape{1, 192, 1, 1},
                    Shape{1, 192, 1, 1},
                    64,
                    1e-6f)};

std::vector<ov::test::GroupNormalizationFusionTestBaseValues> invalid_vals = {
    std::make_tuple(PartialShape{1, 320}, Shape{}, Shape{}, Shape{}, Shape{}, 1, 1e-5f),
    std::make_tuple(PartialShape{1, 320, 2, 2},
                    Shape{1, 1, 1},
                    Shape{1, 1, 1},
                    Shape{1, 1, 1},
                    Shape{1, 1, 1, 1},
                    1,
                    1e-5f),
    std::make_tuple(PartialShape{1, 320, 2, 2}, Shape{}, Shape{}, Shape{320, 1, 1}, Shape{}, 1, 1e-5f),
    std::make_tuple(PartialShape{1, 320, 2, 2}, Shape{}, Shape{}, Shape{}, Shape{1, 320, 1, 1}, 1, 1e-5f),
    std::make_tuple(PartialShape{1, 320, 2, 2},
                    Shape{1, 1, 1},
                    Shape{1, 32, 1},
                    Shape{320, 1, 1},
                    Shape{320, 1, 1},
                    32,
                    1e-5f),
    std::make_tuple(PartialShape{1, 320, 2, 2},
                    Shape{1, 32, 1},
                    Shape{1, 1, 1},
                    Shape{320, 1, 1},
                    Shape{320, 1, 1},
                    32,
                    1e-5f),
    std::make_tuple(PartialShape{Dimension::dynamic(), 512, Dimension::dynamic(), Dimension::dynamic()},
                    Shape{},
                    Shape{},
                    Shape{1, 512, 1, 1},
                    Shape{1, 512, 1, 1},
                    100,
                    1e-6f)};

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionTransformationPositiveTests_f32,
                         GroupNormalizationFusionTransformationTestsF_f32,
                         ValuesIn(ov::test::expand_vals(valid_vals,
                                                        GroupNormalizationFusionSubgraphTestAdditionalValues(true))),
                         GroupNormalizationFusionTransformationTestsF_f32::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionTransformationPositiveTests_f16,
                         GroupNormalizationFusionTransformationTestsF_f16,
                         ValuesIn(ov::test::expand_vals(valid_vals,
                                                        GroupNormalizationFusionSubgraphTestAdditionalValues(true))),
                         GroupNormalizationFusionTransformationTestsF_f16::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionTransformationPositiveTests_bf16,
                         GroupNormalizationFusionTransformationTestsF_bf16,
                         ValuesIn(ov::test::expand_vals(valid_vals,
                                                        GroupNormalizationFusionSubgraphTestAdditionalValues(true))),
                         GroupNormalizationFusionTransformationTestsF_bf16::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionTransformationNegativeTests_f32,
                         GroupNormalizationFusionTransformationTestsF_f32,
                         ValuesIn(ov::test::expand_vals(invalid_vals,
                                                        GroupNormalizationFusionSubgraphTestAdditionalValues(false))),
                         GroupNormalizationFusionTransformationTestsF_f32::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionTransformationNegativeTests_f16,
                         GroupNormalizationFusionTransformationTestsF_f16,
                         ValuesIn(ov::test::expand_vals(invalid_vals,
                                                        GroupNormalizationFusionSubgraphTestAdditionalValues(false))),
                         GroupNormalizationFusionTransformationTestsF_f16::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionTransformationNegativeTests_bf16,
                         GroupNormalizationFusionTransformationTestsF_bf16,
                         ValuesIn(ov::test::expand_vals(invalid_vals,
                                                        GroupNormalizationFusionSubgraphTestAdditionalValues(false))),
                         GroupNormalizationFusionTransformationTestsF_bf16::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionTransformationNegativeTestsValidVals_u8,
                         GroupNormalizationFusionTransformationTestsF_u8,
                         ValuesIn(ov::test::expand_vals(valid_vals,
                                                        GroupNormalizationFusionSubgraphTestAdditionalValues(false))),
                         GroupNormalizationFusionTransformationTestsF_u8::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionTransformationNegativeTestsValidVals_u16,
                         GroupNormalizationFusionTransformationTestsF_u16,
                         ValuesIn(ov::test::expand_vals(valid_vals,
                                                        GroupNormalizationFusionSubgraphTestAdditionalValues(false))),
                         GroupNormalizationFusionTransformationTestsF_u16::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionTransformationNegativeTestsValidVals_u32,
                         GroupNormalizationFusionTransformationTestsF_u32,
                         ValuesIn(ov::test::expand_vals(valid_vals,
                                                        GroupNormalizationFusionSubgraphTestAdditionalValues(false))),
                         GroupNormalizationFusionTransformationTestsF_u32::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionTransformationNegativeTestsValidVals_u64,
                         GroupNormalizationFusionTransformationTestsF_u64,
                         ValuesIn(ov::test::expand_vals(valid_vals,
                                                        GroupNormalizationFusionSubgraphTestAdditionalValues(false))),
                         GroupNormalizationFusionTransformationTestsF_u64::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionTransformationNegativeTestsValidVals_i8,
                         GroupNormalizationFusionTransformationTestsF_i8,
                         ValuesIn(ov::test::expand_vals(valid_vals,
                                                        GroupNormalizationFusionSubgraphTestAdditionalValues(false))),
                         GroupNormalizationFusionTransformationTestsF_i8::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionTransformationNegativeTestsValidVals_i16,
                         GroupNormalizationFusionTransformationTestsF_i16,
                         ValuesIn(ov::test::expand_vals(valid_vals,
                                                        GroupNormalizationFusionSubgraphTestAdditionalValues(false))),
                         GroupNormalizationFusionTransformationTestsF_i16::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionTransformationNegativeTestsValidVals_i32,
                         GroupNormalizationFusionTransformationTestsF_i32,
                         ValuesIn(ov::test::expand_vals(valid_vals,
                                                        GroupNormalizationFusionSubgraphTestAdditionalValues(false))),
                         GroupNormalizationFusionTransformationTestsF_i32::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionTransformationNegativeTestsValidVals_f8e5m2,
                         GroupNormalizationFusionTransformationTestsF_f8e5m2,
                         ValuesIn(ov::test::expand_vals(valid_vals,
                                                        GroupNormalizationFusionSubgraphTestAdditionalValues(false))),
                         GroupNormalizationFusionTransformationTestsF_f8e5m2::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionTransformationNegativeTestsValidVals_f4e2m1,
                         GroupNormalizationFusionTransformationTestsF_f4e2m1,
                         ValuesIn(ov::test::expand_vals(valid_vals,
                                                        GroupNormalizationFusionSubgraphTestAdditionalValues(false))),
                         GroupNormalizationFusionTransformationTestsF_f4e2m1::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionTransformationNegativeTestsValidVals_f8e8m0,
                         GroupNormalizationFusionTransformationTestsF_f8e8m0,
                         ValuesIn(ov::test::expand_vals(valid_vals,
                                                        GroupNormalizationFusionSubgraphTestAdditionalValues(false))),
                         GroupNormalizationFusionTransformationTestsF_f8e8m0::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionTransformationNegativeTestsInvalidVals_u8,
                         GroupNormalizationFusionTransformationTestsF_u8,
                         ValuesIn(ov::test::expand_vals(invalid_vals,
                                                        GroupNormalizationFusionSubgraphTestAdditionalValues(false))),
                         GroupNormalizationFusionTransformationTestsF_u8::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionTransformationNegativeTestsInvalidVals_u16,
                         GroupNormalizationFusionTransformationTestsF_u16,
                         ValuesIn(ov::test::expand_vals(invalid_vals,
                                                        GroupNormalizationFusionSubgraphTestAdditionalValues(false))),
                         GroupNormalizationFusionTransformationTestsF_u16::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionTransformationNegativeTestsInvalidVals_u32,
                         GroupNormalizationFusionTransformationTestsF_u32,
                         ValuesIn(ov::test::expand_vals(invalid_vals,
                                                        GroupNormalizationFusionSubgraphTestAdditionalValues(false))),
                         GroupNormalizationFusionTransformationTestsF_u32::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionTransformationNegativeTestsInvalidVals_u64,
                         GroupNormalizationFusionTransformationTestsF_u64,
                         ValuesIn(ov::test::expand_vals(invalid_vals,
                                                        GroupNormalizationFusionSubgraphTestAdditionalValues(false))),
                         GroupNormalizationFusionTransformationTestsF_u64::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionTransformationNegativeTestsInvalidVals_i8,
                         GroupNormalizationFusionTransformationTestsF_i8,
                         ValuesIn(ov::test::expand_vals(invalid_vals,
                                                        GroupNormalizationFusionSubgraphTestAdditionalValues(false))),
                         GroupNormalizationFusionTransformationTestsF_i8::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionTransformationNegativeTestsInvalidVals_i16,
                         GroupNormalizationFusionTransformationTestsF_i16,
                         ValuesIn(ov::test::expand_vals(invalid_vals,
                                                        GroupNormalizationFusionSubgraphTestAdditionalValues(false))),
                         GroupNormalizationFusionTransformationTestsF_i16::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionTransformationNegativeTestsInvalidVals_i32,
                         GroupNormalizationFusionTransformationTestsF_i32,
                         ValuesIn(ov::test::expand_vals(invalid_vals,
                                                        GroupNormalizationFusionSubgraphTestAdditionalValues(false))),
                         GroupNormalizationFusionTransformationTestsF_i32::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionTransformationNegativeTestsInalidVals_f8e5m2,
                         GroupNormalizationFusionTransformationTestsF_f8e5m2,
                         ValuesIn(ov::test::expand_vals(invalid_vals,
                                                        GroupNormalizationFusionSubgraphTestAdditionalValues(false))),
                         GroupNormalizationFusionTransformationTestsF_f8e5m2::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionTransformationNegativeTestsInvalidVals_f4e2m1,
                         GroupNormalizationFusionTransformationTestsF_f4e2m1,
                         ValuesIn(ov::test::expand_vals(invalid_vals,
                                                        GroupNormalizationFusionSubgraphTestAdditionalValues(false))),
                         GroupNormalizationFusionTransformationTestsF_f4e2m1::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionTransformationNegativeTestsInvalidVals_f8e8m0,
                         GroupNormalizationFusionTransformationTestsF_f8e8m0,
                         ValuesIn(ov::test::expand_vals(invalid_vals,
                                                        GroupNormalizationFusionSubgraphTestAdditionalValues(false))),
                         GroupNormalizationFusionTransformationTestsF_f8e8m0::getTestCaseName);