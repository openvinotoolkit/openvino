// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/group_normalization_fusion.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "transformations/common_optimizations/group_normalization_fusion.hpp"

using namespace testing;

namespace ov {
namespace test {

void GroupNormalizationFusionTestBase::generate_weights_init_values() {
    if (instance_norm_gamma_present) {
        auto instanceNormGammaTensor =
            utils::create_and_fill_tensor(elem_type, instance_norm_gamma_shape, utils::InputGenerateData(1, 10, 1, 2));
        instance_norm_gamma_const = std::make_shared<op::v0::Constant>(instanceNormGammaTensor);
    }
    if (instance_norm_beta_present) {
        auto instanceNormBetaTensor =
            utils::create_and_fill_tensor(elem_type, instance_norm_beta_shape, utils::InputGenerateData(1, 10, 1, 3));
        instance_norm_beta_const = std::make_shared<op::v0::Constant>(instanceNormBetaTensor);
    }

    auto groupNormGammaTensor =
        utils::create_and_fill_tensor(elem_type, group_norm_gamma_shape, utils::InputGenerateData(1, 10, 1, 1));
    group_norm_gamma_const = std::make_shared<op::v0::Constant>(groupNormGammaTensor);

    auto groupNormBetaTensor =
        utils::create_and_fill_tensor(elem_type, group_norm_beta_shape, utils::InputGenerateData(1, 10, 1, 11));
    group_norm_beta_const = std::make_shared<op::v0::Constant>(groupNormBetaTensor);
}

std::shared_ptr<Model> GroupNormalizationFusionTestBase::create_model() {
    auto input = std::make_shared<op::v0::Parameter>(elem_type, data_shape);
    auto pre_mvn_shape_const = op::v0::Constant::create<long long>(element::i64, Shape{3}, {0, num_groups, -1});
    auto pre_mvn_reshape = std::make_shared<op::v1::Reshape>(input, pre_mvn_shape_const, true);

    auto mvn_axes_const = op::v0::Constant::create<long long>(element::i64, Shape{1}, {2});
    auto mvn =
        std::make_shared<op::v6::MVN>(pre_mvn_reshape, mvn_axes_const, true, epsilon, op::MVNEpsMode::INSIDE_SQRT);

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

    return std::make_shared<Model>(NodeVector{group_norm_beta_add}, ParameterVector{input});
}

std::string GroupNormalizationFusionSubgraphTestsF::getTestCaseName(
    const testing::TestParamInfo<GroupNormalizationFusionSubgraphTestValues>& obj) {
    const auto& params = obj.param;

    const auto& data_shape = std::get<0>(params);
    const auto& instance_norm_gamma_shape = std::get<1>(params);
    const auto& instance_norm_beta_shape = std::get<2>(params);
    const auto& group_norm_gamma_shape = std::get<3>(params);
    const auto& group_norm_beta_shape = std::get<4>(params);
    const auto& num_groups = std::get<5>(params);
    const auto& epsilon = std::get<6>(params);
    const auto& elem_type = std::get<7>(params);
    const auto& device_name = std::get<8>(params);
    const auto& device_properties = std::get<9>(params);

    std::ostringstream results;

    results << "T=" << elem_type << "_";
    results << "Input=" << utils::partialShape2str({data_shape}) << "_";
    results << "InstNormGamma=" << utils::partialShape2str({instance_norm_gamma_shape}) << "_";
    results << "InstNormBeta=" << utils::partialShape2str({instance_norm_beta_shape}) << "_";
    results << "GroupNormGamma=" << utils::partialShape2str({group_norm_gamma_shape}) << "_";
    results << "GroupNormBeta=" << utils::partialShape2str({group_norm_beta_shape}) << "_";
    results << "NumGroups=" << num_groups << "_";
    results << "Epsilon=" << epsilon << "_";
    results << "Device=" << device_name << "_";
    results << "DeviceCfg=(";
    for (auto iter = device_properties.begin(); iter != device_properties.end(); iter++) {
        results << iter->first << "=" << iter->second.as<std::string>();
        if (std::next(iter) != device_properties.end())
            results << "_";
    }
    results << ")";
    return results.str();
}

void GroupNormalizationFusionSubgraphTestsF::run() {
    configure_device();
    read_test_parameters();
    generate_weights_init_values();
    functionRefs = create_model();
    function = functionRefs->clone();
    pass::Manager m;
    m.register_pass<pass::GroupNormalizationFusion>();
    OV_ASSERT_NO_THROW(m.run_passes(function));

    ASSERT_EQ(count_ops_of_type<op::v12::GroupNormalization>(functionRefs), 0);
    ASSERT_EQ(count_ops_of_type<op::v12::GroupNormalization>(function), 1);

    ASSERT_FALSE(function->is_dynamic());

    auto input_shapes = static_partial_shapes_to_test_representation({data_shape});
    init_input_shapes(input_shapes);
    SubgraphBaseStaticTest::run();
}

void GroupNormalizationFusionSubgraphTestsF::init_thresholds() {
    size_t problem_size = shape_size(data_shape.get_shape());
    abs_threshold = pow(problem_size, 0.5) * utils::get_eps_by_ov_type(outType);
    rel_threshold = abs_threshold;
}

void GroupNormalizationFusionSubgraphTestsF::generate_inputs(const std::vector<Shape>& targetInputStaticShapes) {
    inputs.clear();

    auto itTargetShape = targetInputStaticShapes.begin();
    for (const auto& param : function->get_parameters()) {
        std::shared_ptr<Node> inputNode = param;
        for (size_t i = 0; i < param->get_output_size(); i++) {
            for (const auto& node : param->get_output_target_inputs(i)) {
                std::shared_ptr<Node> nodePtr = node.get_node()->shared_from_this();
                for (size_t port = 0; port < nodePtr->get_input_size(); ++port) {
                    if (nodePtr->get_input_node_ptr(port)->shared_from_this() == inputNode->shared_from_this()) {
                        const auto& tensor = utils::create_and_fill_tensor(inType, *itTargetShape);
                        inputs.insert({param, tensor});
                        break;
                    }
                }
            }
        }
        itTargetShape++;
    }
}

void GroupNormalizationFusionSubgraphTestsF::configure_device() {
    if (target_configuration.count(hint::inference_precision.name()) <= 0) {
        target_configuration.insert({hint::inference_precision.name(), elem_type});
    }
}

void GroupNormalizationFusionSubgraphTestsF::read_test_parameters() {
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
    target_device_name = std::get<8>(params);
    target_configuration = std::get<9>(params);

    instance_norm_gamma_present = (instance_norm_gamma_shape != Shape{});
    instance_norm_beta_present = (instance_norm_beta_shape != Shape{});

    inType = elem_type;
    outType = elem_type;
    targetDevice = target_device_name;
    configuration = target_configuration;

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

    instance_norm_gamma_present =
        instance_norm_gamma_present && (shape_size(instance_norm_gamma_shape) == static_cast<size_t>(num_groups));
    instance_norm_beta_present =
        instance_norm_beta_present && (shape_size(instance_norm_beta_shape) == static_cast<size_t>(num_groups));
}
}  // namespace test
}  // namespace ov
