// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "functional_test_utils/crash_handler.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "transformations/common_optimizations/group_normalization_fusion.hpp"

using namespace testing;

namespace ov {
namespace test {

using GroupNormalizationFusionTestBaseValues =
    std::tuple<ov::PartialShape,    // (partial) shape of input/output tensor (all dims except channel can be dynamic)
               ov::Shape,           // shape of optional instance norm gamma tensor (or empty shape if not used)
               ov::Shape,           // shape of optional instance norm beta tensor (or empty shape if not used)
               ov::Shape,           // shape of group norm gamma tensor
               ov::Shape,           // shape of group norm beta tensor
               unsigned long long,  // number of groups
               float>;              // epsilon

using GroupNormalizationFusionTransformationsTestValues =
    std::tuple<PartialShape,        // (partial) shape of input/output tensor (all dims except channel can be dynamic)
               Shape,               // shape of optional instance norm gamma tensor (or empty shape if not used)
               Shape,               // shape of optional instance norm beta tensor (or empty shape if not used)
               Shape,               // shape of group norm gamma tensor
               Shape,               // shape of group norm beta tensor
               unsigned long long,  // number of groups
               float,               // epsilon
               bool,                // whether it's a positive test that should run reference model or a negative test
               std::string,         // taget device name
               ov::AnyMap,          // taget device properties
               std::string,         // reference device name
               ov::AnyMap>;         // reference device properties

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

template <element::Type_t T_elem>
class GroupNormalizationFusionTestBase {
public:
    static constexpr element::Type T_elem_t = T_elem;
    typedef typename ov::element_type_traits<T_elem_t>::value_type T_store_t;

protected:
    size_t numChannels;
    bool instanceNormGammaPresent;
    bool instanceNormBetaPresent;

    std::vector<T_store_t> instanceNormGammaVals;
    std::vector<T_store_t> instanceNormBetaVals;
    std::vector<T_store_t> groupNormGammaVals;
    std::vector<T_store_t> groupNormBetaVals;

    PartialShape dataShape;
    Shape instanceNormGammaShape;
    Shape instanceNormBetaShape;
    Shape groupNormGammaShape;
    Shape groupNormBetaShape;
    size_t numGroups;
    float epsilon;

    virtual void read_test_parameters() = 0;

    void generate_weights_init_values() {
        if (instanceNormGammaPresent)
            instanceNormGammaVals = test::utils::generateVector<T_elem_t>(shape_size(instanceNormGammaShape), 10, 1, 1);
        if (instanceNormBetaPresent)
            instanceNormBetaVals = test::utils::generateVector<T_elem_t>(shape_size(instanceNormBetaShape), 10, 1, 2);
        groupNormGammaVals = test::utils::generateVector<T_elem_t>(shape_size(groupNormGammaShape), 10, 1, 3);
        groupNormBetaVals = test::utils::generateVector<T_elem_t>(shape_size(groupNormBetaShape), 10, 1, 4);
    }

    std::shared_ptr<ov::Model> create_model() {
        auto input = std::make_shared<op::v0::Parameter>(T_elem_t, dataShape);
        auto pre_mvn_shape_const =
            op::v0::Constant::create<long long>(element::i64, Shape{3}, {0, static_cast<long long>(numGroups), -1});
        auto pre_mvn_reshape = std::make_shared<ov::op::v1::Reshape>(input, pre_mvn_shape_const, true);

        auto mvn_axes_const = op::v0::Constant::create<long long>(element::i64, Shape{1}, {2});
        auto mvn =
            std::make_shared<op::v6::MVN>(pre_mvn_reshape, mvn_axes_const, true, epsilon, op::MVNEpsMode::INSIDE_SQRT);

        std::shared_ptr<Node> opt_instance_norm_gamma_multiply = mvn;
        if (instanceNormGammaPresent) {
            auto instance_norm_gamma_const =
                op::v0::Constant::create(T_elem_t, instanceNormGammaShape, instanceNormGammaVals);
            opt_instance_norm_gamma_multiply = std::make_shared<op::v1::Multiply>(mvn, instance_norm_gamma_const);
        }

        std::shared_ptr<ov::Node> opt_instance_norm_beta_add = opt_instance_norm_gamma_multiply;
        if (instanceNormBetaPresent) {
            auto instance_norm_beta_const =
                op::v0::Constant::create(T_elem_t, instanceNormBetaShape, instanceNormBetaVals);
            opt_instance_norm_beta_add =
                std::make_shared<ov::op::v1::Add>(opt_instance_norm_gamma_multiply, instance_norm_beta_const);
        }

        auto post_instance_norm_shape = std::make_shared<ov::op::v0::ShapeOf>(input);

        auto post_instance_norm_reshape =
            std::make_shared<op::v1::Reshape>(opt_instance_norm_beta_add, post_instance_norm_shape, true);

        auto group_norm_gamma_const = op::v0::Constant::create(T_elem_t, groupNormGammaShape, groupNormGammaVals);
        auto group_norm_gamma_multiply =
            std::make_shared<op::v1::Multiply>(post_instance_norm_reshape, group_norm_gamma_const);

        auto group_norm_beta_const = op::v0::Constant::create(T_elem_t, groupNormBetaShape, groupNormBetaVals);
        auto group_norm_beta_add = std::make_shared<op::v1::Add>(group_norm_gamma_multiply, group_norm_beta_const);

        return std::make_shared<Model>(NodeVector{group_norm_beta_add}, ParameterVector{input});
    }
};

template <element::Type_t T_elem>
class GroupNormalizationFusionSubgraphTestsF
    : public GroupNormalizationFusionTestBase<T_elem>,
      public ov::test::SubgraphBaseTest,
      public testing::WithParamInterface<GroupNormalizationFusionTransformationsTestValues> {
public:
    static std::string getTestCaseName(
        const testing::TestParamInfo<GroupNormalizationFusionTransformationsTestValues>& obj) {
        const auto& params = obj.param;

        const auto& data_shape = std::get<0>(params);
        const auto& instance_norm_gamma_shape = std::get<1>(params);
        const auto& instance_norm_beta_shape = std::get<2>(params);
        const auto& group_norm_gamma_shape = std::get<3>(params);
        const auto& group_norm_beta_shape = std::get<4>(params);
        const auto& num_groups = std::get<5>(params);
        const auto& epsilon = std::get<6>(params);
        const auto& positive_test = std::get<7>(params);
        const auto& device_name = std::get<8>(params);
        const auto& device_properties = std::get<9>(params);
        const auto& ref_device_name = std::get<10>(params);
        const auto& ref_device_properties = std::get<11>(params);

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
        results << "Device=" << device_name << "_";
        results << "DeviceCfg=(";
        for (auto iter = device_properties.begin(); iter != device_properties.end(); iter++) {
            results << iter->first << "=" << iter->second.as<std::string>();
            if (std::next(iter) != device_properties.end())
                results << "_";
        }
        results << ")_";
        results << "RefDevice=" << ref_device_name << "_";
        results << "RefDeviceCfg=(";
        for (auto iter = ref_device_properties.begin(); iter != ref_device_properties.end(); iter++) {
            results << iter->first << "=" << iter->second.as<std::string>();
            if (std::next(iter) != ref_device_properties.end())
                results << "_";
        }
        results << ")";
        return results.str();
    }

protected:
    bool positiveTest;
    std::string targetDeviceName;
    ov::AnyMap targetConfiguration;
    std::string refDevice;
    ov::AnyMap refConfiguration;

    ElementType refInferencePrecision;
    ov::CompiledModel compiledRefModel;
    ov::InferRequest refInferRequest;

    void TearDown() override {
        SubgraphBaseTest::TearDown();
    }

    virtual void read_test_parameters() {
        const auto& params = GetParam();

        dataShape = std::get<0>(params);
        if (!dataShape.rank().is_static())
            throw std::runtime_error("Rank of input tensor has to be static!");
        if (dataShape.rank().get_max_length() < 2)
            throw std::runtime_error("Expected at least two dimensions in input tensor!");
        if (!dataShape[1].is_static())
            throw std::runtime_error("Channel dimension in input tensor has to be static!");

        numChannels = static_cast<size_t>(dataShape[1].get_max_length());
        instanceNormGammaShape = std::get<1>(params);
        instanceNormBetaShape = std::get<2>(params);
        groupNormGammaShape = std::get<3>(params);
        groupNormBetaShape = std::get<4>(params);
        numGroups = std::get<5>(params);
        epsilon = std::get<6>(params);
        positiveTest = std::get<7>(params);
        targetDeviceName = std::get<8>(params);
        targetConfiguration = std::get<9>(params);
        refDevice = std::get<10>(params);
        refConfiguration = std::get<11>(params);

        instanceNormGammaPresent = (instanceNormGammaShape != Shape{});
        instanceNormBetaPresent = (instanceNormBetaShape != Shape{});

        inType = T_elem_t;
        outType = T_elem_t;
        targetDevice = targetDeviceName;
        configuration = targetConfiguration;

        if (positiveTest) {
            if ((instanceNormGammaShape != Shape{}) && (shape_size(instanceNormGammaShape) != numGroups))
                throw std::runtime_error("Shape of instance norm gamma has to either be empty or contain "
                                         "exactly <numGroups> elements");
            if ((instanceNormBetaShape != Shape{}) && (shape_size(instanceNormBetaShape) != numGroups))
                throw std::runtime_error("Shape of instance norm beta has to either be empty shape or contain "
                                         "exactly <numGroups> elements");
            if (shape_size(groupNormGammaShape) != numChannels)
                throw std::runtime_error("Shape of group norm gamma has to contain exactly <numChannels> elements");
            if (shape_size(groupNormBetaShape) != numChannels)
                throw std::runtime_error("Shape of group norm beta has to contain exactly <numChannels> elements");

            instanceNormGammaPresent = instanceNormGammaPresent && (shape_size(instanceNormGammaShape) == numGroups);
            instanceNormBetaPresent = instanceNormBetaPresent && (shape_size(instanceNormBetaShape) == numGroups);
        }
    }

    void configure_device() {
        if (targetConfiguration.count(ov::hint::inference_precision.name()) <= 0) {
            targetConfiguration.insert({ov::hint::inference_precision.name(), T_elem_t});
        }
    }

    void configure_ref_device() {
        if (refConfiguration.count(ov::hint::inference_precision.name()) <= 0) {
            refConfiguration.insert({ov::hint::inference_precision.name(), T_elem_t});
        }
    }

    void configure_ref_model() {
        // configure input precision
        ov::preprocess::PrePostProcessor p(functionRefs);
        {
            auto& params = functionRefs->get_parameters();
            for (size_t i = 0; i < params.size(); i++) {
                if (inType != ov::element::Type_t::undefined) {
                    p.input(i).tensor().set_element_type(inType);
                }
            }
        }

        // configure output precision
        {
            auto results = functionRefs->get_results();
            for (size_t i = 0; i < results.size(); i++) {
                if (outType != ov::element::Type_t::undefined) {
                    p.output(i).tensor().set_element_type(outType);
                }
            }
        }
        functionRefs = p.build();
    }

    void compile_ref_model() {
        if (is_report_stages) {
            std::cout << "[ REFERENCE   ] `GroupNormalizationFusionSubgraphTestsF::compile_ref_model()` is started"
                      << std::endl;
        }
        auto start_time = std::chrono::system_clock::now();

        configure_ref_model();
        core_configuration(this);
        compiledRefModel = core->compile_model(functionRefs, refDevice, refConfiguration);
        if (is_report_stages) {
            auto end_time = std::chrono::system_clock::now();
            std::chrono::duration<double> duration = end_time - start_time;
            std::cout << "[ REFERENCE   ] `GroupNormalizationFusionSubgraphTestsF::compile_ref_model()` is finished "
                         "successfully. Duration is "
                      << duration.count() << "s" << std::endl;
        }
        try {
            refInferencePrecision = core->get_property(refDevice, ov::hint::inference_precision);
        } catch (std::exception& e) {
            std::cout << "[ WARNING ] Impossible to get Inference Precision with exception: " << e.what() << std::endl;
        }
    }

    void init_thresholds() override {
        if (!targetStaticShapes.empty()) {
            size_t problem_size = shape_size(dataShape.get_shape());

            abs_threshold = pow(problem_size, 0.5) * test::utils::get_eps_by_ov_type(outType);
            rel_threshold = abs_threshold;
        }
    }

    void infer_ref(const std::map<std::shared_ptr<ov::Node>, ov::Tensor>& inputs_ref) {
        refInferRequest = compiledRefModel.create_infer_request();
        for (const auto& input : inputs_ref) {
            refInferRequest.set_tensor(input.first, input.second);
        }
        refInferRequest.infer();
    }

    std::vector<ov::Tensor> calculate_refs() {
        if (is_report_stages) {
            std::cout << "[ REFERENCE   ] `GroupNormalizationFusionSubgraphTestsF::calculate_refs()` is started"
                      << std::endl;
        }
        auto start_time = std::chrono::system_clock::now();

        update_ref_model();
        match_parameters(function->get_parameters(), functionRefs->get_parameters());

        std::map<std::shared_ptr<ov::Node>, ov::Tensor> inputs_ref;
        for (const auto& param : functionRefs->get_parameters()) {
            inputs_ref[param] = inputs.at(matched_parameters[param]);
        }

        infer_ref(inputs_ref);
        auto outputs = std::vector<ov::Tensor>{};
        for (const auto& output : functionRefs->outputs()) {
            outputs.push_back(refInferRequest.get_tensor(output));
        }
        if (is_report_stages) {
            auto end_time = std::chrono::system_clock::now();
            std::chrono::duration<double> duration = end_time - start_time;
            std::cout << "[ REFERENCE   ] `GroupNormalizationFusionSubgraphTestsF::calculate_refs()` is finished "
                         "successfully. Duration is "
                      << duration.count() << "s" << std::endl;
        }
        return outputs;
    }

    virtual void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();

        auto itTargetShape = targetInputStaticShapes.begin();
        for (const auto& param : function->get_parameters()) {
            std::shared_ptr<ov::Node> inputNode = param;
            for (size_t i = 0; i < param->get_output_size(); i++) {
                for (const auto& node : param->get_output_target_inputs(i)) {
                    std::shared_ptr<ov::Node> nodePtr = node.get_node()->shared_from_this();
                    for (size_t port = 0; port < nodePtr->get_input_size(); ++port) {
                        if (nodePtr->get_input_node_ptr(port)->shared_from_this() == inputNode->shared_from_this()) {
                            const auto& tensor = ov::test::utils::create_and_fill_tensor(inType, *itTargetShape);
                            inputs.insert({param, tensor});
                            break;
                        }
                    }
                }
            }
            itTargetShape++;
        }
    }

public:
    void run() {
        is_reported = true;
        bool isCurrentTestDisabled = ov::test::utils::current_test_is_disabled();

        ov::test::utils::PassRate::Statuses status = isCurrentTestDisabled
                                                         ? ov::test::utils::PassRate::Statuses::SKIPPED
                                                         : ov::test::utils::PassRate::Statuses::CRASHED;

        if (isCurrentTestDisabled)
            GTEST_SKIP() << "Disabled test due to configuration" << std::endl;

        // in case of crash jump will be made and work will be continued
        auto crashHandler = std::unique_ptr<ov::test::utils::CrashHandler>(new ov::test::utils::CrashHandler());

        // place to jump in case of a crash
        int jmpRes = 0;
#ifdef _WIN32
        jmpRes = setjmp(ov::test::utils::env);
#else
        jmpRes = sigsetjmp(ov::test::utils::env, 1);
#endif
        if (jmpRes == ov::test::utils::JMP_STATUS::ok) {
            crashHandler->StartTimer();
            std::string errorMessage;
            try {
                read_test_parameters();
                generate_weights_init_values();
                functionRefs = create_model();
                function = functionRefs->clone();
                pass::Manager m;
                m.register_pass<ov::pass::GroupNormalizationFusion>();
                OV_ASSERT_NO_THROW(m.run_passes(function));

                summary.setDeviceName(targetDevice);
                summary.updateOPsStats(function, status, rel_influence_coef);
                if (positiveTest) {
                    ASSERT_EQ(count_ops_of_type<ov::op::v12::GroupNormalization>(functionRefs), 0);
                    ASSERT_EQ(count_ops_of_type<ov::op::v12::GroupNormalization>(function), 1);

                    if (!function->is_dynamic()) {
                        configure_device();
                        configure_ref_device();
                        auto input_shapes = static_partial_shapes_to_test_representation({dataShape});
                        init_input_shapes(input_shapes);
                        ASSERT_FALSE(targetStaticShapes.empty() && !function->get_parameters().empty())
                            << "Target Static Shape is empty!!!";
                        compile_model();
                        compile_ref_model();
                        for (const auto& targetStaticShapeVec : targetStaticShapes) {
                            generate_inputs(targetStaticShapeVec);
                            validate();
                        }
                    }
                } else {
                    ASSERT_EQ(count_ops_of_type<ov::op::v12::GroupNormalization>(functionRefs), 0);
                    ASSERT_EQ(count_ops_of_type<ov::op::v12::GroupNormalization>(function), 0);
                }
                status = ov::test::utils::PassRate::Statuses::PASSED;
            } catch (const std::exception& ex) {
                if (callback_exception != nullptr) {
                    // exception will be checked by callback.
                    callback_exception(ex);
                    return;
                } else {
                    status = ov::test::utils::PassRate::Statuses::FAILED;
                    errorMessage = ex.what();
                }
            } catch (...) {
                status = ov::test::utils::PassRate::Statuses::FAILED;
                errorMessage = "Unknown failure occurred.";
            }
            summary.updateOPsStats(function, status, rel_influence_coef);
            if (status != ov::test::utils::PassRate::Statuses::PASSED) {
                GTEST_FATAL_FAILURE_(errorMessage.c_str());
            }
        } else if (jmpRes == ov::test::utils::JMP_STATUS::anyError) {
            OPENVINO_THROW("Crash happens");
        } else if (jmpRes == ov::test::utils::JMP_STATUS::alarmErr) {
            summary.updateOPsStats(function, ov::test::utils::PassRate::Statuses::HANGED, rel_influence_coef);
            OPENVINO_THROW("Crash happens");
        }
    }
};

class GroupNormalizationFusionSubgraphTestsF_f32 : public GroupNormalizationFusionSubgraphTestsF<element::Type_t::f32> {
};
class GroupNormalizationFusionSubgraphTestsF_f16 : public GroupNormalizationFusionSubgraphTestsF<element::Type_t::f16> {
};
class GroupNormalizationFusionSubgraphTestsF_bf16
    : public GroupNormalizationFusionSubgraphTestsF<element::Type_t::bf16> {};

class GroupNormalizationFusionSubgraphTestsF_u8 : public GroupNormalizationFusionSubgraphTestsF<element::Type_t::u8> {};
class GroupNormalizationFusionSubgraphTestsF_u16 : public GroupNormalizationFusionSubgraphTestsF<element::Type_t::u16> {
};
class GroupNormalizationFusionSubgraphTestsF_u32 : public GroupNormalizationFusionSubgraphTestsF<element::Type_t::u32> {
};
class GroupNormalizationFusionSubgraphTestsF_u64 : public GroupNormalizationFusionSubgraphTestsF<element::Type_t::u64> {
};
class GroupNormalizationFusionSubgraphTestsF_i8 : public GroupNormalizationFusionSubgraphTestsF<element::Type_t::i8> {};
class GroupNormalizationFusionSubgraphTestsF_i16 : public GroupNormalizationFusionSubgraphTestsF<element::Type_t::i16> {
};
class GroupNormalizationFusionSubgraphTestsF_i32 : public GroupNormalizationFusionSubgraphTestsF<element::Type_t::i32> {
};
class GroupNormalizationFusionSubgraphTestsF_i64 : public GroupNormalizationFusionSubgraphTestsF<element::Type_t::i64> {
};
class GroupNormalizationFusionSubgraphTestsF_f8e4m3
    : public GroupNormalizationFusionSubgraphTestsF<element::Type_t::f8e4m3> {};
class GroupNormalizationFusionSubgraphTestsF_f8e5m2
    : public GroupNormalizationFusionSubgraphTestsF<element::Type_t::f8e5m2> {};
class GroupNormalizationFusionSubgraphTestsF_f4e2m1
    : public GroupNormalizationFusionSubgraphTestsF<element::Type_t::f4e2m1> {};
class GroupNormalizationFusionSubgraphTestsF_f8e8m0
    : public GroupNormalizationFusionSubgraphTestsF<element::Type_t::f8e8m0> {};

}  // namespace test
}  // namespace ov
