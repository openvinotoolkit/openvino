// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/test_common.hpp"
#include "common_test_utils/ov_plugin_cache.hpp"
#include "functional_test_utils/summary/op_summary.hpp"
#include "openvino/core/model.hpp"
#include "transformations/convert_precision.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

namespace ov {
namespace test {

using InputShape = std::pair<ov::PartialShape, std::vector<ov::Shape>>;
std::ostream& operator<<(std::ostream& os, const InputShape& inputShape);

using ElementType = ov::element::Type_t;
using Config = ov::AnyMap;
using TargetDevice = std::string;

typedef std::tuple<ov::element::Type,  // Input element type
                   ov::Shape,          // Input Shape
                   TargetDevice        // Target Device
                   >
    BasicParams;

class SubgraphBaseTest : public ov::test::TestsCommon {
public:
    virtual void run();
    virtual void serialize();
    virtual void query_model();
    virtual void import_export();

protected:
    virtual void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual);
    virtual void compile_model();
    virtual void infer();
    virtual void validate();
    virtual void configure_model();
    virtual void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes);

    void compare_models_param_res(const std::shared_ptr<ov::Model>& f, const std::shared_ptr<ov::Model>& f_ref);
    void compare_nodes(const std::shared_ptr<ov::Node>& node1, const std::shared_ptr<ov::Node>& node2, std::ostream& err_log);
    void update_ref_model();
    void match_parameters(const ov::ParameterVector& params, const ov::ParameterVector& ref_params);
    void init_thresholds();
    void init_input_shapes(const std::vector<InputShape>& shapes);
    void set_callback_exception(std::function<void(const std::exception& exp)> callback) { callback_exception = callback; }

    void TearDown() override {
        if (this->HasFailure() && !is_reported) {
            summary.setDeviceName(targetDevice);
            summary.updateOPsStats(function, ov::test::utils::PassRate::Statuses::FAILED, rel_influence_coef);
        }
    }

    std::shared_ptr<ov::Core> core = ov::test::utils::PluginCache::get().core();
    std::string targetDevice;
    ov::AnyMap configuration;

    std::shared_ptr<ov::Model> function, functionRefs = nullptr;
    std::map<std::shared_ptr<ov::Node>, ov::Tensor> inputs;
    std::vector<ov::PartialShape> inputDynamicShapes;
    std::vector<std::vector<ov::Shape>> targetStaticShapes;
    ElementType inType = ov::element::undefined,
                outType = ov::element::undefined,
                inference_precision = ov::element::undefined;

    ov::CompiledModel compiledModel;
    ov::InferRequest inferRequest;

    // to provide correct inputs for reference function
    std::map<std::shared_ptr<ov::op::v0::Parameter>, std::shared_ptr<ov::op::v0::Parameter>> matched_parameters;
    precisions_map convert_precisions;

    std::function<void(const std::exception& exp)> callback_exception = nullptr;

    constexpr static const double disable_threshold = -1;
    constexpr static const double disable_tensor_metrics = 1.f;
    double abs_threshold = disable_threshold,
           rel_threshold = disable_threshold,
           topk_threshold = disable_tensor_metrics,
           mvn_threshold = disable_tensor_metrics;

    ov::test::utils::OpSummary& summary = ov::test::utils::OpSummary::getInstance();
    bool is_report_stages = false;
    bool is_reported = false;
    double rel_influence_coef = 1.f;

    virtual std::vector<ov::Tensor> calculate_refs();
    virtual std::vector<ov::Tensor> get_plugin_outputs();

    friend void core_configuration(SubgraphBaseTest* test);
};

inline std::vector<InputShape> static_partial_shapes_to_test_representation(
    const std::vector<ov::PartialShape>& shapes) {
    std::vector<InputShape> result;
    for (const auto& staticShape : shapes) {
        if (staticShape.is_dynamic())
            throw std::runtime_error(
                "static_partial_shapes_to_test_representation can process only static partial shapes");
        result.push_back({{staticShape}, {staticShape.get_shape()}});
    }
    return result;
}

inline std::vector<std::vector<InputShape>> static_shapes_to_test_representation(
    const std::vector<std::vector<ov::Shape>>& shapes) {
    std::vector<std::vector<InputShape>> result;
    for (const auto& staticShapes : shapes) {
        std::vector<InputShape> tmp;
        for (const auto& staticShape : staticShapes) {
            tmp.push_back({{}, {staticShape}});
        }
        result.push_back(tmp);
    }
    return result;
}

inline std::vector<InputShape> static_shapes_to_test_representation(const std::vector<ov::Shape>& shapes) {
    std::vector<InputShape> result;
    for (const auto& staticShape : shapes) {
        result.push_back({{}, {staticShape}});
    }
    return result;
}

class SubgraphBaseStaticTest : public ov::test::SubgraphBaseTest {
public:
    void run() override {
        std::vector<ov::Shape> input_shapes;
        for (const auto& param : function->get_parameters())
            input_shapes.emplace_back(param->get_shape());
        init_input_shapes(ov::test::static_shapes_to_test_representation(input_shapes));
        ov::test::SubgraphBaseTest::run();
    }
};
}  // namespace test
}  // namespace ov
