// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/memory.hpp"

#include "openvino/pass/low_latency.hpp"
#include "openvino/pass/manager.hpp"
#include "template/properties.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/op/tensor_iterator.hpp"

namespace ov {
namespace test {

std::string MemoryLayerTest::getTestCaseName(const testing::TestParamInfo<MemoryLayerTestParams> &obj) {
    int64_t iteration_count;
    ov::element::Type model_type;
    ov::Shape input_shape;
    std::string target_device;
    ov::test::utils::MemoryTransformation transformation;
    std::tie(transformation, iteration_count, input_shape, model_type, target_device) = obj.param;

    std::ostringstream result;
    result << "transformation=" << transformation << "_";
    result << "iteration_count=" << iteration_count << "_";
    result << "IS=" << ov::test::utils::vec2str(input_shape) << "_";
    result << "modelType=" << model_type.get_type_name() << "_";
    result << "trgDev=" << target_device;
    result << ")";
    return result.str();
}

void MemoryLayerTest::SetUp() {
    ov::element::Type model_type;
    ov::Shape input_shape;
    ov::test::utils::MemoryTransformation transformation;

    std::tie(transformation, iteration_count, input_shape, model_type, targetDevice) = this->GetParam();

    if (transformation == ov::test::utils::MemoryTransformation::NONE) {
        CreateCommonFunc(model_type, input_shape);
    } else {
        CreateTIFunc(model_type, input_shape);
        ApplyLowLatency(transformation);
    }
}

void MemoryLayerTest::CreateCommonFunc(ov::element::Type model_type, ov::Shape input_shape) {
    ov::ParameterVector param {std::make_shared<ov::op::v0::Parameter>(model_type, input_shape)};
    const auto variable_info = ov::op::util::VariableInfo{input_shape, model_type, "v0"};
    auto variable = std::make_shared<ov::op::util::Variable>(variable_info);

    std::shared_ptr<ov::op::util::ReadValueBase> read_value;
    if (use_version_3) {
        read_value = std::make_shared<ov::op::v3::ReadValue>(param[0], variable->get_info().variable_id);
    } else {
        read_value = std::make_shared<ov::op::v6::ReadValue>(param[0], variable);
    }

    auto add = std::make_shared<ov::op::v1::Add>(read_value, param.at(0));

    std::shared_ptr<ov::op::util::AssignBase> assign;
    if (use_version_3) {
        assign = std::make_shared<ov::op::v3::Assign>(add, variable->get_info().variable_id);
    } else {
        assign = std::make_shared<ov::op::v6::Assign>(add, variable);
    }

    auto res = std::make_shared<ov::op::v0::Result>(add);
    function = std::make_shared<ov::Model>(ResultVector{res}, SinkVector{assign}, param, "TestMemory");
}

void MemoryLayerTest::CreateTIFunc(ov::element::Type model_type, ov::Shape input_shape) {
    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape(input_shape));

    std::vector<std::vector<size_t>> shape = {{static_cast<size_t>(iteration_count), 1}};
    auto iter_count = std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape{static_cast<size_t>(iteration_count), 1});

    // Body
    auto X = std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape(input_shape));
    auto Y = std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape(input_shape));
    auto Iter = std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape{1, 1});
    auto add = std::make_shared<ov::op::v1::Add>(X, Y);
    auto res = std::make_shared<ov::op::v0::Result>(add);
    auto Iter_res = std::make_shared<ov::op::v0::Result>(Iter);
    auto body = std::make_shared<ov::Model>(OutputVector{res, Iter_res}, ParameterVector {X, Y, Iter});

    // TI construction
    auto tensor_iterator = std::make_shared<ov::op::v0::TensorIterator>();
    tensor_iterator->set_body(body);

    tensor_iterator->set_merged_input(X, param, res);
    tensor_iterator->set_invariant_input(Y, param);
    tensor_iterator->set_sliced_input(Iter, iter_count, 0, 1, 1, -1, 0);

    auto output = tensor_iterator->get_iter_value(res, -1);
    auto output_iter = tensor_iterator->get_concatenated_slices(Iter_res, 0, 1, 1, -1, 0);
    function = std::make_shared<ov::Model>(OutputVector{output, output_iter},
                                           ParameterVector{param, iter_count},
                                           "PureTI");
}

void MemoryLayerTest::ApplyLowLatency(ov::test::utils::MemoryTransformation transformation) {
    if (transformation == ov::test::utils::MemoryTransformation::LOW_LATENCY_V2) {
        function->validate_nodes_and_infer_types();
        ov::pass::Manager manager;
        manager.register_pass<pass::LowLatency2>();
        manager.run_passes(function);
    } else if (transformation == ov::test::utils::MemoryTransformation::LOW_LATENCY_V2_ORIGINAL_INIT) {
        function->validate_nodes_and_infer_types();
        ov::pass::Manager manager;
        manager.register_pass<pass::LowLatency2>(false);
        manager.run_passes(function);
    }
}

void MemoryLayerTest::infer() {
    inferRequest = compiledModel.create_infer_request();
    for (size_t iter = 0; iter <= iteration_count; iter++) {
        for (const auto& input : inputs) {
            inferRequest.set_tensor(input.first, input.second);
        }
        inferRequest.infer();
    }
}

std::vector<ov::Tensor> MemoryLayerTest::calculate_refs() {
    if (is_report_stages) {
        std::cout << "[ REFERENCE   ] `SubgraphBaseTest::calculate_refs()` is started"<< std::endl;
    }
    auto start_time = std::chrono::system_clock::now();

    update_ref_model();
    match_parameters(function->get_parameters(), functionRefs->get_parameters());

    auto compiledModelRef = core->compile_model(functionRefs, ov::test::utils::DEVICE_TEMPLATE, {{ ov::template_plugin::disable_transformations(true) }});
    auto inferRequestRef = compiledModelRef.create_infer_request();

    for (size_t iter = 0; iter <= iteration_count; iter++) {
        for (const auto& param : functionRefs->get_parameters()) {
            inferRequestRef.set_tensor(param->get_default_output(), inputs.at(matched_parameters[param]));
        }
        inferRequestRef.infer();
    }
    auto outputs = std::vector<ov::Tensor>{};
    for (const auto& output : functionRefs->outputs()) {
        outputs.push_back(inferRequestRef.get_tensor(output));
    }
    if (is_report_stages) {
        auto end_time = std::chrono::system_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;
        std::cout << "[ REFERENCE   ] `SubgraphBaseTest::calculate_refs()` is finished successfully. Duration is " << duration.count() << "s" << std::endl;
    }
    return outputs;
}

void MemoryV3LayerTest::SetUp() {
    use_version_3 = true;
    MemoryLayerTest::SetUp();
}

}  // namespace test
}  // namespace ov

