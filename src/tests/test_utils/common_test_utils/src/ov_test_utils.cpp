// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_test_utils.hpp"

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ov_plugin_cache.hpp"
#include "common_test_utils/test_constants.hpp"
#include "openvino/op/tensor_iterator.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/util/file_util.hpp"
#include "template/properties.hpp"

namespace ov {
namespace pass {

class CopyTensorNamesToRefModel : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("CopyTensorNamesToRefModel");
    CopyTensorNamesToRefModel(const std::shared_ptr<ov::Model>& ref_model) : m_ref_model(ref_model) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& f) override {
        const auto& orig_results = f->get_results();
        const auto& ref_results = m_ref_model->get_results();
        for (size_t idx = 0; idx < orig_results.size(); ++idx) {
            auto ref_res_tensor = ref_results[idx]->input_value(0).get_tensor_ptr();
            auto ref_res_tensor_names = ref_res_tensor->get_names();
            if (ref_res_tensor_names.empty()) {
                auto orig_names = orig_results[idx]->input_value(0).get_tensor_ptr()->get_names();
                ref_res_tensor->add_names(orig_names);
            }
        }
        return false;
    }

private:
    // we store a reference to shared_ptr because it will be initialized later in the test body
    const std::shared_ptr<ov::Model>& m_ref_model;
};

}  // namespace pass
}  // namespace ov

TransformationTestsF::TransformationTestsF() : comparator(FunctionsComparator::no_default()) {
    m_unh = std::make_shared<ov::pass::UniqueNamesHolder>();
    comparator.enable(FunctionsComparator::CmpValues::NODES);
    comparator.enable(FunctionsComparator::CmpValues::PRECISIONS);
    comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
    comparator.enable(FunctionsComparator::CmpValues::SUBGRAPH_DESCRIPTORS);
    // TODO: enable attributes and constant values comparison by default XXX-98039
    // comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    // comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    // comparator.enable(FunctionsComparator::CmpValues::NAMES);
}

void TransformationTestsF::SetUp() {
    manager.register_pass<ov::pass::InitUniqueNames>(m_unh);
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::pass::CopyTensorNamesToRefModel>(model_ref);
}

void TransformationTestsF::TearDown() {
    if (test_skipped) {
        return;
    }
    OPENVINO_ASSERT(model != nullptr, "Test Model is not initialized.");

    std::shared_ptr<ov::Model> cloned_function;
    auto acc_enabled = comparator.should_compare(FunctionsComparator::ACCURACY);
    if (!model_ref) {
        cloned_function = model->clone();
        model_ref = cloned_function;
    } else if (acc_enabled) {
        cloned_function = model->clone();
    }
    manager.register_pass<ov::pass::CheckUniqueNames>(m_unh, m_soft_names_comparison, m_result_friendly_names_check);
    manager.run_passes(model);

    if (!m_disable_rt_info_check) {
        OV_ASSERT_NO_THROW(check_rt_info(model));
    }

    if (acc_enabled) {
        OPENVINO_ASSERT(cloned_function != nullptr, "Accuracy cannot be checked. Cloned Model is not initialized.");
        auto acc_comparator = FunctionsComparator::no_default();
        acc_comparator.set_accuracy_thresholds(m_abs_threshold, m_rel_threshold);
        acc_comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
        auto res = acc_comparator.compare(model, cloned_function);
        ASSERT_TRUE(res.valid) << res.message;
        comparator.disable(FunctionsComparator::CmpValues::ACCURACY);
    }

    auto res = comparator.compare(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

void TransformationTestsF::disable_rt_info_check() {
    m_disable_rt_info_check = true;
}

void TransformationTestsF::enable_soft_names_comparison() {
    m_soft_names_comparison = true;
}

void TransformationTestsF::disable_result_friendly_names_check() {
    m_result_friendly_names_check = false;
}

void init_unique_names(const std::shared_ptr<ov::Model>& f, const std::shared_ptr<ov::pass::UniqueNamesHolder>& unh) {
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitUniqueNames>(unh);
    manager.run_passes(f);
}

void check_unique_names(const std::shared_ptr<ov::Model>& f, const std::shared_ptr<ov::pass::UniqueNamesHolder>& unh) {
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::CheckUniqueNames>(unh, true);
    manager.run_passes(f);
}

namespace ov {
namespace test {
namespace utils {

ov::TensorVector infer_on_template(const std::shared_ptr<ov::Model>& model, const ov::TensorVector& input_tensors) {
    std::map<std::shared_ptr<ov::Node>, ov::Tensor> inputs;
    auto params = model->inputs();
    OPENVINO_ASSERT(params.size() == input_tensors.size());
    for (int i = 0; i < params.size(); i++) {
        inputs[params[i].get_node_shared_ptr()] = input_tensors[i];
    }
    return infer_on_template(model, inputs);
}

ov::TensorVector infer_on_template(const std::shared_ptr<ov::Model>& model,
                                   const std::map<std::shared_ptr<ov::Node>, ov::Tensor>& inputs) {
    auto core = ov::test::utils::PluginCache::get().core();

    auto compiled_model = core->compile_model(model,
                                              ov::test::utils::DEVICE_TEMPLATE,
                                              {{ov::template_plugin::disable_transformations(true)}});
    auto infer_request = compiled_model.create_infer_request();

    for (auto& input : inputs) {
        infer_request.set_tensor(input.first, input.second);
    }
    infer_request.infer();

    ov::TensorVector outputs;
    for (const auto& output : model->outputs()) {
        outputs.push_back(infer_request.get_tensor(output));
    }

    return outputs;
}

bool is_tensor_iterator_exist(const std::shared_ptr<ov::Model>& model) {
    const auto& ops = model->get_ops();
    for (const auto& node : ops) {
        const auto& ti = ov::as_type_ptr<ov::op::v0::TensorIterator>(node);
        if (ti) {
            return true;
        }
    }
    return false;
}

}  // namespace utils
}  // namespace test
}  // namespace ov
