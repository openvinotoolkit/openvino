// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_test_utils.hpp"

namespace ov {
namespace pass {

class CopyTensorNamesToRefModel : public ov::pass::ModelPass {
public:
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
        ASSERT_NO_THROW(check_rt_info(model));
    }

    if (acc_enabled) {
        OPENVINO_ASSERT(cloned_function != nullptr, "Accuracy cannot be checked. Cloned Model is not initialized.");
        auto acc_comparator = FunctionsComparator::no_default();
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

std::shared_ptr<ov::op::v0::Constant> create_zero_constant(const ov::element::Type_t& et, const ov::Shape& shape) {
    return ov::op::v0::Constant::create(et, shape, {0});
}
