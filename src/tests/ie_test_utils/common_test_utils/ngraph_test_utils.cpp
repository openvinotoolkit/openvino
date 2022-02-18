// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional_test_utils/include/functional_test_utils/blob_utils.hpp>
#include "ngraph_test_utils.hpp"
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include "ov_tensor_utils.hpp"


void TransformationTestsF::accuracy_check(const std::shared_ptr<ov::Model>& ref_function,
                                          const std::shared_ptr<ov::Model>& cur_function) {
    try {
        if (ref_function->is_dynamic() || cur_function->is_dynamic()) {
            return;
        }
        IE_ASSERT(ref_function->get_parameters().size() == cur_function->get_parameters().size());

        std::map<std::shared_ptr<ov::Node>, ov::Tensor> ref_input_data;
        std::map<std::shared_ptr<ov::Node>, ov::Tensor> cur_input_data;
        for (size_t i=0; i < ref_function->get_parameters().size(); i++) {
            const auto &tensor = ov::test::utils::create_and_fill_tensor(ref_function->get_parameters()[i]->get_element_type(),
                                                                         ref_function->get_parameters()[i]->get_shape());
            ref_input_data[ref_function->get_parameters()[i]] = tensor;
            cur_input_data[cur_function->get_parameters()[i]] = tensor;
        }

        auto ref_outputs = ngraph::helpers::interpretFunction(ref_function, ref_input_data);
        auto outputs = ngraph::helpers::interpretFunction(cur_function, cur_input_data);

        IE_ASSERT(ref_outputs.size() == outputs.size());

        for (int i=0; i < ref_outputs.size(); i++) {
            ov::test::utils::compare(ref_outputs[i], outputs[i],
                                     std::numeric_limits<double>::max(),
                                     std::numeric_limits<double>::max());
        }
    }
    catch (const std::runtime_error &re) {
        GTEST_FATAL_FAILURE_(re.what());
    } catch (const std::exception &ex) {
        GTEST_FATAL_FAILURE_(ex.what());
    } catch (...) {
        GTEST_FATAL_FAILURE_("Unknown failure occurred.");
    }
}

void init_unique_names(std::shared_ptr<ngraph::Function> f, const std::shared_ptr<ngraph::pass::UniqueNamesHolder>& unh) {
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::InitUniqueNames>(unh);
    manager.run_passes(f);
}

void check_unique_names(std::shared_ptr<ngraph::Function> f, const std::shared_ptr<ngraph::pass::UniqueNamesHolder>& unh) {
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::CheckUniqueNames>(unh, true);
    manager.run_passes(f);
}
