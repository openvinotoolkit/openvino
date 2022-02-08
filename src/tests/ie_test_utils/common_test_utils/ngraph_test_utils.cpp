// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional_test_utils/include/functional_test_utils/blob_utils.hpp>
#include "ngraph_test_utils.hpp"
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <functional_test_utils/include/functional_test_utils/ov_tensor_utils.hpp>


void TransformationTestsF::accuracy_check(const std::shared_ptr<ov::Model>& ref_function,
                                          const std::shared_ptr<ov::Model>& cur_function) {
    try {
        if (ref_function->is_dynamic() || cur_function->is_dynamic()) {
            return;
        }
        std::map<std::shared_ptr<ov::Node>, ov::Tensor> input_data;
        for (const auto& param : ref_function->get_parameters()) {
            const auto &tensor = ov::test::utils::create_and_fill_tensor(param->get_element_type(),
                                                                        param->get_shape());
            input_data[param] = tensor;
        }

        auto ref_outputs = ngraph::helpers::interpretFunction(ref_function, input_data);
        auto outputs = ngraph::helpers::interpretFunction(ref_function, input_data);

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
