// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_ngraph_utils.hpp"

#include "cnn_network_ngraph_impl.hpp"
#include "ie_itt.hpp"
#include "transformations/utils/utils.hpp"

namespace InferenceEngine {
namespace details {

CNNNetwork cloneNetwork(const CNNNetwork& network) {
    OV_ITT_SCOPED_TASK(ov::itt::domains::IE, "cloneNetwork");

    if (network.getFunction()) {
        IE_SUPPRESS_DEPRECATED_START
        return CNNNetwork(std::make_shared<details::CNNNetworkNGraphImpl>(network));
        IE_SUPPRESS_DEPRECATED_END
    }

    IE_THROW() << "InferenceEngine::details::cloneNetwork requires ngraph-based `network` object to clone";
}

// Returns tuple with new params and results
std::tuple<std::vector<std::shared_ptr<const ov::Node>>, std::vector<std::shared_ptr<const ov::Node>>>
CopyInputsOutputs(const std::shared_ptr<const ov::Model>& function,
                  const ConstInputsDataMap& inputsInfo,
                  const ConstOutputsDataMap& outputsInfo) {
    OPENVINO_ASSERT(function != nullptr);

    std::vector<std::shared_ptr<const ov::Node>> const_params;
    std::vector<std::shared_ptr<const ov::Node>> const_results;

    bool add_operation_names = false;
    const auto& rt_info = function->get_rt_info();
    const auto it = rt_info.find("version");
    if (it != rt_info.end()) {
        const int64_t ir_version = it->second.as<int64_t>();
        // here we decide whether we need to add operation_names as tensor names for
        // getInputs / getOutputs. Since these functions are designed to be used in new API only
        // always need to add operation names for IR v10
        add_operation_names = ir_version == 10;
    }

    OPENVINO_ASSERT(inputsInfo.size() == function->get_parameters().size());
    OPENVINO_ASSERT(outputsInfo.size() == function->get_output_size());

    for (const auto& param : function->get_parameters()) {
        auto new_param = ov::as_type_ptr<ov::op::v0::Parameter>(param->copy_with_new_inputs({}));
        new_param->set_friendly_name(param->get_friendly_name());
        if (add_operation_names)
            new_param->output(0).get_tensor().add_names({new_param->get_friendly_name()});
        new_param->set_layout(param->get_layout());
        // WA: use CNNNetwork's precisions since plugins sometimes override their precisions
        // after transformation pipeline is run
        new_param->set_element_type(
            InferenceEngine::details::convertPrecision(inputsInfo.at(new_param->get_friendly_name())->getPrecision()));
        new_param->validate_and_infer_types();
        const_params.emplace_back(new_param);
    }
    for (const auto& result : function->get_results()) {
        auto fake_param = std::make_shared<ov::op::v0::Parameter>(result->get_output_element_type(0),
                                                                  result->get_output_partial_shape(0));
        const std::string param_name = ngraph::op::util::create_ie_output_name(result->input_value(0));
        fake_param->set_friendly_name(param_name);
        fake_param->set_element_type(
            InferenceEngine::details::convertPrecision(outputsInfo.at(param_name)->getPrecision()));
        fake_param->validate_and_infer_types();
        auto new_result = result->copy_with_new_inputs({fake_param});
        new_result->set_friendly_name(result->get_friendly_name());
        if (add_operation_names) {
            new_result->output(0).get_tensor().add_names({fake_param->get_friendly_name()});
        }
        auto r = std::dynamic_pointer_cast<ov::op::v0::Result>(new_result);
        OPENVINO_ASSERT(r, "Internal error. SetNetworkInfo failure casting output copy to Result");
        r->set_layout(result->get_layout());
        const_results.emplace_back(new_result);
    }
    return std::make_tuple(const_params, const_results);
}

}  // namespace details
}  // namespace InferenceEngine
