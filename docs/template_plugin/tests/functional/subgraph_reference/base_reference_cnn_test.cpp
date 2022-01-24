// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "base_reference_cnn_test.hpp"
#include "functional_test_utils/ov_plugin_cache.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

#include <gtest/gtest.h>

#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/allocator.hpp"
#include "openvino/runtime/tensor.hpp"
#include "transformations/utils/utils.hpp"
#include "ie_ngraph_utils.hpp"

using namespace ov;

namespace reference_tests {

ReferenceCNNTest::ReferenceCNNTest(): targetDevice("TEMPLATE") {
    core = test::utils::PluginCache::get().core(targetDevice);
    legacy_core = PluginCache::get().ie(targetDevice);
}

void ReferenceCNNTest::Exec() {
    LoadNetwork();
    LoadNetworkLegacy();

    if (legacy_input_blobs.empty() && inputData.empty()) {
        FillInputs();
    }
    Infer();
    InferLegacy();

    Validate();
}

void ReferenceCNNTest::LoadNetwork() {
    executableNetwork = core->compile_model(function, targetDevice);
}

void ReferenceCNNTest::LoadNetworkLegacy() {
    auto inputInfo = legacy_network.getInputsInfo();
    auto outputInfo = legacy_network.getOutputsInfo();
    for (const auto& param : function->get_parameters()) {
        inputInfo[param->get_friendly_name()]->setPrecision(InferenceEngine::details::convertPrecision(param->get_element_type()));
    }
    for (const auto& result : function->get_results()) {
        outputInfo[ngraph::op::util::create_ie_output_name(result->input_value(0))]->setPrecision(
                InferenceEngine::details::convertPrecision(result->get_element_type()));
    }
    legacy_exec_network = legacy_core->LoadNetwork(legacy_network, targetDevice);
}

void ReferenceCNNTest::FillInputs() {
    const auto& params = function->get_parameters();
    std::default_random_engine random(0); // hard-coded seed to make test results predictable
    std::uniform_int_distribution<int> distrib(0, 255);
    for (const auto& param : params) {
        auto elem_count = shape_size(param->get_output_tensor(0).get_shape());
        InferenceEngine::TensorDesc d(InferenceEngine::Precision::FP32, param->get_output_tensor(0).get_shape(), InferenceEngine::Layout::NCHW);
        auto blob = make_blob_with_precision(d);
        blob->allocate();

        auto mBlob = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
        auto mBlobHolder = mBlob->wmap();
        auto buf = mBlobHolder.as<float*>();
        ASSERT_EQ(mBlob->size(), elem_count);

        ov::Tensor ov_blob;
        ov_blob = ov::Tensor(param->get_element_type(), param->get_shape());
        auto ov_buf = static_cast<float*>(ov_blob.data());

        for (size_t j = 0; j < elem_count; j++) {
            auto v = distrib(random);
            buf[j] = static_cast<float>(v);
            ov_buf[j] = static_cast<float>(v);
        }
        legacy_input_blobs[param->get_friendly_name()] = blob;
        inputData.push_back(ov_blob);
    }
}

void ReferenceCNNTest::Infer() {
    inferRequest = executableNetwork.create_infer_request();
    const auto& functionParams = function->get_parameters();

    for (size_t i = 0; i < functionParams.size(); ++i) {
        const auto& param = functionParams[i];
        inferRequest.set_tensor(param->get_friendly_name(), inputData[i]);
    }
    inferRequest.infer();
}

void ReferenceCNNTest::InferLegacy() {
    legacy_infer_request = legacy_exec_network.CreateInferRequest();
    legacy_infer_request.SetInput(legacy_input_blobs);
    legacy_infer_request.Infer();
}


void ReferenceCNNTest::Validate() {
    for (const auto& result : function->get_results()) {
        auto name = ngraph::op::util::create_ie_output_name(result->input_value(0));
        outputs_ov20.emplace_back(inferRequest.get_tensor(name));
        auto outBlob = legacy_infer_request.GetBlob(name);
        auto outMem = outBlob->buffer();
        void* outData = outMem.as<void*>();
        outputs_legacy.emplace_back(element::f32, result->get_shape(), outData);
    }
    for (size_t i = 0; i < outputs_legacy.size(); i++) {
        CommonReferenceTest::ValidateBlobs(outputs_legacy[i], outputs_ov20[i], threshold, abs_threshold);
    }
}

}  // namespace reference_tests
