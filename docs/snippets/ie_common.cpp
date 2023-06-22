// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef IN_OV_COMPONENT
#    define IN_OV_COMPONENT
#    define WAS_OV_LIBRARY_DEFINED
#endif

#include <ie_core.hpp>

#ifdef WAS_OV_LIBRARY_DEFINED
#    undef IN_OV_COMPONENT
#    undef WAS_OV_LIBRARY_DEFINED
#endif

int main() {
    //! [ie:create_core]
    InferenceEngine::Core core;
    //! [ie:create_core]

    //! [ie:read_model]
    InferenceEngine::CNNNetwork network = core.ReadNetwork("model.xml");
    //! [ie:read_model]

    InferenceEngine::InputsDataMap inputs = network.getInputsInfo();
    InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();

    //! [ie:compile_model]
    InferenceEngine::ExecutableNetwork exec_network = core.LoadNetwork(network, "CPU");
    //! [ie:compile_model]

    //! [ie:create_infer_request]
    InferenceEngine::InferRequest infer_request = exec_network.CreateInferRequest();
    //! [ie:create_infer_request]

    //! [ie:get_input_tensor]
    InferenceEngine::Blob::Ptr input_blob1 = infer_request.GetBlob(inputs.begin()->first);
    // fill first blob
    InferenceEngine::MemoryBlob::Ptr minput1 = InferenceEngine::as<InferenceEngine::MemoryBlob>(input_blob1);
    if (minput1) {
        // locked memory holder should be alive all time while access to its
        // buffer happens
        auto minputHolder = minput1->wmap();
        // Original I64 precision was converted to I32
        auto data = minputHolder.as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::I32>::value_type*>();
        // Fill data ...
    }

    InferenceEngine::Blob::Ptr input_blob2 = infer_request.GetBlob("data2");
    // fill second blob
    InferenceEngine::MemoryBlob::Ptr minput2 = InferenceEngine::as<InferenceEngine::MemoryBlob>(input_blob2);
    if (minput2) {
        // locked memory holder should be alive all time while access to its
        // buffer happens
        auto minputHolder = minput2->wmap();
        // Original I64 precision was converted to I32
        auto data = minputHolder.as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::I32>::value_type*>();
        // Fill data ...
    }
    //! [ie:get_input_tensor]

    //! [ie:inference]
    infer_request.Infer();
    //! [ie:inference]

    //! [ie:start_async_and_wait]
    // NOTE: For demonstration purposes we are trying to set callback
    // which restarts inference inside one more time, so two inferences happen here

    // Start inference without blocking current thread
    auto restart_once = true;
    infer_request.SetCompletionCallback<std::function<void(InferenceEngine::InferRequest, InferenceEngine::StatusCode)>>(
        [&, restart_once](InferenceEngine::InferRequest request, InferenceEngine::StatusCode status) mutable {
            if (status != InferenceEngine::OK) {
                // Process error code
            } else {
                // Extract inference result
                InferenceEngine::Blob::Ptr output_blob = request.GetBlob(outputs.begin()->first);
                // Restart inference if needed
                if (restart_once) {
                    request.StartAsync();
                    restart_once = false;
                }
            }
        });
    infer_request.StartAsync();
    // Get inference status immediately
    InferenceEngine::StatusCode status = infer_request.Wait(InferenceEngine::InferRequest::STATUS_ONLY);
    // Wait for 1 milisecond
    status = infer_request.Wait(1);
    // Wait for inference completion
    infer_request.Wait(InferenceEngine::InferRequest::RESULT_READY);
    //! [ie:start_async_and_wait]

    //! [ie:get_output_tensor]
    InferenceEngine::Blob::Ptr output_blob = infer_request.GetBlob(outputs.begin()->first);
    InferenceEngine::MemoryBlob::Ptr moutput = InferenceEngine::as<InferenceEngine::MemoryBlob>(output_blob);
    if (moutput) {
        // locked memory holder should be alive all time while access to its
        // buffer happens
        auto minputHolder = moutput->rmap();
        // Original I64 precision was converted to I32
        auto data =
            minputHolder.as<const InferenceEngine::PrecisionTrait<InferenceEngine::Precision::I32>::value_type*>();
        // process output data
    }
    //! [ie:get_output_tensor]
    //! [ie:load_old_extension]
    core.AddExtension(std::make_shared<InferenceEngine::Extension>("path_to_extension_library.so"));
    //! [ie:load_old_extension]
    (void)status;
    return 0;
}
