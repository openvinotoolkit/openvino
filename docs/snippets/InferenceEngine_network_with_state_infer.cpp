// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include <ie_core.hpp>

using namespace InferenceEngine;

int main(int argc, char *argv[]) {
    try {
        // --------------------------- 1. Load inference engine -------------------------------------
        std::cout << "Loading Inference Engine" << std::endl;
        Core ie;

        // 2. Read a model in OpenVINO Intermediate Representation (.xml and .bin files) or ONNX (.onnx file) format
        std::cout << "Loading network files" << std::endl;
        CNNNetwork network;
        network = ie.ReadNetwork(std::string("c:\\work\\git\\github_dldt3\\openvino\\model-optimizer\\summator.xml"));
        network.setBatchSize(1);

        // 3. Load network to CPU
        ExecutableNetwork executableNet = ie.LoadNetwork(network, "CPU");
        // 4. Create Infer Request
        InferRequest inferRequest = executableNet.CreateInferRequest();

        // 5. Prepare inputs
        ConstInputsDataMap cInputInfo = executableNet.GetInputsInfo();
        std::vector<Blob::Ptr> ptrInputBlobs;
        for (const auto& input : cInputInfo) {
            ptrInputBlobs.push_back(inferRequest.GetBlob(input.first));
        }
        InputsDataMap inputInfo;
        inputInfo = network.getInputsInfo();
        for (auto &item : inputInfo) {
            Precision inputPrecision = Precision::FP32;
            item.second->setPrecision(inputPrecision);
        }

        // 6. Prepare outputs
        std::vector<Blob::Ptr> ptrOutputBlobs;
        ConstOutputsDataMap cOutputInfo = executableNet.GetOutputsInfo();
        for (const auto& output : cOutputInfo) {
            ptrOutputBlobs.push_back(inferRequest.GetBlob(output.first));
        }
        
        // 7. Initialize memory state before starting
        for (auto &&state : inferRequest.QueryState()) {
            state.Reset();
        }

        //! [part1]
        // input data
        std::vector<float> data = { 1,2,3,4,5,6};
        // infer the first utterance
        for (size_t next_input = 0; next_input < data.size()/2; next_input++) {
            MemoryBlob::Ptr minput = as<MemoryBlob>(ptrInputBlobs[0]);
            auto minputHolder = minput->wmap();

            std::memcpy(minputHolder.as<void *>(),
                &data[next_input],
                sizeof(float));

            inferRequest.Infer();
            // check states
            auto states = inferRequest.QueryState();
            if (states.empty()) {
                throw std::runtime_error("Queried states are empty");
            }
            auto mstate = as<MemoryBlob>(states[0].GetState());
            if (mstate == nullptr) {
                throw std::runtime_error("Can't cast state to MemoryBlob");
            }
            auto state_buf = mstate->rmap();
            float * state =state_buf.as<float*>(); 
            std::cout << state[0] << "\n";
        }

        // resetting state between utterances
        std::cout<<"Reset state\n";
        for (auto &&state : inferRequest.QueryState()) {
            state.Reset();
        }

        // infer the second utterance
        for (size_t next_input = data.size()/2; next_input < data.size(); next_input++) {
            MemoryBlob::Ptr minput = as<MemoryBlob>(ptrInputBlobs[0]);
            auto minputHolder = minput->wmap();

            std::memcpy(minputHolder.as<void *>(),
                &data[next_input],
                sizeof(float));

            inferRequest.Infer();
            // check states
            auto states = inferRequest.QueryState();
            auto mstate = as<MemoryBlob>(states[0].GetState());
            auto state_buf = mstate->rmap();
            float * state =state_buf.as<float*>(); 
            std::cout << state[0] << "\n";
      }
        //! [part1]
    }
    catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "Unknown/internal exception happened" << std::endl;
        return 1;
    }

    std::cerr << "Execution successful" << std::endl;
    return 0;
}
