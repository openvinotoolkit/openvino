// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <inference_engine.hpp>
#include <string>
#include <vector>

using namespace InferenceEngine;

template<typename T>
std::string vec2str(const std::vector<T> &vec) {
    if (!vec.empty()) {
        std::ostringstream result;
        result << "(";
        std::copy(vec.begin(), vec.end() - 1, std::ostream_iterator<T>(result, "."));
        result << vec.back() << ")";
        return result.str();
    }
    return std::string("()");
}

int main(int argc, char* argv[]) {
    Core ie;
    CNNNetwork network = ie.ReadNetwork("/home/maximandronov/test_repo/openvino/models/BERT/fp32/bert-base-chinese-xnli-zh-fp32-onnx-0001.xml",
                                        "/home/maximandronov/test_repo/openvino/models/BERT/fp32/bert-base-chinese-xnli-zh-fp32-onnx-0001.bin");

    InferenceEngine::InputsDataMap inputsInfo = network.getInputsInfo();
    std::map<std::string, ov::PartialShape> shapes;
    for (const auto &in: inputsInfo) {
        std::cout << "INPUT: " << in.first << std::endl;
        shapes[in.first] = {1, -1};
    }
    network.reshape(shapes);

    // network.serialize("/home/maximandronov/test_repo/openvino/models/BERT/dump.xml");

    InferenceEngine::SizeVector initDims = {1, 32};
std::cout << "START LOAD NETWORK" << std::endl; 
    InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(network, "CPU");
std::cout << "END LOAD NETWORK" << std::endl; 
    InferenceEngine::InferRequest infer_request = exeNetwork.CreateInferRequest();

    const size_t inferNum = 5;
    for (size_t i = 0; i < inferNum; i++) {
        std::cout << "START INFER: " << i << std::endl;
        initDims[1] *= 2;
        for (const auto &in: inputsInfo) {
            auto blob = InferenceEngine::make_shared_blob<int32_t>((InferenceEngine::TensorDesc(InferenceEngine::Precision::I32, initDims, InferenceEngine::Layout::BLOCKED)));
            blob->allocate();
            infer_request.SetBlob(in.first, blob);
        }
        infer_request.Infer();
        std::cout << "END INFER: " << i << std::endl;

        for (const auto &out: network.getOutputsInfo()) {
            std::cout << "OUT: " << out.first << " " << vec2str(infer_request.GetBlob(out.first)->getTensorDesc().getDims()) << std::endl;
        }
    }

    return 0;
}
