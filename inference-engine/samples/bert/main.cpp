// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <inference_engine.hpp>
#include <string>
#include <vector>
#include <random>

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

template <typename T>
using uniformDistribution = typename std::conditional<
    std::is_floating_point<T>::value,
    std::uniform_real_distribution<T>,
    typename std::conditional<std::is_integral<T>::value, std::uniform_int_distribution<T>, void>::type>::type;

template <typename T, typename T2>
void fillBlobRandom(InferenceEngine::Blob::Ptr& inputBlob,
                    T rand_min = std::numeric_limits<uint8_t>::min(),
                    T rand_max = std::numeric_limits<uint8_t>::max()) {
    MemoryBlob::Ptr minput = as<MemoryBlob>(inputBlob);
    if (!minput) {
        IE_THROW() << "We expect inputBlob to be inherited from MemoryBlob in "
                      "fillBlobRandom, "
                   << "but by fact we were not able to cast inputBlob to MemoryBlob";
    }
    // locked memory holder should be alive all time while access to its buffer
    // happens
    auto minputHolder = minput->wmap();

    auto inputBlobData = minputHolder.as<T*>();
    std::mt19937 gen(0);
    uniformDistribution<T2> distribution(rand_min, rand_max);
    for (size_t i = 0; i < inputBlob->size(); i++) {
        inputBlobData[i] = static_cast<T>(distribution(gen));
    }
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

    // network.serialize("/home/maximandronov/test_repo/openvino/models/BERT/dynamic_fp32/model.xml");

    InferenceEngine::SizeVector initDims = {1, 16};
std::cout << "START LOAD NETWORK" << std::endl; 
    InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(network, "CPU");
std::cout << "END LOAD NETWORK" << std::endl; 
    InferenceEngine::InferRequest infer_request = exeNetwork.CreateInferRequest();

    const size_t inferNum = 5;
    for (size_t i = 0; i < inferNum; i++) {
        std::cout << "START INFER: " << i << std::endl;
        initDims[1] *= 2;
        for (const auto &in: inputsInfo) {
            InferenceEngine::Blob::Ptr blob = InferenceEngine::make_shared_blob<int32_t>((InferenceEngine::TensorDesc(InferenceEngine::Precision::I32, initDims, InferenceEngine::Layout::BLOCKED)));
            blob->allocate();
            fillBlobRandom<int32_t, int32_t>(blob);
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
