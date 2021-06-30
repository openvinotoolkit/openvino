// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_core.hpp>
#include <ie_ngraph_utils.hpp>
#include <ngraph/ngraph.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

class CommonReferenceTest {
public:
    CommonReferenceTest();

    void Exec();

    void LoadNetwork();

    void FillInputs();

    void Infer();

    void Validate();

private:
    void ValidateBlobs(const InferenceEngine::Blob::Ptr& refBlob, const InferenceEngine::Blob::Ptr& outBlob);

protected:
    const std::string targetDevice;
    std::shared_ptr<InferenceEngine::Core> core;
    std::shared_ptr<ngraph::Function> function;

    InferenceEngine::ExecutableNetwork executableNetwork;
    InferenceEngine::InferRequest inferRequest;
    std::vector<InferenceEngine::Blob::Ptr> inputData;
    std::vector<InferenceEngine::Blob::Ptr> refOutData;
    float threshold = 1e-2f;
};

template <class T>
InferenceEngine::Blob::Ptr CreateBlob(const ngraph::element::Type& element_type, const std::vector<T>& values, size_t size = 0) {
    size_t real_size = size ? size : values.size() * sizeof(T) / element_type.size();
    auto blob = make_blob_with_precision(
        InferenceEngine::TensorDesc(InferenceEngine::details::convertPrecision(element_type), {real_size}, InferenceEngine::Layout::C));
    blob->allocate();
    InferenceEngine::MemoryBlob::Ptr minput = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
    IE_ASSERT(minput);
    auto minputHolder = minput->wmap();

    std::memcpy(minputHolder.as<void*>(), values.data(), std::min(real_size * element_type.size(), sizeof(T) * values.size()));

    return blob;
}

