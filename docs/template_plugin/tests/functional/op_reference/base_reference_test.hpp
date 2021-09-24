// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_core.hpp>
#include <ie_ngraph_utils.hpp>
#include <ngraph/ngraph.hpp>
#include <ngraph/shape.hpp>
#include <ngraph/type/element_type.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

namespace reference_tests {

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

template <class T>
InferenceEngine::Blob::Ptr CreateBlob(const ov::PartialShape& partial_shape,
                                      const ngraph::element::Type& element_type,
                                      const std::vector<T>& values) {
    auto shape = partial_shape.get_shape();
    auto blob = make_blob_with_precision(
            InferenceEngine::TensorDesc(InferenceEngine::details::convertPrecision(element_type), shape,
                                        InferenceEngine::Layout::ANY));
    blob->allocate();
    InferenceEngine::MemoryBlob::Ptr minput = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
    IE_ASSERT(minput);
    auto minputHolder = minput->wmap();

    std::memcpy(minputHolder.as<void*>(), values.data(), sizeof(T) * values.size());

    return blob;
}

///
/// Class which should help to build data for single input
///
struct Tensor {
    Tensor() = default;

    Tensor(const ngraph::Shape& shape, ngraph::element::Type type, const InferenceEngine::Blob::Ptr& data): shape {shape}, type {type}, data {data} {}

    template <typename T>
    Tensor(const ngraph::Shape& shape, ngraph::element::Type type, const std::vector<T>& data_elements)
        : Tensor {shape, type, CreateBlob(type, data_elements)} {}

    // Use this constructor of dynamic network inputs
    template <typename T>
    Tensor(ngraph::element::Type type, const ngraph::Shape& shape, const std::vector<T>& data_elements)
            : Tensor {shape, type, CreateBlob(shape, type, data_elements)} {}

    ngraph::Shape shape;
    ngraph::element::Type type;
    InferenceEngine::Blob::Ptr data;
};

///
/// Class which should helps build test parameters.
///
/// e.g.:
/// struct Params {
///     Tensor i,o;
///     int mul;
/// };
/// struct TestParamsBuilder : ParamsBuilder<Params>
///     REFERENCE_TESTS_ADD_SET_PARAM(TestParamsBuilder, i);
///     REFERENCE_TESTS_ADD_SET_PARAM(TestParamsBuilder, o);
///     REFERENCE_TESTS_ADD_SET_PARAM(TestParamsBuilder, mul);
/// };
///
/// const Params p = TestParamsBuilder{}
///                  .i(Tensor{{0}, i32, {1}})
///                  .o(Tensor{{0}, i32, {1}})
///                  .mul(10);
template <typename Params>
class ParamsBuilder {
protected:
    Params params;

public:
    operator Params() const {
        return params;
    }
};
#define REFERENCE_TESTS_ADD_SET_PARAM(builder_type, param_to_set) \
    builder_type& param_to_set(decltype(params.param_to_set) t) { \
        params.param_to_set = std::move(t);                       \
        return *this;                                             \
    }

}  // namespace reference_tests
