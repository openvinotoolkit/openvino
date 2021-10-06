// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/shape.hpp"
#include "openvino/runtime/allocator.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/core/type/element_type.hpp"

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
    void ValidateBlobs(const ov::runtime::Tensor& refBlob, const ov::runtime::Tensor& outBlob);

protected:
    const std::string targetDevice;
    std::shared_ptr<ov::runtime::Core> core;
    std::shared_ptr<ov::Function> function;

    ov::runtime::ExecutableNetwork executableNetwork;
    ov::runtime::InferRequest inferRequest;
    std::vector<ov::runtime::Tensor> inputData;
    std::vector<ov::runtime::Tensor> refOutData;
    float threshold = 1e-2f;    // Relative diff
    float abs_threshold = -1.f; // Absolute diff (not used when negative)
};

template <class T>
ov::runtime::Tensor CreateTensor(const ov::element::Type& element_type,
                               const std::vector<T>& values,
                               size_t size = 0) {
    size_t real_size = size ? size : values.size() * sizeof(T) / element_type.size();
    ov::runtime::Tensor tensor { element_type, {real_size} };
    std::memcpy(tensor.data(), values.data(), std::min(real_size * element_type.size(), sizeof(T) * values.size()));

    return tensor;
}

// Create blob with correct input shape (not 1-dimensional). Will be used in tests with dynamic input shapes
template <class T>
ov::runtime::Tensor CreateTensor(const ov::Shape& shape,
                               const ov::element::Type& element_type,
                               const std::vector<T>& values) {
    ov::runtime::Tensor tensor { element_type, shape };
    std::memcpy(tensor.data(), values.data(), sizeof(T) * values.size());

    return tensor;
}

///
/// Class which should help to build data for single input
///
struct Tensor {
    Tensor() = default;

    Tensor(const ov::Shape& shape, ov::element::Type type, const ov::runtime::Tensor& data): shape {shape}, type {type}, data {data} {}

    template <typename T>
    Tensor(const ov::Shape& shape, ov::element::Type type, const std::vector<T>& data_elements)
        : Tensor {shape, type, CreateTensor(type, data_elements)} {}

    // Temporary constructor to create blob with passed input shape (not 1-dimensional)
    template <typename T>
    Tensor(ov::element::Type type, const ov::Shape& shape, const std::vector<T>& data_elements)
            : Tensor {shape, type, CreateTensor(shape, type, data_elements)} {}

    ov::Shape shape;
    ov::element::Type type;
    ov::runtime::Tensor data;
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
