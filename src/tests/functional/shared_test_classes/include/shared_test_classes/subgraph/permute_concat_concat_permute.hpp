// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "ngraph_functions/builders.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

namespace SubgraphTestsDefinitions {
typedef std::tuple<std::vector<size_t>,         // input shapes and permute shapes
                   InferenceEngine::Precision,  // Network precision
                   std::string                  // Device name
                   >
    PermuteConcatConcatPermuteTuple;

class PermuteConcatConcatPermute : public testing::WithParamInterface<PermuteConcatConcatPermuteTuple>,
                                   virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<PermuteConcatConcatPermuteTuple>& obj);

protected:
    void SetUp() override;
    void Validate() override;
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& inputInfo) const override;

    static std::shared_ptr<ngraph::opset9::Constant> CreateConst(const std::vector<size_t>& input_shape,
                                                                 const ::ngraph::element::Type& precision,
                                                                 bool use_1_as_first_dimension);
    template <typename T>
    static void CompareValues(const T& expectedValue, const T& value, std::size_t index, float threshold);
    template <typename T>
    static void CompareBuffers(const T* expexctedData, const T* data, std::size_t size, float threshold);

    int32_t range_{};
    int32_t start_{0};
    int32_t step_{1};
};

template <typename T>
inline void PermuteConcatConcatPermute::CompareValues(const T& expectedValue,
                                                      const T& value,
                                                      std::size_t index,
                                                      float threshold) {
    auto result = std::abs(expectedValue - value);
    if (expectedValue == 0.0f && value != 0.0f) {
        IE_THROW() << "Relative comparison of values expected exact 0.0f and actual: " << std::to_string(value)
                   << " at index " << index << " failed";
    } else if (result > threshold) {
        IE_THROW() << "Relative comparison of values expected: " << std::to_string(expectedValue)
                   << " and actual: " << std::to_string(value) << " at index " << index << " with threshold "
                   << threshold << " failed";
    }
}

template <typename T>
inline void PermuteConcatConcatPermute::CompareBuffers(const T* expexctedData,
                                                       const T* data,
                                                       std::size_t size,
                                                       float threshold) {
    for (std::size_t i = 0; i < size; ++i) {
        CompareValues(expexctedData[i], data[i], i, threshold);
    }
}
}  // namespace SubgraphTestsDefinitions
