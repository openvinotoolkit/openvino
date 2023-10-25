// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SHARED_TEST_CLASSES_MEMORY_FQ_CONCAT_PRELU_H
#define SHARED_TEST_CLASSES_MEMORY_FQ_CONCAT_PRELU_H

#include <tuple>
#include <vector>
#include <string>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"

namespace SubgraphTestsDefinitions {

typedef std::tuple<
        std::vector<std::vector<size_t>>,   //input shapes
        InferenceEngine::Precision,         //Network precision
        std::string,                        //Device name
        std::map<std::string, std::string>, //Configuration
        std::tuple<
            std::vector<int64_t>,
            std::vector<int64_t>,
            std::vector<int64_t>,
            std::vector<int64_t>,
            std::vector<int64_t>>,          // StridedSlice
        std::tuple<
            std::size_t,
            std::vector<size_t>,
            std::vector<float>,
            std::vector<float>,
            std::vector<float>,
            std::vector<float>>             // FakeQuantize
> MemoryFqConcatPreluTuple;

class MemoryFqConcatPrelu : public testing::WithParamInterface<MemoryFqConcatPreluTuple>,
    virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MemoryFqConcatPreluTuple> &obj);
    void Run() override;

protected:
    void SetUp() override;
}; // class MemoryFqConcatPrelu

}  // namespace SubgraphTestsDefinitions

#endif // SHARED_TEST_CLASSES_MEMORY_FQ_CONCAT_PRELU_H
