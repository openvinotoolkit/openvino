// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_common.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include <ie_core.hpp>

namespace SubgraphTestsDefinitions {
typedef std::tuple<
    std::string,                        // Target device name
    InferenceEngine::Precision,         // Network precision
    size_t,                             // Input size
    std::map<std::string, std::string>  // Configuration
> multipleInputScaleParams;

class MultipleInputScaleTest : public LayerTestsUtils::LayerTestsCommon,
    public testing::WithParamInterface<multipleInputScaleParams> {
protected:
    void SetUp() override;
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;
    float inputDataMin = -0.2f;
    float range = 0.4f;
    float inputDataResolution = 0.01f;
    int32_t  seed = 1;
public:
    static std::string getTestCaseName(const testing::TestParamInfo<multipleInputScaleParams> &obj);
};
} // namespace SubgraphTestsDefinitions

