// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "common_test_utils/common_utils.hpp"

namespace BehaviorTestsDefinitions {

using SetBlobOfKindConfig = std::remove_reference<decltype(((LayerTestsUtils::LayerTestsCommon*)0)->GetConfiguration())>::type;

using SetBlobOfKindParams = std::tuple<FuncTestUtils::BlobKind, // The kind of blob
                                       std::string,             // Device name
                                       SetBlobOfKindConfig>;    // configuration

class SetBlobOfKindTest : public testing::WithParamInterface<SetBlobOfKindParams>, virtual public LayerTestsUtils::LayerTestsCommon {
public:
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;
    void Run() override;
    static std::string getTestCaseName(testing::TestParamInfo<SetBlobOfKindParams> obj);
    void ExpectSetBlobThrow();

protected:
    void SetUp() override;

private:
    FuncTestUtils::BlobKind blobKind;
};

} // namespace BehaviorTestsDefinitions
