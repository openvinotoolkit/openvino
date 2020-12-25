// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/proposal.hpp"

namespace BehaviorTestsDefinitions {

typedef std::tuple<
        LayerTestsDefinitions::proposalSpecificParams,
        std::vector<float>,
        std::string> proposalBehTestParamsSet;

class ProposalBehTest
        : public testing::WithParamInterface<proposalBehTestParamsSet>,
          virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<proposalBehTestParamsSet> obj);
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;

protected:
    void SetUp() override;
    void Validate() override {};

    const LayerTestsDefinitions::normalize_type normalize = true;
    const LayerTestsDefinitions::feat_stride_type feat_stride = 1;
    const LayerTestsDefinitions::box_size_scale_type box_size_scale = 2.0f;
    const LayerTestsDefinitions::box_coordinate_scale_type box_coordinate_scale = 2.0f;
};

}  // namespace BehaviorTestsDefinitions
