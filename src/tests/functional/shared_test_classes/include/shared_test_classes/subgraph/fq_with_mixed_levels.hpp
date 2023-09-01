// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef FQ_WITH_MIXED_LEVELS_HPP
#define FQ_WITH_MIXED_LEVELS_HPP

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"

namespace SubgraphTestsDefinitions {

typedef std::tuple<
        InferenceEngine::Precision,         // Network Precision
        std::string,                        // Target Device
        std::map<std::string, std::string>  // Configuration
> FqWithMixedLevelsParams;

class FqWithMixedLevelsTest : public testing::WithParamInterface<FqWithMixedLevelsParams>,
                        virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<FqWithMixedLevelsParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace SubgraphTestsDefinitions

#endif // FQ_WITH_MIXED_LEVELS_HPP
