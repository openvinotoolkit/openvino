// Copyright (C) 2019-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/single_layer/psroi_pooling.hpp"
#include "shared_test_classes/single_layer/roi_pooling.hpp"
#include "shared_test_classes/single_layer/roi_align.hpp"

namespace LayerTestsDefinitions {
class ReadIRTest : public testing::WithParamInterface<std::tuple<std::string, std::string>>,
                           public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::tuple<std::string, std::string>> &obj);

protected:
    void SetUp() override;
    void Infer() override;

private:
    std::string pathToModel;
};
} // namespace LayerTestsDefinitions
