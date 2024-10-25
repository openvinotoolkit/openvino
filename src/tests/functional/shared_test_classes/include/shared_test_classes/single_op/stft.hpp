// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>

#include "common_test_utils/test_enums.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
typedef std::tuple<std::vector<InputShape>,
                   int64_t,            // frame size value
                   int64_t,            // frame step value
                   bool,               // transpose_frames
                   ov::element::Type,  // data type
                   ov::element::Type,  // size/step type
                   ov::test::utils::InputLayerType,
                   ov::test::TargetDevice>
    STFTParams;

class STFTLayerTest : public testing::WithParamInterface<STFTParams>, virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<STFTParams>& obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
