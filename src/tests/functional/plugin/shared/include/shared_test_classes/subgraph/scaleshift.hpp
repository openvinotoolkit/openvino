// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
using ScaleShiftParamsTuple = typename std::tuple<
        std::vector<ov::Shape>,  //input shapes
        ov::element::Type,       //Model type
        std::string,             //Device name
        std::vector<float>,      //scale
        std::vector<float>>;     //shift

class ScaleShiftLayerTest:
        public testing::WithParamInterface<ScaleShiftParamsTuple>,
        virtual public ov::test::SubgraphBaseStaticTest{
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ScaleShiftParamsTuple> &obj);
protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
