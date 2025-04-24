// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
using extractImagePatchesTuple = typename std::tuple<
        std::vector<InputShape>,   // input shape
        std::vector<size_t>,       // kernel size
        std::vector<size_t>,       // strides
        std::vector<size_t>,       // rates
        ov::op::PadType,           // pad type
        ov::element::Type,         // model type
        std::string>;              // device name

class ExtractImagePatchesTest : public testing::WithParamInterface<extractImagePatchesTuple>,
                              virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<extractImagePatchesTuple> &obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
