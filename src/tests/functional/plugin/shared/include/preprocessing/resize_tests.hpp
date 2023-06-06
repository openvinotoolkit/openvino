// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/preprocess/resize_algorithm.hpp>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace preprocess {

using ResizeTestsParams = std::tuple<std::string, ResizeAlgorithm, std::vector<float>>;

class PreprocessingResizeTests : public testing::WithParamInterface<ResizeTestsParams>,
                                 virtual public test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ResizeTestsParams>& obj);
    // void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override;
    // void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override;

protected:
    void SetUp() override;
    void run() override;
    std::vector<ov::Tensor> calculate_refs() override;
};

}  // namespace preprocess
}  // namespace ov
