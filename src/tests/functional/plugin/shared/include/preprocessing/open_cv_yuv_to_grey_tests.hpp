// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/preprocess/resize_algorithm.hpp>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace preprocess {

using TParams = std::tuple<std::string>;

class PreprocessingYUV2GreyTest : public testing::WithParamInterface<TParams>, public test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<TParams>& obj);

protected:
    void SetUp() override;
    void run() override;
    ov::TensorVector calculate_refs() override;

    size_t get_full_height();
    void test_model_color_conversion(ColorFormat from, ColorFormat to);

    ov::TensorVector ref_out_data;
    size_t width, height;
    int b_step;
};

}  // namespace preprocess
}  // namespace ov
