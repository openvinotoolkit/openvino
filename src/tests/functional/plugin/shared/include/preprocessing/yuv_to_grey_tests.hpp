// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include "shared_test_classes/base/ov_subgraph.hpp"

using TParams = std::tuple<std::string>;

namespace ov {
namespace preprocess {
class PreprocessingYUV2GreyTest : public testing::WithParamInterface<TParams>, public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<TParams>& obj);

protected:
    void SetUp() override;
    void run() override;
    ov::TensorVector calculate_refs() override;

    size_t get_full_height();
    std::shared_ptr<ov::Model> build_test_model(const element::Type_t et, const Shape& shape);
    void set_test_model_color_conversion(ColorFormat from, ColorFormat to);

    ov::TensorVector ref_out_data;
    size_t width, height;
    int b_step;
};

}  // namespace preprocess
}  // namespace ov
