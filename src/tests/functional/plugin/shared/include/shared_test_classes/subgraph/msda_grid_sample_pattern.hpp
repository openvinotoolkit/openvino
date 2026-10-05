// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "common_test_utils/test_constants.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

// GridSample based multi-scale deformable attention with a non-default
// geometry (3 levels x 2 points, 2 heads) fused into the internal MSDA op by
// the GPU plugin pipeline.
class MSDAGridSamplePattern : public ov::test::SubgraphBaseTest {
protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    size_t expected_msda_count() const {
        return 1;
    }
};

}  // namespace test
}  // namespace ov
