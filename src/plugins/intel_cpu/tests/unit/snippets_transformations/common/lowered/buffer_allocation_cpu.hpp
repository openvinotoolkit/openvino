// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "lowered/pass/buffer_allocation.hpp"
#include "transformations/snippets/common/shape_inference.hpp"

namespace ov {
namespace test {
namespace snippets {

class BufferAllocationCPUTest : public BufferAllocationTest {
protected:
    std::shared_ptr<ov::snippets::IShapeInferSnippetsFactory> GetShapeInferFactory() const override {
        return std::make_shared<ov::snippets::CPUShapeInferSnippetsFactory>();
    }
};

namespace BufferAllocationCPUTest_Instances {

inline const std::vector<ov::PartialShape> static_shapes = {
    {1, 12, 1024, 1024},
    {1, 128, 12, 1024},
    {1, 12, 128, 256},
};

inline const std::vector<ov::PartialShape> dynamic_shapes = {
    {-1, -1, -1, -1},
    {-1, -1, -1, -1},
    {-1, -1, -1, -1},
};

}  // namespace BufferAllocationCPUTest_Instances

}  // namespace snippets
}  // namespace test
}  // namespace ov
