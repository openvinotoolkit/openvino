// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <common_test_utils/ov_test_utils.hpp>

#include "snippets/op/brgemm.hpp"
#include "snippets/lowered/pass/pass.hpp"

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<
    bool,   // Optimized pipeline
    bool,   // With SplitLoops opt
    size_t, // Expected Buffer size in bytes
    size_t  // Expected unique Buffer IDs count
> BufferAllocationParams;

class BufferAllocationTest : public testing::TestWithParam<BufferAllocationParams> {
public:
    using VectorDims = ov::snippets::VectorDims;
    static std::string getTestCaseName(testing::TestParamInfo<BufferAllocationParams> obj);

protected:
    void SetUp() override;
    void ApplyTransformations(bool is_optimized, bool with_split_loops);
    void Validate();

    virtual std::shared_ptr<ov::Model> GetModel() const = 0;

    static void MarkOp(const std::shared_ptr<ov::Node>& node, const std::vector<size_t>& subtensor);

    size_t m_buffer_scratchpad = 0;
    ov::snippets::lowered::LinearIR m_linear_ir;

    size_t m_expected_size = 0;
    size_t m_expected_count = 0;

    size_t m_loop_depth = 2;
    size_t m_vector_size = 16;
};

class EltwiseBufferAllocationTest : public BufferAllocationTest {
protected:
    std::shared_ptr<ov::Model> GetModel() const override;
};

class MHABufferAllocationTest : public BufferAllocationTest {
protected:
    std::shared_ptr<ov::Model> GetModel() const override;

    static void MarkBrgemm(const std::shared_ptr<ov::snippets::op::Brgemm>& node, const std::vector<size_t>& subtensor);
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
