// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_sparse_attention.hpp"

#include <gtest/gtest.h>

#include <string>

#include "openvino/op/parameter.hpp"

using namespace ov;

namespace {
class KeyBlockSizeVisitor : public AttributeVisitor {
public:
    void on_adapter(const std::string&, ValueAccessor<void>&) override {}

    void on_adapter(const std::string& name, ValueAccessor<int64_t>& adapter) override {
        if (name == "key_block_size") {
            adapter.set(key_block_size);
        }
    }

    int64_t key_block_size = 8;
};
}  // namespace

TEST(attributes, paged_sparse_attention) {
    OutputVector inputs{std::make_shared<op::v0::Parameter>(element::f32, PartialShape{2, 32}),
                        std::make_shared<op::v0::Parameter>(element::f32, PartialShape{2, 16}),
                        std::make_shared<op::v0::Parameter>(element::f32, PartialShape{2, 16}),
                        std::make_shared<op::v0::Parameter>(element::f16, PartialShape{4, 2, 16, 8}),
                        std::make_shared<op::v0::Parameter>(element::f16, PartialShape{4, 2, 16, 8}),
                        std::make_shared<op::v0::Parameter>(element::i32, PartialShape{1}),
                        std::make_shared<op::v0::Parameter>(element::i32, PartialShape{2}),
                        std::make_shared<op::v0::Parameter>(element::i32, PartialShape{4}),
                        std::make_shared<op::v0::Parameter>(element::i32, PartialShape{2}),
                        std::make_shared<op::v0::Parameter>(element::i32, PartialShape{2, 1, 4}),
                        std::make_shared<op::v0::Parameter>(element::i32, PartialShape{2, 1}),
                        std::make_shared<op::v0::Parameter>(element::f32, PartialShape{}),
                        std::make_shared<op::v0::Parameter>(element::f16, PartialShape{4})};
    const auto op = std::make_shared<ov::op::internal::PagedSparseAttention>(inputs, 4);
    KeyBlockSizeVisitor visitor;
    EXPECT_TRUE(op->visit_attributes(visitor));
    const auto cloned = ov::as_type_ptr<ov::op::internal::PagedSparseAttention>(op->clone_with_new_inputs(inputs));

    EXPECT_EQ(op->get_key_block_size(), visitor.key_block_size);
    ASSERT_NE(cloned, nullptr);
    EXPECT_EQ(op->get_key_block_size(), cloned->get_key_block_size());
}
