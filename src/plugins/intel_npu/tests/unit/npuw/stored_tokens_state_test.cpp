// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>

// Forward-declare LLMInferRequest so the friend declaration in StoredTokensState resolves.
namespace ov { namespace npuw { class LLMInferRequest; } }

#include "llm_stored_tokens_state.hpp"

namespace {
int64_t read_stored_tokens(const ov::npuw::StoredTokensState& state) {
    auto tensor = state.get_state();
    return tensor->data<int64_t>()[0];
}

TEST(StoredTokensStateTest, InitialValueIsZero) {
    ov::npuw::StoredTokensState state;
    EXPECT_EQ(read_stored_tokens(state), 0);
}

TEST(StoredTokensStateTest, StateNameIsCorrect) {
    ov::npuw::StoredTokensState state;
    EXPECT_EQ(state.get_name(), "npuw_stored_tokens_state");
}

TEST(StoredTokensStateTest, ResetState) {
    ov::npuw::StoredTokensState state;
    state.reset();
    EXPECT_EQ(read_stored_tokens(state), 0);
}

// --- set_state(...) should not be called externally ---
TEST(StoredTokensStateTest, SetStateThrows) {
    ov::npuw::StoredTokensState state;
    auto tensor = ov::Tensor(ov::element::i64, ov::Shape{1});
    EXPECT_ANY_THROW(state.set_state(ov::get_tensor_impl(tensor)));
}

TEST(StoredTokensStateTest, GetStateReturnsTensor) {
    ov::npuw::StoredTokensState state;
    auto tensor = state.get_state();
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->get_element_type(), ov::element::i64);
    EXPECT_EQ(tensor->get_shape(), ov::Shape{1});
}

// --- get_state() returns a copy, so external mutation does not affect internal state ---
TEST(StoredTokensStateTest, GetStateReturnsCopyNotReference) {
    ov::npuw::StoredTokensState state;
    EXPECT_EQ(read_stored_tokens(state), 0);

    auto tensor = state.get_state();
    tensor->data<int64_t>()[0] = 42;

    EXPECT_EQ(read_stored_tokens(state), 0);
}
}  // namespace
