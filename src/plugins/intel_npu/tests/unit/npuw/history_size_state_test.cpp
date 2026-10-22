// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>

#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"
#include "variable_state.hpp"

namespace {

ov::SoPtr<ov::ITensor> make_history(const ov::Shape& shape, int64_t value) {
    ov::Tensor tensor(ov::element::i64, shape);
    std::fill_n(tensor.data<int64_t>(), tensor.get_size(), value);
    return ov::get_tensor_impl(tensor);
}

TEST(HistorySizeStateTest, NameAndInitialValue) {
    ov::npuw::HistorySizeState state;
    EXPECT_EQ(state.get_name(), "npuw_history_size");
    EXPECT_EQ(state.get(), ov::npuw::HistorySizeState::kUnset);
    EXPECT_EQ(state.get_state()->data<int64_t>()[0], -1);
}

TEST(HistorySizeStateTest, SetStateUpdatesHistory) {
    ov::npuw::HistorySizeState state;
    state.set_state(make_history(ov::Shape{1}, 128));
    EXPECT_EQ(state.get(), 128);
    EXPECT_EQ(state.get_state()->data<int64_t>()[0], 128);
}

TEST(HistorySizeStateTest, InternalSetterIsVisibleThroughGetState) {
    ov::npuw::HistorySizeState state;
    state.set(512);
    EXPECT_EQ(state.get_state()->data<int64_t>()[0], 512);
}

TEST(HistorySizeStateTest, ResetMarksHistoryUnset) {
    ov::npuw::HistorySizeState state;
    state.set(64);
    state.reset();
    EXPECT_EQ(state.get(), ov::npuw::HistorySizeState::kUnset);
}

// The caller's tensor must not be aliased: later changes to it, or to the tensor returned
// by get_state(), must not alter the history.
TEST(HistorySizeStateTest, DoesNotAliasCallerOrReturnedTensors) {
    ov::npuw::HistorySizeState state;

    auto input = make_history(ov::Shape{1}, 32);
    state.set_state(input);
    input->data<int64_t>()[0] = 999;
    EXPECT_EQ(state.get(), 32);

    state.get_state()->data<int64_t>()[0] = 999;
    EXPECT_EQ(state.get(), 32);
}

TEST(HistorySizeStateTest, SetStateRejectsInvalidTensors) {
    ov::npuw::HistorySizeState state;
    state.set(7);

    ov::Tensor wrong_type(ov::element::i32, ov::Shape{1});
    EXPECT_ANY_THROW(state.set_state(ov::get_tensor_impl(wrong_type)));
    EXPECT_ANY_THROW(state.set_state(make_history(ov::Shape{2}, 1)));
    EXPECT_ANY_THROW(state.set_state(make_history(ov::Shape{1}, -1)));
    EXPECT_ANY_THROW(state.set_state(ov::SoPtr<ov::ITensor>{}));

    EXPECT_EQ(state.get(), 7) << "a rejected set_state must leave the history untouched";
}

}  // namespace
