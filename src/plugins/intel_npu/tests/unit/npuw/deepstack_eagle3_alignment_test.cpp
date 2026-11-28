// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <array>

#include "llm_eagle3_extension.hpp"
#include "llm_infer_request.hpp"
#include "openvino/openvino.hpp"

namespace {

TEST(DeepStackAlignmentTest, PartialChunkUsesLeftPositionsAndContinuesVisualRows) {
    ov::Tensor source(ov::element::f32, ov::Shape{1, 3, 1});
    constexpr std::array<float, 3> source_values{10.f, 20.f, 30.f};
    std::copy(source_values.begin(), source_values.end(), source.data<float>());
    ov::Tensor mask(ov::element::i64, ov::Shape{1, 3});
    constexpr std::array<int64_t, 3> mask_values{1, 0, 1};
    std::copy(mask_values.begin(), mask_values.end(), mask.data<int64_t>());
    ov::Tensor destination(ov::element::f32, ov::Shape{1, 4, 1});

    const auto scattered = ov::npuw::util::scatter_deepstack_visual_embeds(ov::get_tensor_impl(source),
                                                                           ov::get_tensor_impl(mask),
                                                                           ov::get_tensor_impl(destination),
                                                                           1u);

    EXPECT_EQ(scattered, 2u);
    EXPECT_FLOAT_EQ(destination.data<float>()[0], 20.f);
    EXPECT_FLOAT_EQ(destination.data<float>()[1], 0.f);
    EXPECT_FLOAT_EQ(destination.data<float>()[2], 30.f);
    EXPECT_FLOAT_EQ(destination.data<float>()[3], 0.f);
}

TEST(Eagle3AlignmentTest, PartialDraftHiddenStateUsesLeftRows) {
    ov::Tensor source(ov::element::f32, ov::Shape{1, 2, 1});
    source.data<float>()[0] = 10.f;
    source.data<float>()[1] = 20.f;
    ov::Tensor destination(ov::element::f32, ov::Shape{1, 4, 1});

    ov::npuw::util::pad_eagle3_hidden_state(ov::get_tensor_impl(source), ov::get_tensor_impl(destination));

    EXPECT_FLOAT_EQ(destination.data<float>()[0], 10.f);
    EXPECT_FLOAT_EQ(destination.data<float>()[1], 20.f);
    EXPECT_FLOAT_EQ(destination.data<float>()[2], 0.f);
    EXPECT_FLOAT_EQ(destination.data<float>()[3], 0.f);
}

TEST(Eagle3AlignmentTest, PartialDraftAndTargetOutputsUseOnlyLeftRows) {
    for (const auto role : {ov::npuw::Eagle3ModelRole::Draft, ov::npuw::Eagle3ModelRole::Target}) {
        SCOPED_TRACE(role == ov::npuw::Eagle3ModelRole::Draft ? "Draft" : "Target");
        ov::Tensor first_chunk(ov::element::f32, ov::Shape{1, 4, 1});
        constexpr std::array<float, 4> first_values{1.f, 2.f, 90.f, 91.f};
        std::copy(first_values.begin(), first_values.end(), first_chunk.data<float>());
        ov::Tensor final_chunk(ov::element::f32, ov::Shape{1, 4, 1});
        constexpr std::array<float, 4> final_values{3.f, 92.f, 93.f, 94.f};
        std::copy(final_values.begin(), final_values.end(), final_chunk.data<float>());
        ov::Tensor accumulated(ov::element::f32, ov::Shape{1, 3, 1});

        ov::npuw::util::copy_eagle3_chunk_output(ov::get_tensor_impl(first_chunk),
                                                 ov::get_tensor_impl(accumulated),
                                                 2u,
                                                 0u);
        ov::npuw::util::copy_eagle3_chunk_output(ov::get_tensor_impl(final_chunk),
                                                 ov::get_tensor_impl(accumulated),
                                                 1u,
                                                 2u);

        EXPECT_FLOAT_EQ(accumulated.data<float>()[0], 1.f);
        EXPECT_FLOAT_EQ(accumulated.data<float>()[1], 2.f);
        EXPECT_FLOAT_EQ(accumulated.data<float>()[2], 3.f);
    }
}

}  // namespace