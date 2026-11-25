// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <vector>

#include "openvino/op/ops.hpp"
#include "openvino/openvino.hpp"

namespace {

std::shared_ptr<ov::Model> make_audio_model(bool use_length = false) {
    auto features = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 8, -1});
    features->output(0).set_names({"padded_feature"});
    auto mask = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, ov::PartialShape{-1, -1});
    mask->output(0).set_names({"padded_mask_after_cnn"});
    auto lengths = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1});
    lengths->output(0).set_names({"aftercnn_lens"});
    auto offsets = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    offsets->output(0).set_names({"cu_seqlens"});
    auto mean =
        std::make_shared<ov::op::v1::ReduceMean>(features,
                                                 ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}),
                                                 false);
    auto first = std::make_shared<ov::op::v8::Gather>(mean,
                                                      ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0}),
                                                      ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1}));
    auto masked =
        std::make_shared<ov::op::v1::Multiply>(first, std::make_shared<ov::op::v0::Convert>(mask, ov::element::f32));
    ov::Output<ov::Node> values = masked;
    if (use_length) {
        values =
            std::make_shared<ov::op::v1::Add>(values, std::make_shared<ov::op::v0::Convert>(lengths, ov::element::f32));
    }
    auto output =
        std::make_shared<ov::op::v0::Unsqueeze>(values,
                                                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}));
    output->output(0).set_names({"audio_features"});
    return std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{features, mask, lengths, offsets});
}

ov::AnyMap properties() {
    return {{"NPU_USE_NPUW", "YES"},
            {"NPUW_QWEN3_ASR_AUDIO_ENCODER", "YES"},
            {"NPUW_QWEN3_ASR_AUDIO_WINDOW", uint64_t{16}},
            {"NPUW_DEVICES", "CPU"},
            {"NPUW_ONLINE_PIPELINE", "NONE"}};
}

void check_windows(ov::CompiledModel& compiled) {
    auto request = compiled.create_infer_request();
    for (const std::size_t frames : {1u, 8u, 15u, 16u}) {
        ov::Tensor features{ov::element::f32, {2, 8, frames}};
        for (std::size_t row = 0; row < 2; ++row) {
            std::fill_n(features.data<float>() + row * 8 * frames, 8 * frames, static_cast<float>(row + 1));
        }
        const auto tokens = (frames + 7) / 8;
        ov::Tensor mask{ov::element::boolean, {2, tokens}};
        std::fill_n(mask.data<bool>(), mask.get_size(), true);
        ov::Tensor lengths{ov::element::i64, {1}};
        lengths.data<int64_t>()[0] = static_cast<int64_t>(2 * tokens);
        ov::Tensor offsets{ov::element::i32, {2}};
        offsets.data<int32_t>()[0] = 0;
        offsets.data<int32_t>()[1] = static_cast<int32_t>(2 * tokens);
        request.set_tensor("padded_feature", features);
        request.set_tensor("padded_mask_after_cnn", mask);
        request.set_tensor("aftercnn_lens", lengths);
        request.set_tensor("cu_seqlens", offsets);
        ASSERT_NO_THROW(request.infer());
        const auto output = request.get_tensor("audio_features");
        ASSERT_EQ(output.get_shape(), (ov::Shape{2, tokens, 1}));
        for (std::size_t row = 0; row < 2; ++row) {
            for (std::size_t token = 0; token < tokens; ++token) {
                EXPECT_FLOAT_EQ(output.data<const float>()[row * tokens + token], static_cast<float>(row + 1));
            }
        }
    }
}

TEST(NPUWQwen3ASRAudioTest, RunsShortAndBatchedWindows) {
    ov::Core core;
    auto compiled = core.compile_model(make_audio_model(), "NPU", properties());
    EXPECT_TRUE(compiled.get_property("NPUW_QWEN3_ASR_AUDIO_ENCODER").as<bool>());
    EXPECT_EQ(compiled.get_property("NPUW_QWEN3_ASR_AUDIO_WINDOW").as<uint64_t>(), 16u);
    check_windows(compiled);
}

TEST(NPUWQwen3ASRAudioTest, RejectsCrossWindowMetadataAndInvalidWindow) {
    ov::Core core;
    EXPECT_THROW(core.compile_model(make_audio_model(true), "NPU", properties()), ov::Exception);
    auto invalid = properties();
    invalid["NPUW_QWEN3_ASR_AUDIO_WINDOW"] = uint64_t{0};
    EXPECT_THROW(core.compile_model(make_audio_model(), "NPU", invalid), ov::Exception);
}

TEST(NPUWQwen3ASRAudioTest, RejectsOversizedWindowsAndNonPrefixMasks) {
    ov::Core core;
    auto compiled = core.compile_model(make_audio_model(), "NPU", properties());
    auto request = compiled.create_infer_request();
    request.set_tensor("padded_feature", ov::Tensor(ov::element::f32, {1, 8, 17}));
    request.set_tensor("padded_mask_after_cnn", ov::Tensor(ov::element::boolean, {1, 3}));
    EXPECT_THROW(request.infer(), ov::Exception);

    request.set_tensor("padded_feature", ov::Tensor(ov::element::f32, {1, 8, 16}));
    ov::Tensor mask{ov::element::boolean, {1, 2}};
    mask.data<bool>()[0] = false;
    mask.data<bool>()[1] = true;
    request.set_tensor("padded_mask_after_cnn", mask);
    EXPECT_THROW(request.infer(), ov::Exception);
}

TEST(NPUWQwen3ASRAudioTest, ExportsAndImportsDynamicInterface) {
    ov::Core core;
    auto config = properties();
    config[ov::cache_mode.name()] = ov::CacheMode::OPTIMIZE_SPEED;
    auto compiled = core.compile_model(make_audio_model(), "NPU", config);
    std::stringstream blob;
    compiled.export_model(blob);
    auto imported = core.import_model(blob, "NPU", properties());
    EXPECT_TRUE(imported.get_property("NPUW_QWEN3_ASR_AUDIO_ENCODER").as<bool>());
    EXPECT_EQ(imported.get_property("NPUW_QWEN3_ASR_AUDIO_WINDOW").as<uint64_t>(), 16u);
    check_windows(imported);
}

}  // namespace
