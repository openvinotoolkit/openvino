// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>

#include "cnpy.h"
#include "gtest/gtest.h"
#include "op_test_utils.hpp"
#include "openvino/frontend/extension/decoder_transformation.hpp"
#include "openvino/frontend/gguf/make_stateful.hpp"
#include "openvino/openvino.hpp"

namespace {
class GGUFLfm3Speech : public ::testing::Test {
protected:
    void SetUp() override {
        const auto path = std::filesystem::path(ov_gguf_test::test_data_dir()) / "mmproj_accuracy/lfm3_speech.npz";
        ASSERT_TRUE(std::filesystem::exists(path)) << "Regenerate with gen_lfm3_speech_fixtures.py";
        arrays = cnpy::npz_load(path.string());
        fixture = std::make_unique<ov_gguf_test::TemporaryGguf>(ov_gguf_test::npz_array(arrays, "model"));
        ov::frontend::gguf::FrontEnd frontend;
        frontend.add_extension(std::make_shared<ov::frontend::DecoderTransformationExtension>(
            ov::frontend::gguf::pass::GGUFMakeStateful()));
        auto model = frontend.convert(frontend.load(fixture->path));
        ov::Core core;
        compiled = core.compile_model(model,
                                      "CPU",
                                      {{"INFERENCE_PRECISION_HINT", "f32"},
                                       {"DYNAMIC_QUANTIZATION_GROUP_SIZE", 0},
                                       {"KV_CACHE_PRECISION", "f32"}});
        request = compiled.create_infer_request();
    }
    std::vector<float> floats(const std::string& name) const {
        const auto& values = ov_gguf_test::npz_array(arrays, name);
        return {values.data<float>(), values.data<float>() + values.num_vals};
    }
    void input(const std::string& name, const ov::Shape& shape, const std::vector<float>& values) {
        ov::Tensor tensor(ov::element::f32, shape);
        std::copy(values.begin(), values.end(), tensor.data<float>());
        request.set_tensor(name, tensor);
    }
    void codes(const std::string& name, size_t offset, size_t frames, bool masked = false) {
        ov::Tensor tensor(ov::element::i32, {1, 1, frames, 12});
        const auto* source =
            ov_gguf_test::npz_array(arrays, name == "audio.codec.codes" ? "codes" : "depth_codes").data<int32_t>();
        std::copy_n(source + offset * 12, frames * 12, tensor.data<int32_t>());
        if (masked)
            std::fill_n(tensor.data<int32_t>(), tensor.get_size(), 0);
        request.set_tensor(name, tensor);
    }
    void prepare(size_t offset, size_t frames, size_t base, int phase) {
        input("audio.depth.hidden_states", {1, 1, 9, 6}, floats("hidden"));
        codes("audio.depth.codes", 0, 9, phase == 0);
        codes("audio.feedback.codes", 0, 9);
        codes("audio.codec.codes", offset, frames);
        std::vector<float> mask(9 * 12);
        for (size_t frame = 0; frame < 9; ++frame)
            for (size_t book = 0; book < 12; ++book)
                mask[frame * 12 + book] = phase == 0 ? 0.f : float(book % 3 == 0);
        input("audio.depth.mask", {1, 1, 9, 12}, mask);
        const auto count = 6 * frames;
        ov::Tensor positions(ov::element::i32, {1, 1, 1, count});
        for (size_t i = 0; i < count; ++i)
            positions.data<int32_t>()[i] = int32_t(base + i);
        request.set_tensor("audio.codec.positions", positions);
        std::vector<float> attention(count * (29 + count), -std::numeric_limits<float>::infinity());
        for (size_t q = 0; q < count; ++q)
            for (size_t k = q; k <= q + 29; ++k)
                if (int64_t(base) - 29 + int64_t(k) >= 0)
                    attention[q * (29 + count) + k] = 0.f;
        input("audio.codec.attention_mask", {1, 1, count, 29 + count}, attention);
    }
    void check_depth(int phase) {
        std::vector<float> expected, feedback;
        for (size_t frame = 0; frame < 9; ++frame) {
            const auto value = floats("depth_" + std::to_string(phase) + "_" + std::to_string(frame));
            expected.insert(expected.end(), value.begin(), value.end());
            const auto embedding = floats("feedback_" + std::to_string(frame));
            feedback.insert(feedback.end(), embedding.begin(), embedding.end());
        }
        ov_gguf_test::expect_near(request.get_tensor("audio.depth_logits"), expected);
        ov_gguf_test::expect_near(request.get_tensor("audio.feedback"), feedback);
    }
    void check_state(int step) {
        const auto states = request.query_state();
        ASSERT_EQ(states.size(), 13);
        for (const auto& state : states) {
            const auto name = state.get_name();
            const auto prefix = std::string("audio.codec.state.");
            ASSERT_EQ(name.find(prefix), 0);
            auto slot = name.substr(prefix.size());
            std::replace(slot.begin(), slot.end(), '.', '_');
            const auto expected = floats("state_" + std::to_string(step) + "_gen_state_" + slot);
            auto actual = state.get_state();
            SCOPED_TRACE(name + " " + ov::PartialShape(actual.get_shape()).to_string());
            if (slot.find("conv_") == 0) {
                // GGUFMakeStateful retains one extra sample for causal-convolution fusion.
                ASSERT_EQ(actual.get_shape(), (ov::Shape{1, 512, 4}));
                actual = ov_gguf_test::trailing_conv_taps(actual, 512, 3);
            }
            ov_gguf_test::expect_near(actual, expected);
        }
    }
    void check_spectrum(int step, size_t frames) {
        const auto actual = request.get_tensor("audio.spectrogram");
        const auto expected = floats("spectra_" + std::to_string(step));
        ASSERT_EQ(actual.get_shape(), (ov::Shape{1, 1, 6 * frames, 1282}));
        ASSERT_EQ(actual.get_size(), expected.size());
        for (size_t i = 0; i < actual.get_size(); ++i)
            ASSERT_TRUE(std::isfinite(actual.data<const float>()[i])) << i;
        ov_gguf_test::expect_nmse_below(
            ov_gguf_test::nmse(actual.data<const float>(), expected.data(), actual.get_size()),
            1e-5);
    }
    cnpy::npz_t arrays;
    std::unique_ptr<ov_gguf_test::TemporaryGguf> fixture;
    ov::CompiledModel compiled;
    ov::InferRequest request;
};

TEST_F(GGUFLfm3Speech, MaskedAndRevealedBidirectionalDepthAndFeedback) {
    for (int phase = 0; phase < 2; ++phase) {
        for (auto state : request.query_state())
            state.reset();
        prepare(0, 1, 0, phase);
        request.infer();
        check_depth(phase);
    }
}

TEST_F(GGUFLfm3Speech, StreamingSpectraStateResetAndWindowBoundary) {
    const size_t counts[] = {1, 2, 6, 2, 9};
    size_t offset = 0, base = 0;
    for (int step = 0; step < 5; ++step) {
        SCOPED_TRACE("codec step " + std::to_string(step));
        if (step >= 3) {
            for (auto state : request.query_state())
                state.reset();
            offset = step == 3 ? 1 : 0;
            base = 0;
        }
        prepare(offset, counts[step], base, 1);
        request.infer();
        check_spectrum(step, counts[step]);
        check_state(step);
        offset += counts[step];
        base += counts[step] * 6;
    }
}

TEST_F(GGUFLfm3Speech, FiniteScalarCodebookBoundaries) {
    prepare(0, 1, 0, 0);
    const auto source = request.get_tensor("audio.codec.codes");
    ASSERT_EQ(source.data<int32_t>()[0], 0);
    ASSERT_EQ(source.data<int32_t>()[1], 2015);
    request.infer();
    check_spectrum(0, 1);
}
}  // namespace
