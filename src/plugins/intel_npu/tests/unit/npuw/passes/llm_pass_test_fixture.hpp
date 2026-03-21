// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Shared GTest fixture for NPUW LLM pass unit tests.
//
// All pass-specific test fixtures should derive from LLMPassTestFixture instead of
// ::testing::Test directly.  The base provides:
//   • SetUp()            – creates m_plugin (NullPlugin)
//   • base_props()       – minimal LLM compile properties
//   • merge_props()      – merge two AnyMap instances
//   • create_compiled_model()         – builds a standard LLM sub-pipeline
//   • create_whisper_compiled_model() – builds a Whisper-decoder sub-pipeline
//   • count_ops<Op>()    – count nodes of a given type in a model graph
//   • port_has_name()    – checks whether any name on a port contains a needle
//   • count_inputs()     – counts inputs whose name contains a needle
//   • find_input()       – returns the first input port whose name contains a needle
//   • find_output()      – returns the first output port whose name contains a needle
//   • all_inputs_static() – returns true when every model input is fully static
//   • input_shape()       – returns the concrete shape of a named input
//
// Pass-specific helpers (e.g. any_matmul_has_transpose_b, has_input_with_name,
// all_inputs_with_name_have_type) should remain in the individual test files.

#pragma once

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

#include "llm_compiled_model.hpp"
#include "../llm_test_helpers.hpp"

namespace ov::test::npuw {

class LLMPassTestFixture : public ::testing::Test {
protected:
    void SetUp() override {
        m_plugin = std::make_shared<NullPlugin>();
    }

    // ── Properties ────────────────────────────────────────────────────────────

    static ov::AnyMap base_props() {
        return {{"NPUW_LLM", "YES"}, {"NPUW_LLM_MAX_PROMPT_LEN", "128"}, {"NPUW_LLM_MIN_RESPONSE_LEN", "64"}};
    }

    static void merge_props(ov::AnyMap& dst, const ov::AnyMap& src) {
        for (const auto& [key, value] : src) {
            dst[key] = value;
        }
    }

    // ── Model construction ───────────────────────────────────────────────────

    std::unique_ptr<ov::npuw::LLMCompiledModel> create_compiled_model(const ov::AnyMap& extra_props,
                                                                       RecordingFactory& recorder) const {
        auto props = base_props();
        merge_props(props, extra_props);
        return std::make_unique<ov::npuw::LLMCompiledModel>(
            build_llm_test_model(), m_plugin, props, recorder.make_factory());
    }

    std::unique_ptr<ov::npuw::LLMCompiledModel> create_whisper_compiled_model(const ov::AnyMap& extra_props,
                                                                               RecordingFactory& recorder) const {
        auto props = base_props();
        merge_props(props, extra_props);
        return std::make_unique<ov::npuw::LLMCompiledModel>(
            build_whisper_decoder_test_model(), m_plugin, props, recorder.make_factory());
    }

    // ── Graph inspection ─────────────────────────────────────────────────────

    template <class Op>
    static std::size_t count_ops(const std::shared_ptr<ov::Model>& model) {
        const auto ops = model->get_ops();
        return std::count_if(ops.begin(), ops.end(), [](const auto& op) {
            return ov::is_type<Op>(op);
        });
    }

    template <class Port>
    static bool port_has_name(const Port& port, std::string_view needle) {
        const auto& names = port.get_names();
        return std::any_of(names.begin(), names.end(), [needle](const std::string& name) {
            return name.find(needle) != std::string::npos;
        });
    }

    static std::size_t count_inputs(const std::shared_ptr<ov::Model>& model, std::string_view needle) {
        const auto inputs = model->inputs();
        return std::count_if(inputs.begin(), inputs.end(), [needle](const auto& input) {
            return port_has_name(input, needle);
        });
    }

    static std::optional<ov::Output<const ov::Node>> find_input(const std::shared_ptr<ov::Model>& model,
                                                                 std::string_view needle) {
        const auto inputs = model->inputs();
        const auto it = std::find_if(inputs.begin(), inputs.end(), [needle](const auto& input) {
            return port_has_name(input, needle);
        });
        return it == inputs.end() ? std::nullopt : std::optional<ov::Output<const ov::Node>>{*it};
    }

    static std::optional<ov::Output<const ov::Node>> find_output(const std::shared_ptr<ov::Model>& model,
                                                                  std::string_view needle) {
        const auto outputs = model->outputs();
        const auto it = std::find_if(outputs.begin(), outputs.end(), [needle](const auto& output) {
            return port_has_name(output, needle);
        });
        return it == outputs.end() ? std::nullopt : std::optional<ov::Output<const ov::Node>>{*it};
    }

    // Returns true only when every input of the model has a fully static partial shape.
    static bool all_inputs_static(const std::shared_ptr<ov::Model>& model) {
        const auto inputs = model->inputs();
        return std::all_of(inputs.begin(), inputs.end(), [](const auto& input) {
            return input.get_partial_shape().is_static();
        });
    }

    // Finds the first input whose name contains `needle` and returns its concrete shape.
    // Returns std::nullopt when no such input exists or its shape is not yet static.
    static std::optional<ov::Shape> input_shape(const std::shared_ptr<ov::Model>& model, std::string_view needle) {
        const auto port = find_input(model, needle);
        if (!port.has_value()) {
            return std::nullopt;
        }
        const auto& ps = port->get_partial_shape();
        if (!ps.is_static()) {
            return std::nullopt;
        }
        return ps.to_shape();
    }

    // ── Shared state ─────────────────────────────────────────────────────────

    std::shared_ptr<ov::IPlugin> m_plugin;
};

}  // namespace ov::test::npuw
