// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Shared GTest fixture for NPUW LLM pass unit tests.
//
// All pass-specific test fixtures should derive from LLMPassTestFixture instead of
// ::testing::Test directly.  The base provides:
//   * SetUp()            - creates m_plugin (NullPlugin)
//   * base_props()       - minimal LLM compile properties
//   * merge_props()      - merge two AnyMap instances
//   * create_compiled_model()         - builds a standard LLM sub-pipeline
//   * create_whisper_compiled_model() - builds a Whisper-decoder sub-pipeline
//   * count_ops<Op>()    - count nodes of a given type in a model graph
//   * port_has_name()    - checks whether any name on a port contains a needle
//   * count_inputs()     - counts inputs whose name contains a needle
//   * find_input()       - returns the first input port whose name contains a needle
//   * find_output()      - returns the first output port whose name contains a needle
//   * all_inputs_static()               - returns true when every model input is fully static
//   * input_shape()                     - returns the concrete shape of a named input
//   * all_inputs_with_name_have_type()  - true iff every input matching needle has expected_type; throws if needle unmatched
//   * all_outputs_with_name_have_type() - true iff every output matching needle has expected_type; throws if needle unmatched
//   * no_inputs_with_name_have_type()   - true iff no input matching needle has excluded_type; throws if needle unmatched
//   * require_sub_model()           - asserts a CompileCall with the given exact suffix exists and returns it
//   * require_sub_model_containing() - asserts a CompileCall whose name contains fragment exists and returns it
//                                      Prefer this over require_sub_model() for KV-cache sub-models whose suffix
//                                      encodes a concrete cache size (e.g. "_kv192") that changes with properties.
//                                      Use the stable "_kv" fragment instead of a size-specific suffix.
//
// Pass-specific helpers (e.g. any_matmul_has_transpose_b) should remain in the individual test files.

#pragma once

#include <algorithm>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
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

    // -- Properties -------------------------------------------------------------

    static ov::AnyMap base_props() {
        return {{"NPUW_LLM", "YES"}, {"NPUW_LLM_MAX_PROMPT_LEN", "128"}, {"NPUW_LLM_MIN_RESPONSE_LEN", "64"}};
    }

    static void merge_props(ov::AnyMap& dst, const ov::AnyMap& src) {
        for (const auto& [key, value] : src) {
            dst[key] = value;
        }
    }

    // -- Model construction --------------------------------------------------------

    std::unique_ptr<ov::npuw::LLMCompiledModel> create_compiled_model(const ov::AnyMap& extra_props,
                                                                       RecordingFactory& recorder) const {
        auto props = base_props();
        merge_props(props, extra_props);
        return std::make_unique<ov::npuw::LLMCompiledModel>(
            build_llm_test_model(), m_plugin, props, recorder.make_factory());
    }

    std::unique_ptr<ov::npuw::LLMCompiledModel> create_compiled_model(const std::shared_ptr<ov::Model>& model,
                                                                       const ov::AnyMap& extra_props,
                                                                       RecordingFactory& recorder) const {
        auto props = base_props();
        merge_props(props, extra_props);
        return std::make_unique<ov::npuw::LLMCompiledModel>(
            model, m_plugin, props, recorder.make_factory());
    }

    std::unique_ptr<ov::npuw::LLMCompiledModel> create_whisper_compiled_model(const ov::AnyMap& extra_props,
                                                                               RecordingFactory& recorder) const {
        auto props = base_props();
        merge_props(props, extra_props);
        return std::make_unique<ov::npuw::LLMCompiledModel>(
            build_whisper_decoder_test_model(), m_plugin, props, recorder.make_factory());
    }

    // -- Graph inspection ----------------------------------------------------------

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

    // Returns true iff every input whose name contains `needle` has `expected_type`.
    // Throws std::runtime_error if no input matching `needle` is found.
    static bool all_inputs_with_name_have_type(const std::shared_ptr<ov::Model>& model,
                                               std::string_view needle,
                                               ov::element::Type expected_type) {
        bool found_any = false;
        std::ostringstream mismatch;
        bool has_mismatch = false;

        for (const auto& input : model->inputs()) {
            if (port_has_name(input, needle)) {
                found_any = true;
                if (input.get_element_type() != expected_type) {
                    has_mismatch = true;
                    mismatch << "\n  - type=" << input.get_element_type() << ", names={";
                    bool first = true;
                    for (const auto& name : input.get_names()) {
                        if (!first) {
                            mismatch << ", ";
                        }
                        mismatch << name;
                        first = false;
                    }
                    if (first) {
                        mismatch << "<none>";
                    }
                    mismatch << "}";
                }
            }
        }
        if (!found_any) {
            std::ostringstream ss;
            ss << "No input matching needle '" << needle << "' found in model '" << model->get_friendly_name()
               << "'. Available inputs:";

            for (const auto& input : model->inputs()) {
                ss << "\n  - type=" << input.get_element_type() << ", names={";
                bool first = true;
                for (const auto& name : input.get_names()) {
                    if (!first) {
                        ss << ", ";
                    }
                    ss << name;
                    first = false;
                }
                if (first) {
                    ss << "<none>";
                }
                ss << "}";
            }

            throw std::runtime_error(ss.str());
        }

        if (has_mismatch) {
            ADD_FAILURE() << "Inputs matching needle '" << needle << "' in model '" << model->get_friendly_name()
                          << "' do not all have expected type " << expected_type
                          << ". Mismatched inputs:" << mismatch.str();
            return false;
        }

        return true;
    }

    // Returns true iff every output whose name contains `needle` has `expected_type`.
    // Throws std::runtime_error if no output matching `needle` is found.
    static bool all_outputs_with_name_have_type(const std::shared_ptr<ov::Model>& model,
                                                std::string_view needle,
                                                ov::element::Type expected_type) {
        bool found_any = false;
        std::ostringstream mismatch;
        bool has_mismatch = false;

        for (const auto& output : model->outputs()) {
            if (port_has_name(output, needle)) {
                found_any = true;
                if (output.get_element_type() != expected_type) {
                    has_mismatch = true;
                    mismatch << "\n  - type=" << output.get_element_type() << ", names={";
                    bool first = true;
                    for (const auto& name : output.get_names()) {
                        if (!first) {
                            mismatch << ", ";
                        }
                        mismatch << name;
                        first = false;
                    }
                    if (first) {
                        mismatch << "<none>";
                    }
                    mismatch << "}";
                }
            }
        }
        if (!found_any)
            throw std::runtime_error(std::string("No output matching needle '") + std::string(needle) +
                                     "' found in model '" + model->get_friendly_name() + "'");

        if (has_mismatch) {
            ADD_FAILURE() << "Outputs matching needle '" << needle << "' in model '" << model->get_friendly_name()
                          << "' do not all have expected type " << expected_type
                          << ". Mismatched outputs:" << mismatch.str();
            return false;
        }

        return true;
    }

    // Returns true iff no input whose name contains `needle` has `excluded_type`.
    // Throws std::runtime_error if no input matching `needle` is found.
    static bool no_inputs_with_name_have_type(const std::shared_ptr<ov::Model>& model,
                                              std::string_view needle,
                                              ov::element::Type excluded_type) {
        bool found_any = false;
        for (const auto& input : model->inputs()) {
            if (port_has_name(input, needle)) {
                found_any = true;
                if (input.get_element_type() == excluded_type)
                    return false;
            }
        }
        if (!found_any)
            throw std::runtime_error(std::string("No input matching needle '") + std::string(needle) +
                                     "' found in model '" + model->get_friendly_name() + "'");
        return true;
    }

    // -- Sub-model lookup ----------------------------------------------------------

    // Returns the CompileCall whose friendly name ends with `suffix`.
    // Triggers a fatal assertion if no such call exists.
    // Use for sub-models with a stable naming convention (_prefill, _lm_head, ...).
    static const CompileCall& require_sub_model(const RecordingFactory& recorder, std::string_view suffix) {
        const auto* call = recorder.find_suffix(suffix);
        OPENVINO_ASSERT(call != nullptr, "Missing compile call with suffix: ", std::string(suffix));
        return *call;
    }

    // Returns the CompileCall whose friendly name contains `fragment`.
    // Triggers a fatal assertion if no such call exists.
    // Prefer this over require_sub_model() for KV-cache sub-models: their suffix
    // encodes the concrete cache size (e.g. "_kv192") which changes with compile
    // properties.  Match on the stable fragment "_kv" instead.
    static const CompileCall& require_sub_model_containing(const RecordingFactory& recorder,
                                                           std::string_view fragment) {
        const auto* call = recorder.find_contains(fragment);
        OPENVINO_ASSERT(call != nullptr, "Missing compile call containing: ", std::string(fragment));
        return *call;
    }

    // -- Shared state ----------------------------------------------------------

    std::shared_ptr<ov::IPlugin> m_plugin;
};

}  // namespace ov::test::npuw
