// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Utilities shared by the GGUF frontend per-op test suite (test_ops.cpp).
//
// The GGUF frontend translation layer operates purely on the abstract GgufDecoder
// interface; it does not read .gguf files.  For tests we therefore provide a tiny
// in-memory decoder (SingleOpDecoder) that describes a single ggml op plus its
// input/output tensors, and drive ov::frontend::gguf::FrontEnd::convert on it.  This
// keeps the tests free of any llama.cpp / ggml dependency.
//
// Provides:
//   - SingleOpDecoder / SingleOpBuilder — describe one op and build an ov::Model.
//   - run_on_cpu  — compile an ov::Model on CPU and run one inference.
//   - expect_near — element-wise |actual-expected| <= atol check via GTest.

#pragma once

#include <cnpy.h>
#include <gtest/gtest.h>

#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "common_test_utils/file_utils.hpp"
#include "input_model.h"
#include "op_table.h"
#include "openvino/core/model.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/frontend/gguf/frontend.h"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/util/file_util.hpp"

// TEST_DATA_DIR is injected by CMakeLists.txt as an absolute path to the source-tree
// test_data/ directory; used as a fallback when running from the build tree.
#ifndef TEST_DATA_DIR
#    error "TEST_DATA_DIR must be defined by CMake (add_compile_definitions)"
#endif

namespace ov_gguf_test {

using namespace ov::frontend::gguf;

// Description of one tensor (graph input or op output) in the single-op model.
struct TensorDesc {
    std::string name;
    ov::element::Type type;
    ov::PartialShape shape;
};

// A minimal GgufDecoder that exposes exactly one ggml op.  The op has `inputs`
// (all of which become graph Parameters) and a single output.  Operation parameters
// are exposed through the typed get_attribute(node_idx, name) accessor, matching the
// decoder API the op translators consume; a real decoder (e.g. the llama.cpp cgraph
// decoder) populates the same attributes from ggml's op_params.
class SingleOpDecoder : public GgufDecoder, public std::enable_shared_from_this<SingleOpDecoder> {
public:
    SingleOpDecoder(std::string op_type,
                    std::vector<TensorDesc> inputs,
                    TensorDesc output,
                    std::map<std::string, ov::Any> attributes,
                    int op_case)
        : m_op_type(std::move(op_type)),
          m_inputs(std::move(inputs)),
          m_output(std::move(output)),
          m_attributes(std::move(attributes)),
          m_op_case(op_case) {
        for (const auto& in : m_inputs) {
            m_input_names.push_back(in.name);
            auto p = std::make_shared<ov::op::v0::Parameter>(in.type, in.shape);
            p->set_friendly_name(in.name);
            p->output(0).set_names({in.name});
            m_model_inputs[in.name] = p;
        }
    }

    // ── typed attribute access (node_idx is always 0; we hold a single op) ──────
    ov::Any get_attribute(const std::string& name) const override {
        return get_attribute(0, name);
    }
    ov::Any get_attribute(int, const std::string& name) const override {
        auto it = m_attributes.find(name);
        return it == m_attributes.end() ? ov::Any{} : it->second;
    }

    // ── per-node metadata ───────────────────────────────────────────────────────
    ov::PartialShape get_input_shape(int, const std::string& name) const override {
        return find_input(name).shape;
    }
    std::vector<size_t> get_input_stride(int, const std::string&) const override {
        return {};
    }
    int64_t get_input_view_offset(int, const std::string&) const override {
        return 0;
    }
    ov::element::Type get_input_type(int, const std::string& name) const override {
        return find_input(name).type;
    }
    size_t get_input_size() const override {
        return m_inputs.size();
    }
    size_t get_input_size(int) const override {
        return m_inputs.size();
    }

    void get_input_node(size_t, std::string&, std::string&, size_t&) const override {}

    std::vector<std::string> get_input_names(int) const override {
        return m_input_names;
    }

    ov::PartialShape get_output_shape(int) const override {
        return m_output.shape;
    }
    ov::element::Type get_output_type(int) const override {
        return m_output.type;
    }

    std::vector<std::string> get_output_names(int) const override {
        return {m_output.name};
    }

    const std::string& get_op_type() const override {
        return m_op_type;
    }
    const std::string& get_op_type(int) const override {
        return m_op_type;
    }
    const std::string& get_op_name() const override {
        return m_output.name;
    }
    const std::string& get_op_name(int) const override {
        return m_output.name;
    }

    void visit_subgraph(std::function<void(std::shared_ptr<GgufDecoder>, int)> node_visitor) const override {
        node_visitor(std::const_pointer_cast<SingleOpDecoder>(shared_from_this()), 0);
    }

    int get_op_case(int) const override {
        return m_op_case;
    }

    const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_inputs() const override {
        return m_model_inputs;
    }
    const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_extra_inputs() const override {
        return m_empty;
    }
    std::vector<std::string> get_model_output_names() const override {
        return {m_output.name};
    }

    bool has_rope() const override {
        return false;
    }
    bool use_per_op_rope() const override {
        return false;
    }
    RopeConfig get_rope_config() const override {
        return {};
    }
    std::map<std::string, std::string> get_kv_param_res_names() const override {
        return {};
    }
    bool is_static() const override {
        return false;
    }
    bool is_stateful() const override {
        return false;
    }
    bool is_swa_layer(int) const override {
        return false;
    }

private:
    const TensorDesc& find_input(const std::string& name) const {
        for (const auto& in : m_inputs) {
            if (in.name == name) {
                return in;
            }
        }
        throw std::runtime_error("SingleOpDecoder: unknown input '" + name + "'");
    }

    std::string m_op_type;
    std::vector<TensorDesc> m_inputs;
    TensorDesc m_output;
    std::map<std::string, ov::Any> m_attributes;
    int m_op_case;
    std::vector<std::string> m_input_names;
    std::map<std::string, std::shared_ptr<ov::Node>> m_model_inputs;
    std::map<std::string, std::shared_ptr<ov::Node>> m_empty;
};

// Fluent builder: describe a single op and convert it to an ov::Model.
class SingleOpBuilder {
public:
    SingleOpBuilder& op(const std::string& op_type) {
        m_op_type = op_type;
        return *this;
    }
    SingleOpBuilder& input(const std::string& name, ov::element::Type type, const ov::PartialShape& shape) {
        m_inputs.push_back({name, type, shape});
        return *this;
    }
    SingleOpBuilder& output(const std::string& name, ov::element::Type type, const ov::PartialShape& shape) {
        m_output = {name, type, shape};
        return *this;
    }
    // Set a typed operation attribute (e.g. "scale", "eps", "swapped"), the way a real
    // decoder exposes ggml op_params to the translators.
    template <typename T>
    SingleOpBuilder& attr(const std::string& name, const T& value) {
        m_attributes[name] = ov::Any(value);
        return *this;
    }
    SingleOpBuilder& op_case(int c) {
        m_op_case = c;
        return *this;
    }

    // Build the single-op InputModel (naive=true skips the LLM preprocess step -- rope
    // sin/cos, attention mask -- which is irrelevant for single-op tests).
    std::shared_ptr<InputModel> input_model() const {
        auto decoder = std::make_shared<SingleOpDecoder>(m_op_type, m_inputs, m_output, m_attributes, m_op_case);
        return std::make_shared<InputModel>(decoder, true);
    }

    std::shared_ptr<ov::Model> build() const {
        return FrontEnd().convert(input_model());
    }

    // Convert through a FrontEnd that has the given extensions registered first.
    std::shared_ptr<ov::Model> build_with_extensions(
        const std::vector<std::shared_ptr<ov::Extension>>& extensions) const {
        FrontEnd fe;
        for (const auto& ext : extensions) {
            fe.add_extension(ext);
        }
        return fe.convert(input_model());
    }

private:
    std::string m_op_type;
    std::vector<TensorDesc> m_inputs;
    TensorDesc m_output;
    std::map<std::string, ov::Any> m_attributes;
    int m_op_case = 0;
};

// ── inference / comparison helpers ──────────────────────────────────────────────

inline ov::Tensor make_f32_tensor(const ov::Shape& shape, const std::vector<float>& data) {
    ov::Tensor t(ov::element::f32, shape);
    std::copy(data.begin(), data.end(), t.data<float>());
    return t;
}

// Compile on CPU and run one inference with the given named inputs; return the single output.
inline ov::Tensor run_on_cpu(const std::shared_ptr<ov::Model>& model, const std::map<std::string, ov::Tensor>& inputs) {
    ov::Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();
    for (const auto& kv : inputs) {
        req.set_tensor(kv.first, kv.second);
    }
    req.infer();
    return req.get_output_tensor(0);
}

inline void expect_near(const ov::Tensor& actual, const std::vector<float>& expected, float atol = 1e-4f) {
    ASSERT_EQ(actual.get_element_type(), ov::element::f32);
    ASSERT_EQ(actual.get_size(), expected.size());
    const float* a = actual.data<float>();
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(a[i], expected[i], atol) << "mismatch at index " << i;
    }
}

// ── npy helpers (used by the dequant tests) ─────────────────────────────────────

// Locate the test_data directory.  When installed (CI), the data lives in a "test_data"
// folder next to the test executable; when running from the build tree it is in the
// source tree at the compile-time TEST_DATA_DIR path.
inline std::string test_data_dir() {
    const std::string installed =
        ov::util::path_join({ov::test::utils::getExecutableDirectory(), "test_data"}).string();
    if (ov::util::directory_exists(installed)) {
        return installed;
    }
    return TEST_DATA_DIR;
}

inline std::string test_data_path(const std::string& stem) {
    return ov::util::path_join({test_data_dir(), stem + ".npy"}).string();
}

// Load a .npy array as a flat vector of T.
template <typename T>
std::vector<T> load_npy(const std::string& stem) {
    cnpy::NpyArray arr = cnpy::npy_load(test_data_path(stem));
    const T* begin = arr.data<T>();
    return std::vector<T>(begin, begin + arr.num_vals);
}

}  // namespace ov_gguf_test
