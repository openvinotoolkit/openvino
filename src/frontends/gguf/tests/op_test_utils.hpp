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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "cnpy.h"
#include "common_test_utils/file_utils.hpp"
#include "gtest/gtest.h"
#include "op_table.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/visibility.hpp"
#include "openvino/frontend/gguf/frontend.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/util/file_util.hpp"

// TEST_DATA_DIR is injected by CMakeLists.txt as an absolute path to the source-tree
// test_data/ directory; used as a fallback when running from the build tree.
#ifndef TEST_DATA_DIR
#    error "TEST_DATA_DIR must be defined by CMake (add_compile_definitions)"
#endif

namespace ov_gguf_test {

using namespace ov::frontend::gguf;

// Set of ggml op types that some test in this binary has actually converted.  Every
// SingleOpDecoder construction records its op type here, so the record is a by-product of the tests
// running rather than a hand-maintained list that can drift.  Checked against op_table.cpp by the
// coverage gate in test_op_coverage.cpp, which therefore fails when a new op is registered without
// a test.  Populated at run time, so the gate has to run last -- see that file for how.
inline std::set<std::string>& converted_op_types() {
    static std::set<std::string> ops;
    return ops;
}

// Description of one tensor (graph input or op output) in the single-op model.
struct TensorDesc {
    std::string name;
    ov::element::Type type;
    ov::PartialShape shape;
};

// A minimal GgufDecoder that exposes exactly one ggml op.  The op has `inputs`
// (all of which become graph Parameters) and a single output.  Operation parameters
// are exposed through the typed node-scoped get_attribute(name) accessor, matching the
// decoder API the op translators consume; a real decoder (e.g. the llama.cpp cgraph
// decoder) populates the same attributes from ggml's op_params.
class SingleOpDecoder : public GgufDecoder, public std::enable_shared_from_this<SingleOpDecoder> {
public:
    SingleOpDecoder(std::string op_type,
                    std::vector<TensorDesc> inputs,
                    TensorDesc output,
                    std::map<std::string, ov::Any> attributes)
        : m_op_type(std::move(op_type)),
          m_inputs(std::move(inputs)),
          m_output(std::move(output)),
          m_attributes(std::move(attributes)) {
        converted_op_types().insert(m_op_type);
        for (const auto& in : m_inputs) {
            m_input_names.push_back(in.name);
            auto p = std::make_shared<ov::op::v0::Parameter>(in.type, in.shape);
            p->set_friendly_name(in.name);
            p->output(0).set_names({in.name});
            m_model_inputs[in.name] = p;
        }
    }

    // ── typed node-scoped attribute access (we hold a single op) ────────────────
    ov::Any get_attribute(const std::string& name) const override {
        auto it = m_attributes.find(name);
        return it == m_attributes.end() ? ov::Any{} : it->second;
    }

    // ── per-node metadata ───────────────────────────────────────────────────────
    int64_t get_input_view_element_offset(const std::string&) const override {
        return 0;
    }
    ov::PartialShape get_input_shape(const std::string& name) const override {
        return find_input(name).shape;
    }
    size_t get_input_size() const override {
        return m_inputs.size();
    }

    std::vector<std::string> get_input_names() const override {
        return m_input_names;
    }

    ov::PartialShape get_output_shape() const override {
        return m_output.shape;
    }

    std::vector<std::string> get_output_names() const override {
        return {m_output.name};
    }

    const std::string& get_op_type() const override {
        return m_op_type;
    }
    const std::string& get_op_name() const override {
        return m_output.name;
    }

    void visit_subgraph(std::function<void(std::shared_ptr<GgufDecoder>)> node_visitor) const override {
        node_visitor(std::const_pointer_cast<SingleOpDecoder>(shared_from_this()));
    }

    const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_inputs() const override {
        return m_model_inputs;
    }
    std::vector<std::string> get_model_output_names() const override {
        return {m_output.name};
    }

    // The optional model-scope accessors (get_model_extra_inputs, get_tokenizer_config) both
    // default to empty on GgufDecoder, which is exactly right for a single-op test decoder: no
    // auxiliary inputs and no tokenizer metadata. So neither is overridden here.

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
    std::vector<std::string> m_input_names;
    std::map<std::string, std::shared_ptr<ov::Node>> m_model_inputs;
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
        m_attributes["op_case"] = ov::Any(c);
        return *this;
    }

    // Build the single-op decoder. A SingleOpDecoder exposes no "rope_config" attribute, so the
    // frontend's InputModel::get_rope_config returns a default (no shared rope table) and the LLM
    // preprocess step no-ops -- exactly what a single-op test wants, with no naive flag.
    std::shared_ptr<GgufDecoder> decoder() const {
        auto attrs = m_attributes;
        attrs.emplace("output_type", ov::Any(m_output.type));
        return std::make_shared<SingleOpDecoder>(m_op_type, m_inputs, m_output, attrs);
    }

    std::shared_ptr<ov::Model> build() const {
        FrontEnd fe;
        return fe.convert(fe.load(decoder()));
    }

    // Convert through a FrontEnd that has the given extensions registered first.
    std::shared_ptr<ov::Model> build_with_extensions(
        const std::vector<std::shared_ptr<ov::Extension>>& extensions) const {
        FrontEnd fe;
        for (const auto& ext : extensions) {
            fe.add_extension(ext);
        }
        return fe.convert(fe.load(decoder()));
    }

private:
    std::string m_op_type;
    std::vector<TensorDesc> m_inputs;
    TensorDesc m_output;
    std::map<std::string, ov::Any> m_attributes;
};

// ── inference / comparison helpers ──────────────────────────────────────────────

inline ov::Tensor make_f32_tensor(const ov::Shape& shape, const std::vector<float>& data) {
    ov::Tensor t(ov::element::f32, shape);
    std::copy(data.begin(), data.end(), t.data<float>());
    return t;
}

// Compile on CPU and run one inference with the given named inputs; return the single output.
//
// Inference precision is requested as f32: these tests validate the converted graph against an fp32
// reference, not the plugin's reduced-precision arithmetic, so wherever fp32 inference is available we
// want it regardless of the plugin's performance-mode default (e.g. bf16 on avx512_core_bf16 hosts).
// Where fp32 is not supported (ARM, which always infers in fp16) the request is silently ignored, and
// the wider tolerance below covers the resulting rounding error.
inline ov::Tensor run_on_cpu(const std::shared_ptr<ov::Model>& model, const std::map<std::string, ov::Tensor>& inputs) {
    ov::Core core;
    auto compiled = core.compile_model(model, "CPU", ov::hint::inference_precision(ov::element::f32));
    auto req = compiled.create_infer_request();
    for (const auto& kv : inputs) {
        req.set_tensor(kv.first, kv.second);
    }
    req.infer();
    return req.get_output_tensor(0);
}

// Default relative tolerance, per inference precision the CPU plugin actually uses.
//
// On ARM the f32 request above cannot be honored (the plugin always infers in fp16), so rounding
// error accumulates through long op chains such as rope; the measured worst case needs ~3e-3.
//
// Everywhere else the f32 request holds and the measured worst case across this suite is ~1.4e-6,
// so the bound stays near fp32 precision — tight enough that a real conversion error cannot hide
// inside it.
#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
#    define OV_GGUF_TEST_DEFAULT_RTOL 1e-2f
#else
#    define OV_GGUF_TEST_DEFAULT_RTOL 1e-5f
#endif

// Compare against an fp32 reference with a combined absolute + relative tolerance:
//   |actual - expected| <= atol + rtol * |expected|
// The relative term matters on hardware that runs the graph in fp16 (e.g. ARM CPU), where the
// rounding error grows with the magnitude of the value.
inline void expect_near(const ov::Tensor& actual,
                        const std::vector<float>& expected,
                        float atol = 1e-4f,
                        float rtol = OV_GGUF_TEST_DEFAULT_RTOL) {
    ASSERT_EQ(actual.get_element_type(), ov::element::f32);
    ASSERT_EQ(actual.get_size(), expected.size());
    const float* a = actual.data<float>();
    for (size_t i = 0; i < expected.size(); ++i) {
        const float tol = atol + rtol * std::fabs(expected[i]);
        EXPECT_NEAR(a[i], expected[i], tol) << "mismatch at index " << i;
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
