// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>

#include "all_close.hpp"
#include "all_close_f.hpp"
#include "engine_factory.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/function.hpp"
#include "ngraph/ngraph.hpp"
#include "openvino/runtime/core.hpp"
#include "test_tools.hpp"

namespace ngraph {
namespace test {
inline std::string backend_name_to_device(const std::string& backend_name) {
    if (backend_name == "INTERPRETER")
        return "TEMPLATE";
    if (backend_name == "IE_CPU")
        return "CPU";
    if (backend_name == "IE_GPU")
        return "GPU";
    throw ngraph_error("Unsupported backend name");
}

std::shared_ptr<Function> function_from_ir(const std::string& xml_path, const std::string& bin_path = {});

class TestCase {
public:
    TestCase(const std::shared_ptr<Function>& function, const std::string& dev = "TEMPLATE") : m_function{function} {
        try {
            // Register template plugin
            m_core.register_plugin(std::string("openvino_template_plugin") + IE_BUILD_POSTFIX, "TEMPLATE");
        } catch (...) {
        }
        m_request = m_core.compile_model(function, dev).create_infer_request();
    }

    template <typename T>
    void add_input(const Shape& shape, const std::vector<T>& values) {
        const auto params = m_function->get_parameters();
        NGRAPH_CHECK(m_input_index < params.size(), "All function parameters already have inputs.");

        const auto& input_pshape = params.at(m_input_index)->get_partial_shape();
        NGRAPH_CHECK(input_pshape.compatible(shape),
                     "Provided input shape ",
                     shape,
                     " is not compatible with nGraph function's expected input shape ",
                     input_pshape,
                     " for input ",
                     m_input_index);

        auto t_shape = m_request.get_input_tensor(m_input_index).get_shape();
        bool is_dynamic = false;
        for (const auto& dim : t_shape) {
            if (!dim) {
                is_dynamic = true;
                break;
            }
        }

        if (is_dynamic) {
            ov::Tensor tensor(params.at(m_input_index)->get_element_type(), shape);

            std::copy(values.begin(), values.end(), tensor.data<T>());
            m_request.set_input_tensor(m_input_index, tensor);
        } else {
            auto tensor = m_request.get_input_tensor(m_input_index);
            NGRAPH_CHECK(tensor.get_size() >= values.size(),
                         "Tensor and values have different sizes. Tensor (",
                         tensor.get_shape(),
                         ") size: ",
                         tensor.get_size(),
                         " and values size is ",
                         values.size());
            std::copy(values.begin(), values.end(), tensor.data<T>());
        }

        ++m_input_index;
    }

    template <typename T>
    void add_input(const std::vector<T>& values) {
        const auto& input_pshape = m_function->get_parameters().at(m_input_index)->get_partial_shape();

        NGRAPH_CHECK(input_pshape.is_static(),
                     "Input number ",
                     m_input_index,
                     " in the tested graph has dynamic shape. You need to provide ",
                     "shape information when setting values for this input.");

        add_input<T>(input_pshape.to_shape(), values);
    }

    template <typename T>
    void add_multiple_inputs(const std::vector<std::vector<T>>& vector_of_values) {
        for (const auto& value : vector_of_values) {
            add_input<T>(value);
        }
    }

    template <typename T>
    void add_input_from_file(const Shape& shape, const std::string& basepath, const std::string& filename) {
        NGRAPH_SUPPRESS_DEPRECATED_START
        const auto filepath = ngraph::file_util::path_join(basepath, filename);
        add_input_from_file<T>(shape, filepath);
        NGRAPH_SUPPRESS_DEPRECATED_END
    }

    template <typename T>
    void add_input_from_file(const std::string& basepath, const std::string& filename) {
        NGRAPH_SUPPRESS_DEPRECATED_START
        const auto filepath = ngraph::file_util::path_join(basepath, filename);
        add_input_from_file<T>(filepath);
        NGRAPH_SUPPRESS_DEPRECATED_END
    }

    template <typename T>
    void add_input_from_file(const Shape& shape, const std::string& filepath) {
        const auto value = read_binary_file<T>(filepath);
        add_input<T>(shape, value);
    }

    template <typename T>
    void add_input_from_file(const std::string& filepath) {
        const auto value = read_binary_file<T>(filepath);
        add_input<T>(value);
    }

    template <typename T>
    void add_expected_output(const Shape& expected_shape, const std::vector<T>& values) {
        const auto results = m_function->get_results();

        NGRAPH_CHECK(m_output_index < results.size(), "All model results already have expected outputs.");

        const auto& output_pshape = results.at(m_output_index)->get_output_partial_shape(0);
        NGRAPH_CHECK(output_pshape.compatible(expected_shape),
                     "Provided expected output shape ",
                     expected_shape,
                     " is not compatible with OpenVINO model's output shape ",
                     output_pshape,
                     " for output ",
                     m_output_index);

        ov::Tensor tensor(results[m_output_index]->get_output_element_type(0), expected_shape);
        std::copy(values.begin(), values.end(), tensor.data<T>());

        m_expected_outputs.push_back(std::move(tensor));

        ++m_output_index;
    }

    template <typename T>
    void add_expected_output(const std::vector<T>& values) {
        const auto results = m_function->get_results();

        NGRAPH_CHECK(m_output_index < results.size(), "All model results already have expected outputs.");

        const auto shape = results.at(m_output_index)->get_shape();
        add_expected_output<T>(shape, values);
    }

    template <typename T>
    void add_expected_output_from_file(const ngraph::Shape& expected_shape,
                                       const std::string& basepath,
                                       const std::string& filename) {
        NGRAPH_SUPPRESS_DEPRECATED_START
        const auto filepath = ngraph::file_util::path_join(basepath, filename);
        add_expected_output_from_file<T>(expected_shape, filepath);
        NGRAPH_SUPPRESS_DEPRECATED_END
    }

    template <typename T>
    void add_expected_output_from_file(const ngraph::Shape& expected_shape, const std::string& filepath) {
        const auto values = read_binary_file<T>(filepath);
        add_expected_output<T>(expected_shape, values);
    }

    void run(const size_t tolerance_bits = DEFAULT_FLOAT_TOLERANCE_BITS) {
        m_request.infer();
        const auto res = compare_results(tolerance_bits);

        if (res != testing::AssertionSuccess()) {
            std::cout << res.message() << std::endl;
        }

        m_input_index = 0;
        m_output_index = 0;

        m_expected_outputs.clear();

        EXPECT_TRUE(res);
    }

    void run_with_tolerance_as_fp(const float tolerance = 1.0e-5f) {
        m_request.infer();
        const auto res = compare_results_with_tolerance_as_fp(tolerance);

        if (res != testing::AssertionSuccess()) {
            std::cout << res.message() << std::endl;
        }

        m_input_index = 0;
        m_output_index = 0;

        m_expected_outputs.clear();

        EXPECT_TRUE(res);
    }

private:
    std::shared_ptr<Function> m_function;
    ov::Core m_core;
    ov::InferRequest m_request;
    std::vector<ov::Tensor> m_expected_outputs;
    size_t m_input_index = 0;
    size_t m_output_index = 0;
    testing::AssertionResult compare_results(size_t tolerance_bits);
    testing::AssertionResult compare_results_with_tolerance_as_fp(float tolerance_bits);
};
}  // namespace test
}  // namespace ngraph
