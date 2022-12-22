// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_case.hpp"

#include "common_test_utils/file_utils.hpp"
#include "openvino/util/file_util.hpp"
#include "shared_utils.hpp"

namespace {
template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, testing::AssertionResult>::type
compare_values(const ov::Tensor& expected, const ov::Tensor& result, const size_t tolerance_bits) {
    return ngraph::test::all_close_f(expected, result, static_cast<int>(tolerance_bits));
}

testing::AssertionResult compare_with_fp_tolerance(const ov::Tensor& expected_tensor,
                                                   const ov::Tensor& result_tensor,
                                                   const float tolerance) {
    auto comparison_result = testing::AssertionSuccess();

    auto exp_host_t = std::make_shared<ngraph::HostTensor>(expected_tensor.get_element_type(),
                                                           expected_tensor.get_shape(),
                                                           expected_tensor.data());
    auto res_host_t = std::make_shared<ngraph::HostTensor>(result_tensor.get_element_type(),
                                                           result_tensor.get_shape(),
                                                           result_tensor.data());
    const auto expected = read_vector<float>(exp_host_t);
    const auto result = read_vector<float>(res_host_t);

    return ngraph::test::compare_with_tolerance(expected, result, tolerance);
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value, testing::AssertionResult>::type
compare_values(const ov::Tensor& expected, const ov::Tensor& result, const size_t) {
    return ov::test::all_close(expected, result);
}

// used for float16 and bfloat 16 comparisons
template <typename T>
typename std::enable_if<std::is_class<T>::value, testing::AssertionResult>::type
compare_values(const ov::Tensor& expected_tensor, const ov::Tensor& result_tensor, const size_t tolerance_bits) {
    auto exp_host_t = std::make_shared<ngraph::HostTensor>(expected_tensor.get_element_type(),
                                                           expected_tensor.get_shape(),
                                                           expected_tensor.data());
    auto res_host_t = std::make_shared<ngraph::HostTensor>(result_tensor.get_element_type(),
                                                           result_tensor.get_shape(),
                                                           result_tensor.data());
    const auto expected = read_vector<T>(exp_host_t);
    const auto result = read_vector<T>(res_host_t);

    // TODO: add testing infrastructure for float16 and bfloat16 to avoid cast to double
    std::vector<double> expected_double(expected.size());
    std::vector<double> result_double(result.size());

    NGRAPH_CHECK(expected.size() == result.size(), "Number of expected and computed results don't match");

    for (size_t i = 0; i < expected.size(); ++i) {
        expected_double[i] = static_cast<double>(expected[i]);
        result_double[i] = static_cast<double>(result[i]);
    }

    return ngraph::test::all_close_f(expected_double, result_double, static_cast<int>(tolerance_bits));
}
};  // namespace

namespace ngraph {
namespace test {
std::shared_ptr<Function> function_from_ir(const std::string& xml_path, const std::string& bin_path) {
    ov::Core c;
    return c.read_model(xml_path, bin_path);
}

testing::AssertionResult TestCase::compare_results(size_t tolerance_bits) {
    auto compare_results = testing::AssertionSuccess();
    for (size_t i = 0; i < m_expected_outputs.size(); i++) {
        const auto& result_tensor = m_request.get_output_tensor(i);
        const auto& exp_result = m_expected_outputs.at(i);

        const auto& element_type = result_tensor.get_element_type();
        const auto& res_shape = result_tensor.get_shape();
        const auto& exp_shape = exp_result.get_shape();

        if (exp_shape != res_shape) {
            compare_results = testing::AssertionFailure();
            compare_results << "Computed data shape(" << res_shape << ") does not match the expected shape("
                            << exp_shape << ") for output " << i << std::endl;
            break;
        }

        switch (element_type) {
        case ov::element::Type_t::f16:
            compare_results = compare_values<ov::float16>(exp_result, result_tensor, tolerance_bits);
            break;
        case ov::element::Type_t::bf16:
            compare_results = compare_values<ov::bfloat16>(exp_result, result_tensor, tolerance_bits);
            break;
        case element::Type_t::f32:
            compare_results = compare_values<float>(exp_result, result_tensor, tolerance_bits);
            break;
        case element::Type_t::f64:
            compare_results = compare_values<double>(exp_result, result_tensor, tolerance_bits);
            break;
        case element::Type_t::i8:
            compare_results = compare_values<int8_t>(exp_result, result_tensor, tolerance_bits);
            break;
        case element::Type_t::i16:
            compare_results = compare_values<int16_t>(exp_result, result_tensor, tolerance_bits);
            break;
        case element::Type_t::i32:
            compare_results = compare_values<int32_t>(exp_result, result_tensor, tolerance_bits);
            break;
        case element::Type_t::i64:
            compare_results = compare_values<int64_t>(exp_result, result_tensor, tolerance_bits);
            break;
        case element::Type_t::u8:
            compare_results = compare_values<uint8_t>(exp_result, result_tensor, tolerance_bits);
            break;
        case element::Type_t::u16:
            compare_results = compare_values<uint16_t>(exp_result, result_tensor, tolerance_bits);
            break;
        case element::Type_t::u32:
            compare_results = compare_values<uint32_t>(exp_result, result_tensor, tolerance_bits);
            break;
        case element::Type_t::u64:
            compare_results = compare_values<uint64_t>(exp_result, result_tensor, tolerance_bits);
            break;
        case element::Type_t::boolean:
            compare_results = compare_values<char>(exp_result, result_tensor, tolerance_bits);
            break;
        default:
            compare_results = testing::AssertionFailure()
                              << "Unsupported data type encountered in 'compare_results' method";
        }
        if (compare_results == testing::AssertionFailure())
            break;
    }
    return compare_results;
}

testing::AssertionResult TestCase::compare_results_with_tolerance_as_fp(float tolerance) {
    auto comparison_result = testing::AssertionSuccess();

    for (size_t i = 0; i < m_expected_outputs.size(); ++i) {
        const auto& result_tensor = m_request.get_output_tensor(i);
        const auto& exp_result = m_expected_outputs.at(i);
        const auto& element_type = result_tensor.get_element_type();

        const auto& expected_shape = exp_result.get_shape();
        const auto& result_shape = result_tensor.get_shape();

        if (expected_shape != result_shape) {
            comparison_result = testing::AssertionFailure();
            comparison_result << "Computed data shape(" << result_shape << ") does not match the expected shape("
                              << expected_shape << ") for output " << i << std::endl;
            break;
        }

        switch (element_type) {
        case element::Type_t::f32:
            comparison_result = compare_with_fp_tolerance(exp_result, result_tensor, tolerance);
            break;
        case element::Type_t::i32:
            comparison_result = compare_values<int32_t>(exp_result, result_tensor, 0);
            break;
        default:
            comparison_result = testing::AssertionFailure() << "Unsupported data type encountered in "
                                                               "'compare_results_with_tolerance_as_fp' method";
        }

        if (comparison_result == testing::AssertionFailure()) {
            break;
        }
    }

    return comparison_result;
}

TestCase::TestCase(const std::shared_ptr<Function>& function, const std::string& dev) : m_function{function} {
    try {
        // Register template plugin
        m_core.register_plugin(
            ov::util::make_plugin_library_name(CommonTestUtils::getExecutableDirectory(),
                                               std::string("openvino_template_plugin") + IE_BUILD_POSTFIX),
            "TEMPLATE");
    } catch (...) {
    }
    m_request = m_core.compile_model(function, dev).create_infer_request();
}

}  // namespace test
}  // namespace ngraph
