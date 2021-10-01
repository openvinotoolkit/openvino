// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif
#ifdef ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#define DEFAULT_DOUBLE_TOLERANCE_BITS ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#endif
// clang-format on

#include "onnx_import/onnx.hpp"
#include "default_opset.hpp"
#include <test_case.hpp>
#include <test_control.hpp>
#include <engine/test_engines.hpp>

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace ngraph;

static std::string s_manifest = "${MANIFEST}";

using Inputs = std::vector<std::vector<float>>;
using Outputs = std::vector<std::vector<float>>;

using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_bias_gelu) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/bias_gelu.onnx"));

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({0.5488135,
                                0.71518934,
                                0.60276335,
                                0.5448832,
                                0.4236548,
                                0.6458941,
                                0.4375872,
                                0.891773,
                                0.96366274,
                                0.3834415});
    test_case.add_input<float>({0.79172504, 0.5288949, 0.56804454, 0.92559665, 0.07103606});
    test_case.add_expected_output<float>(
        {1.2198428, 1.1112978, 1.0293297, 1.366493, 0.3411342, 1.329408, 0.8051748, 1.354462, 1.8336612, 0.3068893});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_skip_layer_normalization_with_gamma_beta_bias) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/skip_layer_normalization_with_gamma_beta_bias.onnx"));

    std::vector<float> input = {
        0.54881352, 0.71518934, 0.60276335, 0.54488319, 0.42365479, 0.64589411, 0.43758720, 0.89177299,
        0.96366274, 0.38344151, 0.79172504, 0.52889490, 0.56804454, 0.92559665, 0.07103606, 0.08712930,
        0.02021840, 0.83261985, 0.77815676, 0.87001216, 0.97861832, 0.79915857, 0.46147937, 0.78052920,
    };
    std::vector<float> skip = {
        0.11827443, 0.63992101, 0.14335328, 0.94466889, 0.52184832, 0.41466194, 0.26455560, 0.77423370,
        0.45615032, 0.56843394, 0.01878980, 0.61763549, 0.61209571, 0.61693400, 0.94374806, 0.68182027,
        0.35950789, 0.43703195, 0.69763118, 0.06022547, 0.66676670, 0.67063785, 0.21038257, 0.12892629,
    };
    std::vector<float> expected = {
        -0.19721794, -0.42944565, 0.18620640, 0.61282152,  -0.11097327, -0.59518522, 0.13393641,  0.66901535,
        0.04256713,  -0.71902490, 0.23107991, 0.17300847,  -0.04390603, -0.31109563, 0.51021838,  -0.66914201,
        -0.20009395, -0.43313017, 0.67281967, -0.01712347, 0.09767530,  -0.43024653, -0.01836969, -0.29238200,
    };
    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>(input);
    test_case.add_input<float>(skip);
    test_case.add_expected_output<float>(expected);
    test_case.run(5);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_skip_layer_normalization_with_gamma_beta) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/skip_layer_normalization_with_gamma_beta.onnx"));

    std::vector<float> input = {
        0.54881352, 0.71518934, 0.60276335, 0.54488319, 0.42365479, 0.64589411, 0.43758720, 0.89177299,
        0.96366274, 0.38344151, 0.79172504, 0.52889490, 0.56804454, 0.92559665, 0.07103606, 0.08712930,
        0.02021840, 0.83261985, 0.77815676, 0.87001216, 0.97861832, 0.79915857, 0.46147937, 0.78052920,
    };
    std::vector<float> skip = {
        0.11827443, 0.63992101, 0.14335328, 0.94466889, 0.52184832, 0.41466194, 0.26455560, 0.77423370,
        0.45615032, 0.56843394, 0.01878980, 0.61763549, 0.61209571, 0.61693400, 0.94374806, 0.68182027,
        0.35950789, 0.43703195, 0.69763118, 0.06022547, 0.66676670, 0.67063785, 0.21038257, 0.12892629,
    };
    std::vector<float> expected = {
        -0.17974678, -0.23946194, -0.04376268, 0.46959469,  -0.11171167, -0.41859278, -0.11082965, 0.64513868,
        0.07773457,  -0.51403606, -0.13661698, 0.11262375,  -0.05096011, -0.10416907, 0.10070466,  -0.50876135,
        -0.22290939, -0.27663514, 0.55416691,  -0.08064821, 0.04857478,  -0.25121087, -0.15912610, -0.26637587,
    };
    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>(input);
    test_case.add_input<float>(skip);
    test_case.add_expected_output<float>(expected);
    test_case.run(7);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_skip_layer_normalization_with_gamma) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/skip_layer_normalization_with_gamma.onnx"));

    std::vector<float> input = {
        0.54881352, 0.71518934, 0.60276335, 0.54488319, 0.42365479, 0.64589411, 0.43758720, 0.89177299,
        0.96366274, 0.38344151, 0.79172504, 0.52889490, 0.56804454, 0.92559665, 0.07103606, 0.08712930,
        0.02021840, 0.83261985, 0.77815676, 0.87001216, 0.97861832, 0.79915857, 0.46147937, 0.78052920,
    };
    std::vector<float> skip = {
        0.11827443, 0.63992101, 0.14335328, 0.94466889, 0.52184832, 0.41466194, 0.26455560, 0.77423370,
        0.45615032, 0.56843394, 0.01878980, 0.61763549, 0.61209571, 0.61693400, 0.94374806, 0.68182027,
        0.35950789, 0.43703195, 0.69763118, 0.06022547, 0.66676670, 0.67063785, 0.21038257, 0.12892629,
    };
    std::vector<float> expected = {
        -0.10974677, 0.16053806,  -0.26376268, 0.46959469,  -0.04171166, -0.01859277, -0.33082965, 0.64513868,
        0.14773457,  -0.11403608, -0.35661697, 0.11262375,  0.01903989,  0.29583094,  -0.11929534, -0.50876135,
        -0.15290938, 0.12336487,  0.33416691,  -0.08064821, 0.11857478,  0.14878914,  -0.37912610, -0.26637587,
    };
    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>(input);
    test_case.add_input<float>(skip);
    test_case.add_expected_output<float>(expected);
    test_case.run(6);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_skip_layer_normalization_dynamic_shapes) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/dynamic_shapes/skip_layer_normalization.onnx"));

    std::vector<float> input = {
        0.54881352, 0.71518934, 0.60276335, 0.54488319, 0.42365479, 0.64589411, 0.43758720, 0.89177299,
        0.96366274, 0.38344151, 0.79172504, 0.52889490, 0.56804454, 0.92559665, 0.07103606, 0.08712930,
        0.02021840, 0.83261985, 0.77815676, 0.87001216, 0.97861832, 0.79915857, 0.46147937, 0.78052920,
    };
    std::vector<float> skip = {
        0.11827443, 0.63992101, 0.14335328, 0.94466889, 0.52184832, 0.41466194, 0.26455560, 0.77423370,
        0.45615032, 0.56843394, 0.01878980, 0.61763549, 0.61209571, 0.61693400, 0.94374806, 0.68182027,
        0.35950789, 0.43703195, 0.69763118, 0.06022547, 0.66676670, 0.67063785, 0.21038257, 0.12892629,
    };
    std::vector<float> gamma = {
        0.31542835,
        0.36371076,
        0.57019675,
        0.43860152,
    };
    std::vector<float> beta = {
        0.98837382,
        0.10204481,
        0.20887676,
        0.16130951,
    };
    std::vector<float> bias = {
        0.65310830,
        0.25329161,
        0.46631077,
        0.24442559,
    };
    std::vector<float> expected = {
        0.76600611, 0.34308332,  -0.48470584, 0.71335256,  1.10028172, -0.13354334, -0.45232186, 0.79840088,
        1.52454257, -0.19450217, -0.13759643, 0.03988872,  1.27861762, 0.39529073,  0.12247884,  -0.52944231,
        0.64228040, 0.21059875,  1.05966032,  -0.14278713, 1.46366918, 0.21215858,  -0.31640187, -0.22832340,
    };

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(function);
    test_case.add_input<float>(Shape{3, 2, 4}, input);
    test_case.add_input<float>(Shape{3, 2, 4}, skip);
    test_case.add_input<float>(Shape{4}, gamma);
    test_case.add_input<float>(Shape{4}, beta);
    test_case.add_input<float>(Shape{4}, bias);
    test_case.add_expected_output<float>(Shape{3, 2, 4}, expected);
    test_case.run(7);
}
