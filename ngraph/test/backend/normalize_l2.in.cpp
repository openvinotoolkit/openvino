// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/random.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"
#include "util/engine/test_engines.hpp"


using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

static void normalize_l2_results_test(std::vector<float>& data, Shape& data_shape, std::vector<int32_t>& axes, ngraph::op::EpsMode eps_mode, float eps, std::vector<float>& expected_output)
{
    auto data_input = std::make_shared<op::Parameter>(element::f32, data_shape);
    const auto axes_input = std::make_shared<op::Constant>(element::i32, Shape{axes.size()}, axes);

    auto normalize = std::make_shared<op::v0::NormalizeL2>(data_input, axes_input, eps, eps_mode);
    auto function = std::make_shared<Function>(normalize, ParameterVector{data_input});

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(data_shape, expected_output);

    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 4);
}

// ----------------------- eps_mode = ngraph::op::EpsMode::ADD ----------------------- //

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_across_all_2d_add)
{
    std::vector<float> data{1, 2, 3, 4};
    Shape data_shape{2, 2};
    std::vector<int32_t> axes{0, 1};
    float eps = 1e-7;
    auto eps_mode = ngraph::op::EpsMode::ADD;
    std::vector<float> expected_output{0.18257418, 0.36514837, 0.5477226, 0.73029673};

    normalize_l2_results_test(data, data_shape, axes, eps_mode, eps, expected_output);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_across_empty_2d_add)
{
    std::vector<float> data{0, 3, 0, 8};
    Shape data_shape{2, 2};
    std::vector<int32_t> axes{};
    float eps = 1e-7;
    auto eps_mode = ngraph::op::EpsMode::ADD;
    std::vector<float> expected_output{0, 1, 0, 1};

    normalize_l2_results_test(data, data_shape, axes, eps_mode, eps, expected_output);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_across_0_2d_add)
{
    std::vector<float> data{1, 2, 3, 4};
    Shape data_shape{2, 2};
    std::vector<int32_t> axes{0};
    float eps = 1e-7;
    auto eps_mode = ngraph::op::EpsMode::ADD;
    std::vector<float> expected_output{0.31622776, 0.4472136, 0.94868326, 0.8944272};

    normalize_l2_results_test(data, data_shape, axes, eps_mode, eps, expected_output);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_across_1_3d_add)
{
    std::vector<float> data{1, 2, 3, 4};
    Shape data_shape{1, 2, 2};
    std::vector<int32_t> axes{1};
    float eps = 1e-7;
    auto eps_mode = ngraph::op::EpsMode::ADD;
    std::vector<float> expected_output{0.31622776, 0.4472136, 0.94868326, 0.8944272};

    normalize_l2_results_test(data, data_shape, axes, eps_mode, eps, expected_output);

}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_across_1_2d_add)
{
    std::vector<float> data{1, 2, 3, 4};
    Shape data_shape{2, 2};
    std::vector<int32_t> axes{1};
    float eps = 1e-7;
    auto eps_mode = ngraph::op::EpsMode::ADD;
    std::vector<float> expected_output{0.4472136, 0.8944272, 0.6, 0.8};

    normalize_l2_results_test(data, data_shape, axes, eps_mode, eps, expected_output);

}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_across_2_3d_add)
{
    std::vector<float> data{1, 2, 3, 4};
    Shape data_shape{1, 2, 2};
    std::vector<int32_t> axes{2};
    float eps = 1e-7;
    auto eps_mode = ngraph::op::EpsMode::ADD;
    std::vector<float> expected_output{0.4472136, 0.8944272, 0.6, 0.8};

    normalize_l2_results_test(data, data_shape, axes, eps_mode, eps, expected_output);
}

// ----------------------- eps_mode = ngraph::op::EpsMode::MAX ----------------------- //

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_across_all_2d_max)
{
    std::vector<float> data{1, 2, 3, 4};
    Shape data_shape{2, 2};
    std::vector<int32_t> axes{0, 1};
    float eps = 1e-7;
    auto eps_mode = ngraph::op::EpsMode::MAX;
    std::vector<float> expected_output{0.18257419, 0.36514837, 0.54772256, 0.73029674};

    normalize_l2_results_test(data, data_shape, axes, eps_mode, eps, expected_output);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_across_empty_2d_max)
{
    std::vector<float> data{0, 3, 0, 8};
    Shape data_shape{2, 2};
    std::vector<int32_t> axes{};
    float eps = 1e-7;
    auto eps_mode = ngraph::op::EpsMode::MAX;
    std::vector<float> expected_output{0, 1, 0, 1};

    normalize_l2_results_test(data, data_shape, axes, eps_mode, eps, expected_output);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_across_0_2d_max)
{
    std::vector<float> data{1, 2, 3, 4};
    Shape data_shape{2, 2};
    std::vector<int32_t> axes{0};
    float eps = 1e-7;
    auto eps_mode = ngraph::op::EpsMode::MAX;
    std::vector<float> expected_output{0.31622777, 0.4472136, 0.9486833, 0.89442719};

    normalize_l2_results_test(data, data_shape, axes, eps_mode, eps, expected_output);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_across_1_2d_max)
{
    std::vector<float> data{1, 2, 3, 4};
    Shape data_shape{2, 2};
    std::vector<int32_t> axes{1};
    float eps = 1e-7;
    auto eps_mode = ngraph::op::EpsMode::MAX;
    std::vector<float> expected_output{0.4472136, 0.89442719, 0.6, 0.8};

    normalize_l2_results_test(data, data_shape, axes, eps_mode, eps, expected_output);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_across_123_4d_add)
{
    Shape data_shape{1, 2, 3, 4};
    std::vector<float> data(shape_size(data_shape));
    iota(begin(data), end(data), 1);
    std::vector<int32_t> axes{1, 2, 3};
    float eps = 1e-7;
    auto eps_mode = ngraph::op::EpsMode::ADD;
    std::vector<float> expected_output{0.01428571f, 0.02857143f, 0.04285714f, 0.05714286f, 0.07142857f, 0.08571429f,
                     0.1f,        0.11428571f, 0.12857144f, 0.14285715f, 0.15714286f, 0.17142858f,
                     0.18571429f, 0.2f,        0.21428572f, 0.22857143f, 0.24285714f, 0.25714287f,
                     0.27142859f, 0.2857143f,  0.30000001f, 0.31428573f, 0.32857144f, 0.34285715f};

    normalize_l2_results_test(data, data_shape, axes, eps_mode, eps, expected_output);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_across_empty_4D_add)
{
    Shape data_shape{1, 2, 3, 4};
    std::vector<float> data(shape_size(data_shape));
    iota(begin(data), end(data), 1);
    std::vector<int32_t> axes{};
    float eps = 1e-7;
    auto eps_mode = ngraph::op::EpsMode::ADD;
    std::vector<float> expected_output(shape_size(data_shape), 1);

    normalize_l2_results_test(data, data_shape, axes, eps_mode, eps, expected_output);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_across_empty_4D_max)
{
    Shape data_shape{1, 2, 3, 4};
    std::vector<float> data(shape_size(data_shape));
    iota(begin(data), end(data), 1);
    std::vector<int32_t> axes{};
    float eps = 1e-7;
    auto eps_mode = ngraph::op::EpsMode::MAX;
    std::vector<float> expected_output(shape_size(data_shape), 1);

    normalize_l2_results_test(data, data_shape, axes, eps_mode, eps, expected_output);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_across_1_4d_add)
{
    Shape data_shape{1, 2, 3, 4};
    std::vector<float> data(shape_size(data_shape));
    iota(begin(data), end(data), 1);
    std::vector<int32_t> axes{1};
    float eps = 1e-6f;
    auto eps_mode = ngraph::op::EpsMode::ADD;
    std::vector<float> expected_output{0.0766965f,  0.14142136f, 0.19611613f, 0.24253564f, 0.28216633f, 0.31622776f,
                     0.34570536f, 0.37139067f, 0.39391932f, 0.41380295f, 0.43145549f, 0.44721359f,
                     0.99705452f, 0.98994946f, 0.98058069f, 0.97014254f, 0.95936549f, 0.94868332f,
                     0.93834311f, 0.92847669f, 0.91914505f, 0.91036648f, 0.90213418f, 0.89442718f};

    normalize_l2_results_test(data, data_shape, axes, eps_mode, eps, expected_output);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_across_1_5d_add)
{
    Shape data_shape{1, 2, 2, 2, 3};
    std::vector<float> data(shape_size(data_shape));
    iota(begin(data), end(data), 1);
    std::vector<int32_t> axes{1};
    float eps = 1e-6f;
    auto eps_mode = ngraph::op::EpsMode::ADD;
    std::vector<float> expected_output{0.0766965f,  0.14142136f, 0.19611613f, 0.24253564f, 0.28216633f, 0.31622776f,
                     0.34570536f, 0.37139067f, 0.39391932f, 0.41380295f, 0.43145549f, 0.44721359f,
                     0.99705452f, 0.98994946f, 0.98058069f, 0.97014254f, 0.95936549f, 0.94868332f,
                     0.93834311f, 0.92847669f, 0.91914505f, 0.91036648f, 0.90213418f, 0.89442718f};

    normalize_l2_results_test(data, data_shape, axes, eps_mode, eps, expected_output);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_across_123_5d_add)
{
    Shape data_shape{1, 2, 2, 2, 3};
    std::vector<float> data(shape_size(data_shape));
    iota(begin(data), end(data), 1);
    std::vector<int32_t> axes{1, 2, 3};
    float eps = 1e-6f;
    auto eps_mode = ngraph::op::EpsMode::ADD;
    std::vector<float> expected_output{0.026389, 0.0495682, 0.070014, 0.105556, 0.12392, 0.140028, 0.184723, 0.198273, 0.210042, 0.26389, 0.272625, 0.280056, 0.343057, 0.346977, 0.35007, 0.422224, 0.421329, 0.420084, 0.501391, 0.495682, 0.490098, 0.580558, 0.570034, 0.560112};

    normalize_l2_results_test(data, data_shape, axes, eps_mode, eps, expected_output);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_across_1_2x2_add)
{
    Shape data_shape{2, 2};
    std::vector<float> data(shape_size(data_shape));
    iota(begin(data), end(data), 1);
    std::vector<int32_t> axes{1};
    float eps = 1e-6f;
    auto eps_mode = ngraph::op::EpsMode::ADD;
    std::vector<float> expected_output{0.44721353f, 0.89442706f, 0.60000002f, 0.80000001f};

    normalize_l2_results_test(data, data_shape, axes, eps_mode, eps, expected_output);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_across_1_2x4_add)
{
    Shape data_shape{2, 4};
    std::vector<float> data(shape_size(data_shape));
    iota(begin(data), end(data), 1);
    std::vector<int32_t> axes{1};
    float eps = 1e-6f;
    auto eps_mode = ngraph::op::EpsMode::ADD;
    std::vector<float> expected_output {0.18257418f,
                                          0.36514837f,
                                          0.54772252f,
                                          0.73029673f,
                                          0.37904903f,
                                          0.45485884f,
                                          0.53066862f,
                                          0.60647845f};

    normalize_l2_results_test(data, data_shape, axes, eps_mode, eps, expected_output);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_across_123_4d_max)
{
    Shape data_shape{2, 4};
    std::vector<float> data(shape_size(data_shape));
    iota(begin(data), end(data), 1);
    std::vector<int32_t> axes{1};
    float eps = 1e-6f;
    auto eps_mode = ngraph::op::EpsMode::ADD;
    std::vector<float> expected_output{0.18257418f,
                                          0.36514837f,
                                          0.54772252f,
                                          0.73029673f,
                                          0.37904903f,
                                          0.45485884f,
                                          0.53066862f,
                                          0.60647845f};

    normalize_l2_results_test(data, data_shape, axes, eps_mode, eps, expected_output);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_across_123_4d_max_test)
{
    Shape data_shape{1, 2, 3, 4};
    std::vector<float> data(shape_size(data_shape));
    iota(begin(data), end(data), 1);
    std::vector<int32_t> axes{1, 2, 3};
    float eps = 1e-6f;
    auto eps_mode = ngraph::op::EpsMode::MAX;
    std::vector<float> expected_output{0.0142857, 0.0285714, 0.0428571, 0.0571429, 0.0714286, 0.0857143, 0.1, 0.114286, 0.128571, 0.142857, 0.157143, 0.171429, 0.185714, 0.2, 0.214286, 0.228571, 0.242857, 0.257143, 0.271429, 0.285714, 0.3, 0.314286, 0.328571, 0.342857};

    normalize_l2_results_test(data, data_shape, axes, eps_mode, eps, expected_output);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_l2_across_1_4d_max_test)
{
    Shape data_shape{2, 2, 3, 4};
    std::vector<float> data(shape_size(data_shape));
    iota(begin(data), end(data), 1);
    std::vector<int32_t> axes{1};
    float eps = 1e-6f;
    auto eps_mode = ngraph::op::EpsMode::MAX;
    std::vector<float> expected_output{0.0766965, 0.141421, 0.196116, 0.242536, 0.282166, 0.316228, 0.345705, 0.371391, 0.393919, 0.413803, 0.431455, 0.447214, 0.997055, 0.98995, 0.980581, 0.970143, 0.959365, 0.948683, 0.938343, 0.928477, 0.919145, 0.910366, 0.902134, 0.894427, 0.559857, 0.564684, 0.56921, 0.573462, 0.577465, 0.581238, 0.584802, 0.588172, 0.591364, 0.594391, 0.597266, 0.6, 0.828589, 0.825307, 0.822192, 0.819232, 0.816416, 0.813733, 0.811176, 0.808736, 0.806405, 0.804176, 0.802043, 0.8};

    normalize_l2_results_test(data, data_shape, axes, eps_mode, eps, expected_output);
}
