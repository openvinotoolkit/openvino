//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "runtime/backend.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

template <typename T>
class BatchNormInferenceTester
{
public:
    BatchNormInferenceTester(const std::shared_ptr<ngraph::runtime::Backend>& backend,
                             const Shape& input_shape,
                             element::Type etype,
                             double epsilon)
        : m_backend(backend)
    {
        Shape channel_shape{input_shape.at(1)};

        auto Input = make_shared<op::Parameter>(etype, input_shape);
        auto Gamma = make_shared<op::Parameter>(etype, channel_shape);
        auto Beta = make_shared<op::Parameter>(etype, channel_shape);
        auto Mean = make_shared<op::Parameter>(etype, channel_shape);
        auto Variance = make_shared<op::Parameter>(etype, channel_shape);
        auto BN =
            make_shared<op::v5::BatchNormInference>(Input, Gamma, Beta, Mean, Variance, epsilon);
        m_function = make_shared<Function>(BN, ParameterVector{Input, Gamma, Beta, Mean, Variance});

        m_input = backend->create_tensor(etype, input_shape);
        m_gamma = backend->create_tensor(etype, channel_shape);
        m_beta = backend->create_tensor(etype, channel_shape);
        m_mean = backend->create_tensor(etype, channel_shape);
        m_variance = backend->create_tensor(etype, channel_shape);
        m_normed_input = backend->create_tensor(etype, input_shape);
    }

    bool call(const std::vector<T>& input,
              const std::vector<T>& gamma,
              const std::vector<T>& beta,
              const std::vector<T>& mean,
              const std::vector<T>& variance,
              const std::vector<T>& normed_input)
    {
        copy_data(m_input, input);
        copy_data(m_gamma, gamma);
        copy_data(m_beta, beta);
        copy_data(m_mean, mean);
        copy_data(m_variance, variance);
        auto handle = m_backend->compile(m_function);
        handle->call_with_validate({m_normed_input},
                                   {m_input, m_gamma, m_beta, m_mean, m_variance});
        auto res_normed_input = read_vector<T>(m_normed_input);
        return test::all_close(normed_input, res_normed_input);
    }

protected:
    const std::shared_ptr<ngraph::runtime::Backend>& m_backend;
    std::shared_ptr<Function> m_function;
    std::shared_ptr<ngraph::runtime::Tensor> m_input;
    std::shared_ptr<ngraph::runtime::Tensor> m_gamma;
    std::shared_ptr<ngraph::runtime::Tensor> m_beta;
    std::shared_ptr<ngraph::runtime::Tensor> m_mean;
    std::shared_ptr<ngraph::runtime::Tensor> m_variance;
    std::shared_ptr<ngraph::runtime::Tensor> m_normed_input;
};

template <typename T>
class BatchNormInferenceTesterZeroEpsilon : public BatchNormInferenceTester<T>
{
public:
    // These are for documentation purposes only below
    using Input = test::NDArray<T, 2>;
    using Gamma = test::NDArray<T, 1>;
    using Beta = test::NDArray<T, 1>;
    using Mean = test::NDArray<T, 1>;
    using Variance = test::NDArray<T, 1>;
    using NormedInput = test::NDArray<T, 2>;

    BatchNormInferenceTesterZeroEpsilon(const std::shared_ptr<ngraph::runtime::Backend>& backend,
                                        element::Type etype)
        : BatchNormInferenceTester<T>(backend, Shape{2, 3}, etype, 0.0)
    {
    }

    bool test(const Input& input,
              const Gamma& gamma,
              const Beta& beta,
              const Mean& mean,
              const Variance& variance,
              const NormedInput& normed_input)
    {
        return BatchNormInferenceTester<T>::call(input.get_vector(),
                                                 gamma.get_vector(),
                                                 beta.get_vector(),
                                                 mean.get_vector(),
                                                 variance.get_vector(),
                                                 normed_input.get_vector());
    }
    bool test_gamma()
    {
        return test(Input{{1.0, 2.0, 3.0}, {-1.0, -2.0, -3.0}},
                    Gamma{2.0, 3.0, 4.0},
                    Beta{0.0, 0.0, 0.0},
                    Mean{0.0, 0.0, 0.0},
                    Variance{1.0, 1.0, 1.0},
                    NormedInput{{2.0, 6.0, 12.0}, {-2.0, -6.0, -12.0}});
    }
    bool test_beta()
    {
        return test(Input{{1.0, 2.0, 3.0}, {-1.0, -2.0, -3.0}},
                    Gamma{1.0, 1.0, 1.0},
                    Beta{2.0, -2.0, 3.0},
                    Mean{0.0, 0.0, 0.0},
                    Variance{1.0, 1.0, 1.0},
                    NormedInput{{3.0, 0.0, 6.0}, {1.0, -4.0, 0.0}});
    }
    bool test_mean()
    {
        return test(Input{{1.0, 2.0, 3.0}, {-1.0, -2.0, -3.0}},
                    Gamma{1.0, 1.0, 1.0},
                    Beta{0.0, 0.0, 0.0},
                    Mean{-2.0, 2.0, -3.0},
                    Variance{1.0, 1.0, 1.0},
                    NormedInput{{3.0, 0.0, 6.0}, {1.0, -4.0, 0.0}});
    }
    bool test_variance()
    {
        return test(Input{{1.0, 2.0, 3.0}, {-1.0, -2.0, -3.0}},
                    Gamma{1.0, 1.0, 1.0},
                    Beta{0.0, 0.0, 0.0},
                    Mean{0.0, 0.0, 0.0},
                    Variance{0.25, .0625, 4.0},
                    NormedInput{{2.0, 8.0, 1.5}, {-2.0, -8.0, -1.5}});
    }
};

NGRAPH_TEST(${BACKEND_NAME}, batch_norm_inference_0eps_f64)
{
    using T = double;
    auto& et = element::f64;
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    BatchNormInferenceTesterZeroEpsilon<T> bnt(backend, et);
    EXPECT_TRUE(bnt.test_gamma()) << "Gamma test";
    EXPECT_TRUE(bnt.test_beta()) << "Beta test";
    EXPECT_TRUE(bnt.test_mean()) << "Mean test";
    EXPECT_TRUE(bnt.test_variance()) << "Variance test";
}

NGRAPH_TEST(${BACKEND_NAME}, batch_norm_inference_0eps_f32)
{
    using T = float;
    auto& et = element::f32;
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    BatchNormInferenceTesterZeroEpsilon<T> bnt(backend, et);
    EXPECT_TRUE(bnt.test_gamma()) << "Gamma test";
    EXPECT_TRUE(bnt.test_beta()) << "Beta test";
    EXPECT_TRUE(bnt.test_mean()) << "Mean test";
    EXPECT_TRUE(bnt.test_variance()) << "Variance test";
}

template <typename T>
class BatchNormInferenceTesterNonZeroEpsilon : public BatchNormInferenceTester<T>
{
public:
    // These are for documentation purposes only below
    using Input = test::NDArray<T, 2>;
    using Gamma = test::NDArray<T, 1>;
    using Beta = test::NDArray<T, 1>;
    using Mean = test::NDArray<T, 1>;
    using Variance = test::NDArray<T, 1>;
    using NormedInput = test::NDArray<T, 2>;

    BatchNormInferenceTesterNonZeroEpsilon(const std::shared_ptr<ngraph::runtime::Backend>& backend,
                                           element::Type etype)
        : BatchNormInferenceTester<T>(backend, Shape{2, 3}, etype, 0.25)
    {
    }

    bool test(const Input& input,
              const Gamma& gamma,
              const Beta& beta,
              const Mean& mean,
              const Variance& variance,
              const NormedInput& normed_input)
    {
        return BatchNormInferenceTester<T>::call(input.get_vector(),
                                                 gamma.get_vector(),
                                                 beta.get_vector(),
                                                 mean.get_vector(),
                                                 variance.get_vector(),
                                                 normed_input.get_vector());
    }
    bool test_gamma()
    {
        return test(Input{{1.0, 2.0, 3.0}, {-1.0, -2.0, -3.0}},
                    Gamma{2.0, 3.0, 4.0},
                    Beta{0.0, 0.0, 0.0},
                    Mean{0.0, 0.0, 0.0},
                    Variance{0.75, 0.75, 0.75},
                    NormedInput{{2.0, 6.0, 12.0}, {-2.0, -6.0, -12.0}});
    }
    bool test_beta()
    {
        return test(Input{{1.0, 2.0, 3.0}, {-1.0, -2.0, -3.0}},
                    Gamma{1.0, 1.0, 1.0},
                    Beta{2.0, -2.0, 3.0},
                    Mean{0.0, 0.0, 0.0},
                    Variance{0.75, 0.75, 0.75},
                    NormedInput{{3.0, 0.0, 6.0}, {1.0, -4.0, 0.0}});
    }
    bool test_mean()
    {
        return test(Input{{1.0, 2.0, 3.0}, {-1.0, -2.0, -3.0}},
                    Gamma{1.0, 1.0, 1.0},
                    Beta{0.0, 0.0, 0.0},
                    Mean{-2.0, 2.0, -3.0},
                    Variance{0.75, 0.75, 0.75},
                    NormedInput{{3.0, 0.0, 6.0}, {1.0, -4.0, 0.0}});
    }
    bool test_variance()
    {
        return test(Input{{3.0, 5.0, 1.0}, {-3.0, -5.0, -1.0}},
                    Gamma{1.0, 1.0, 1.0},
                    Beta{0.0, 0.0, 0.0},
                    Mean{0.0, 0.0, 0.0},
                    Variance{2.0, 6.0, 0.0},
                    NormedInput{{2.0, 2.0, 2.0}, {-2.0, -2.0, -2.0}});
    }
};

NGRAPH_TEST(${BACKEND_NAME}, batch_norm_inference_f64)
{
    using T = double;
    auto& et = element::f64;
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    BatchNormInferenceTesterNonZeroEpsilon<T> bnt(backend, et);
    EXPECT_TRUE(bnt.test_gamma()) << "Gamma test";
    EXPECT_TRUE(bnt.test_beta()) << "Beta test";
    EXPECT_TRUE(bnt.test_mean()) << "Mean test";
    EXPECT_TRUE(bnt.test_variance()) << "Variance test";
}

NGRAPH_TEST(${BACKEND_NAME}, batch_norm_inference_f32)
{
    using T = float;
    auto& et = element::f32;
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    BatchNormInferenceTesterNonZeroEpsilon<T> bnt(backend, et);
    EXPECT_TRUE(bnt.test_gamma()) << "Gamma test";
    EXPECT_TRUE(bnt.test_beta()) << "Beta test";
    EXPECT_TRUE(bnt.test_mean()) << "Mean test";
    EXPECT_TRUE(bnt.test_variance()) << "Variance test";
}

NGRAPH_TEST(${BACKEND_NAME}, batch_norm_inference_parameters_duplication)
{
    auto input_shape = Shape{2, 2, 2, 1};
    auto input = make_shared<op::Parameter>(element::f32, input_shape);

    auto mvgb_shape = Shape{2};
    auto mvgb = make_shared<op::Parameter>(element::f32, mvgb_shape);

    double eps = 0.001;
    auto shape_r = Shape{2, 2, 2, 1};
    auto bn = make_shared<op::v0::BatchNormInference>(input, mvgb, mvgb, mvgb, mvgb, eps);

    auto f = make_shared<Function>(bn, ParameterVector{input, mvgb, mvgb, mvgb, mvgb});
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    // Create some tensors for input/output
    auto _input = backend->create_tensor(element::f32, input_shape);
    copy_data(_input,
              vector<float>{0.54881352f,
                            0.71518934f,
                            0.60276335f,
                            0.54488319f,
                            0.42365479f,
                            0.64589411f,
                            0.4375872f,
                            0.89177299f});

    auto _mvgb = backend->create_tensor(element::f32, mvgb_shape);
    copy_data(_mvgb, vector<float>{1.0f, 1.0f});
    auto bn_output = backend->create_tensor(element::f32, shape_r);

    vector<float> expected_result{0.54903894f,
                                  0.71533161f,
                                  0.60296183f,
                                  0.54511058f,
                                  0.42394274f,
                                  0.64607101f,
                                  0.43786817f,
                                  0.89182704f};
    auto handle = backend->compile(f);
    handle->call_with_validate({bn_output}, {_input, _mvgb, _mvgb, _mvgb, _mvgb});

    ASSERT_TRUE(
        ngraph::test::all_close(expected_result, read_vector<float>(bn_output), 1e-3f, 1e-4f));
}

NGRAPH_TEST(${BACKEND_NAME}, batch_norm_inference_parameters_duplication_v5)
{
    auto input_shape = Shape{2, 2, 2, 1};
    auto input = make_shared<op::Parameter>(element::f32, input_shape);

    auto mvgb_shape = Shape{2};
    auto mvgb = make_shared<op::Parameter>(element::f32, mvgb_shape);

    double eps = 0.001;
    auto shape_r = Shape{2, 2, 2, 1};
    auto bn = make_shared<op::v5::BatchNormInference>(input, mvgb, mvgb, mvgb, mvgb, eps);

    auto f = make_shared<Function>(bn, ParameterVector{input, mvgb, mvgb, mvgb, mvgb});
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    // Create some tensors for input/output
    auto _input = backend->create_tensor(element::f32, input_shape);
    copy_data(_input,
              vector<float>{0.54881352f,
                            0.71518934f,
                            0.60276335f,
                            0.54488319f,
                            0.42365479f,
                            0.64589411f,
                            0.4375872f,
                            0.89177299f});

    auto _mvgb = backend->create_tensor(element::f32, mvgb_shape);
    copy_data(_mvgb, vector<float>{1.0f, 1.0f});
    auto bn_output = backend->create_tensor(element::f32, shape_r);

    vector<float> expected_result{0.54903894f,
                                  0.71533161f,
                                  0.60296183f,
                                  0.54511058f,
                                  0.42394274f,
                                  0.64607101f,
                                  0.43786817f,
                                  0.89182704f};
    auto handle = backend->compile(f);
    handle->call_with_validate({bn_output}, {_input, _mvgb, _mvgb, _mvgb, _mvgb});

    ASSERT_TRUE(
        ngraph::test::all_close(expected_result, read_vector<float>(bn_output), 1e-3f, 1e-4f));
}

NGRAPH_TEST(${BACKEND_NAME}, batch_norm_fprop_inference_b2c2h2w1)
{
    auto input_shape = Shape{2, 2, 2, 1};
    auto input = make_shared<op::Parameter>(element::f32, input_shape);
    auto gamma_shape = Shape{2};
    auto gamma = make_shared<op::Parameter>(element::f32, gamma_shape);
    auto beta_shape = Shape{2};
    auto beta = make_shared<op::Parameter>(element::f32, beta_shape);
    auto mean_shape = Shape{2};
    auto mean = make_shared<op::Parameter>(element::f32, mean_shape);
    auto var_shape = Shape{2};
    auto var = make_shared<op::Parameter>(element::f32, var_shape);
    double eps = 0.001;
    auto shape_r = Shape{2, 2, 2, 1};
    auto bn = make_shared<op::v0::BatchNormInference>(input, gamma, beta, mean, var, eps);

    auto f = make_shared<Function>(bn, ParameterVector{input, gamma, beta, mean, var});
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    // Create some tensors for input/output
    auto _input = backend->create_tensor(element::f32, input_shape);
    copy_data(_input,
              vector<float>{0.54881352f,
                            0.71518934f,
                            0.60276335f,
                            0.54488319f,
                            0.42365479f,
                            0.64589411f,
                            0.4375872f,
                            0.89177299f});

    auto _gamma = backend->create_tensor(element::f32, gamma_shape);
    copy_data(_gamma, vector<float>{1.0f, 1.0f});
    auto _beta = backend->create_tensor(element::f32, beta_shape);
    copy_data(_beta, vector<float>{0.0f, 0.0f});
    auto _mean = backend->create_tensor(element::f32, mean_shape);
    copy_data(_mean, vector<float>{0.583388f, 0.619252f});
    auto _var = backend->create_tensor(element::f32, var_shape);
    copy_data(_var, vector<float>{0.0119972f, 0.0282681f});
    auto bn_output = backend->create_tensor(element::f32, shape_r);

    vector<float> expected_result{
        -0.30327f, 1.1561f, -0.0963782f, -0.434702f, -1.4011f, 0.548275f, -1.06187f, 1.59295f};
    auto handle = backend->compile(f);
    handle->call_with_validate({bn_output}, {_input, _gamma, _beta, _mean, _var});

    ASSERT_TRUE(
        ngraph::test::all_close(expected_result, read_vector<float>(bn_output), 1e-3f, 1e-4f));
}

NGRAPH_TEST(${BACKEND_NAME}, batch_norm_fprop_inference_b2c2h2w1_v5)
{
    auto input_shape = Shape{2, 2, 2, 1};
    auto input = make_shared<op::Parameter>(element::f32, input_shape);
    auto gamma_shape = Shape{2};
    auto gamma = make_shared<op::Parameter>(element::f32, gamma_shape);
    auto beta_shape = Shape{2};
    auto beta = make_shared<op::Parameter>(element::f32, beta_shape);
    auto mean_shape = Shape{2};
    auto mean = make_shared<op::Parameter>(element::f32, mean_shape);
    auto var_shape = Shape{2};
    auto var = make_shared<op::Parameter>(element::f32, var_shape);
    double eps = 0.001;
    auto shape_r = Shape{2, 2, 2, 1};
    auto bn = make_shared<op::v5::BatchNormInference>(input, gamma, beta, mean, var, eps);

    auto f = make_shared<Function>(bn, ParameterVector{input, gamma, beta, mean, var});
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    // Create some tensors for input/output
    auto _input = backend->create_tensor(element::f32, input_shape);
    copy_data(_input,
              vector<float>{0.54881352f,
                            0.71518934f,
                            0.60276335f,
                            0.54488319f,
                            0.42365479f,
                            0.64589411f,
                            0.4375872f,
                            0.89177299f});

    auto _gamma = backend->create_tensor(element::f32, gamma_shape);
    copy_data(_gamma, vector<float>{1.0f, 1.0f});
    auto _beta = backend->create_tensor(element::f32, beta_shape);
    copy_data(_beta, vector<float>{0.0f, 0.0f});
    auto _mean = backend->create_tensor(element::f32, mean_shape);
    copy_data(_mean, vector<float>{0.583388f, 0.619252f});
    auto _var = backend->create_tensor(element::f32, var_shape);
    copy_data(_var, vector<float>{0.0119972f, 0.0282681f});
    auto bn_output = backend->create_tensor(element::f32, shape_r);

    vector<float> expected_result{
        -0.30327f, 1.1561f, -0.0963782f, -0.434702f, -1.4011f, 0.548275f, -1.06187f, 1.59295f};
    auto handle = backend->compile(f);
    handle->call_with_validate({bn_output}, {_input, _gamma, _beta, _mean, _var});

    ASSERT_TRUE(
        ngraph::test::all_close(expected_result, read_vector<float>(bn_output), 1e-3f, 1e-4f));
}
