//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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
        auto BN = make_shared<op::BatchNormInference>(Input, Gamma, Beta, Mean, Variance, epsilon);
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

template <typename T>
class BatchNormTrainingTester
{
public:
    BatchNormTrainingTester(const std::shared_ptr<ngraph::runtime::Backend>& backend,
                            const Shape& input_shape,
                            element::Type etype,
                            double epsilon)
        : m_backend(backend)
    {
        Shape channel_shape{input_shape.at(1)};

        auto Input = make_shared<op::Parameter>(etype, input_shape);
        auto Gamma = make_shared<op::Parameter>(etype, channel_shape);
        auto Beta = make_shared<op::Parameter>(etype, channel_shape);
        auto BN = make_shared<op::BatchNormTraining>(Input, Gamma, Beta, epsilon);
        auto NormedInput = make_shared<op::Result>(make_shared<op::GetOutputElement>(BN, 0));
        auto Mean = make_shared<op::Result>(make_shared<op::GetOutputElement>(BN, 1));
        auto Variance = make_shared<op::Result>(make_shared<op::GetOutputElement>(BN, 2));
        m_function = make_shared<Function>(ResultVector{NormedInput, Mean, Variance},
                                           ParameterVector{Input, Gamma, Beta});

        m_input = backend->create_tensor(etype, input_shape);
        m_gamma = backend->create_tensor(etype, channel_shape);
        m_beta = backend->create_tensor(etype, channel_shape);
        m_normed_input = backend->create_tensor(etype, input_shape);
        m_mean = backend->create_tensor(etype, channel_shape);
        m_variance = backend->create_tensor(etype, channel_shape);
    }

    std::tuple<bool, bool, bool> call(const std::vector<T>& input,
                                      const std::vector<T>& gamma,
                                      const std::vector<T>& beta,
                                      const std::vector<T>& normed_input,
                                      const std::vector<T>& mean,
                                      const std::vector<T>& variance)
    {
        copy_data(m_input, input);
        copy_data(m_gamma, gamma);
        copy_data(m_beta, beta);
        auto handle = m_backend->compile(m_function);
        handle->call_with_validate({m_normed_input, m_mean, m_variance},
                                   {m_input, m_gamma, m_beta});
        auto res_normed_input = read_vector<T>(m_normed_input);
        bool normed_input_test = test::all_close(normed_input, res_normed_input);

        auto res_mean = read_vector<T>(m_mean);
        bool mean_test = test::all_close(mean, res_mean);

        auto res_variance = read_vector<T>(m_variance);
        bool variance_test = test::all_close(variance, res_variance);

        return std::tuple<bool, bool, bool>(normed_input_test, mean_test, variance_test);
    }

protected:
    const std::shared_ptr<ngraph::runtime::Backend>& m_backend;
    std::shared_ptr<Function> m_function;
    std::shared_ptr<ngraph::runtime::Tensor> m_input;
    std::shared_ptr<ngraph::runtime::Tensor> m_gamma;
    std::shared_ptr<ngraph::runtime::Tensor> m_beta;
    std::shared_ptr<ngraph::runtime::Tensor> m_normed_input;
    std::shared_ptr<ngraph::runtime::Tensor> m_mean;
    std::shared_ptr<ngraph::runtime::Tensor> m_variance;
};

template <typename T>
class BatchNormTrainingTesterZeroEpsilon : public BatchNormTrainingTester<T>
{
public:
    // These are for documentation purposes only below
    using Input = test::NDArray<T, 2>;
    using Gamma = test::NDArray<T, 1>;
    using Beta = test::NDArray<T, 1>;
    using NormedInput = test::NDArray<T, 2>;
    using Mean = test::NDArray<T, 1>;
    using Variance = test::NDArray<T, 1>;

    BatchNormTrainingTesterZeroEpsilon(const std::shared_ptr<ngraph::runtime::Backend>& backend,
                                       element::Type etype)
        : BatchNormTrainingTester<T>(backend, Shape{10, 3}, etype, 0.0)
    {
    }

    std::tuple<bool, bool, bool> test(const Input& input,
                                      const Gamma& gamma,
                                      const Beta& beta,
                                      const NormedInput& normed_input,
                                      const Mean& mean,
                                      const Variance& variance)
    {
        return BatchNormTrainingTester<T>::call(input.get_vector(),
                                                gamma.get_vector(),
                                                beta.get_vector(),
                                                normed_input.get_vector(),
                                                mean.get_vector(),
                                                variance.get_vector());
    }
    std::tuple<bool, bool, bool> test_mean_variance()
    {
        return test(Input{{0.0, 1.0, 0.0},
                          {1.0, 2.0, 0.25},
                          {1.0, 2.0, 0.25},
                          {3.0, 4.0, 0.75},
                          {3.0, 4.0, 0.75},
                          {0.0, 1.0, 0.0},
                          {-1.0, 0.0, -0.25},
                          {-1.0, 0.0, -0.25},
                          {-3.0, -2.0, -0.75},
                          {-3.0, -2.0, -0.75}},
                    Gamma{1.0, 1.0, 1.0},
                    Beta{0.0, 0.0, 0.0},
                    NormedInput{{0.0, 0.0, 0.0},
                                {0.5, 0.5, 0.5},
                                {0.5, 0.5, 0.5},
                                {1.5, 1.5, 1.5},
                                {1.5, 1.5, 1.5},
                                {0.0, 0.0, 0.0},
                                {-0.5, -0.5, -0.5},
                                {-0.5, -0.5, -0.5},
                                {-1.5, -1.5, -1.5},
                                {-1.5, -1.5, -1.5}},
                    Mean{0.0, 1.0, 0.0},
                    Variance{4.0, 4.0, 0.25});
    }
    std::tuple<bool, bool, bool> test_gamma_beta()
    {
        return test(Input{{0.0, 1.0, 0.0},
                          {1.0, 2.0, 0.25},
                          {1.0, 2.0, 0.25},
                          {3.0, 4.0, 0.75},
                          {3.0, 4.0, 0.75},
                          {0.0, 1.0, 0.0},
                          {-1.0, 0.0, -0.25},
                          {-1.0, 0.0, -0.25},
                          {-3.0, -2.0, -0.75},
                          {-3.0, -2.0, -0.75}},
                    Gamma{2.0, 1.0, 2.0},
                    Beta{0.0, 1.0, 1.0},
                    NormedInput{{0.0, 1.0, 1.0},
                                {1.0, 1.5, 2.0},
                                {1.0, 1.5, 2.0},
                                {3.0, 2.5, 4.0},
                                {3.0, 2.5, 4.0},
                                {0.0, 1.0, 1.0},
                                {-1.0, 0.5, 0.0},
                                {-1.0, 0.5, 0.0},
                                {-3.0, -0.5, -2.0},
                                {-3.0, -0.5, -2.0}},
                    Mean{0.0, 1.0, 0.0},
                    Variance{4.0, 4.0, 0.25});
    }
};

NGRAPH_TEST(${BACKEND_NAME}, batch_norm_training_0eps_f64)
{
    using T = double;
    auto& et = element::f64;
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    BatchNormTrainingTesterZeroEpsilon<T> bnt(backend, et);
    std::tuple<bool, bool, bool> result;
    result = bnt.test_mean_variance();
    EXPECT_TRUE(std::get<0>(result)) << "Mean variance test normed input";
    EXPECT_TRUE(std::get<1>(result)) << "Mean variance test mean";
    EXPECT_TRUE(std::get<2>(result)) << "Mean variance test variance";

    result = bnt.test_gamma_beta();
    EXPECT_TRUE(std::get<0>(result)) << "Gamma beta test normed input";
    EXPECT_TRUE(std::get<1>(result)) << "Gamma beta test mean";
    EXPECT_TRUE(std::get<2>(result)) << "Gamma test variance";
}

NGRAPH_TEST(${BACKEND_NAME}, batch_norm_training_0eps_f32)
{
    using T = float;
    auto& et = element::f32;
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    BatchNormTrainingTesterZeroEpsilon<T> bnt(backend, et);
    std::tuple<bool, bool, bool> result;
    result = bnt.test_mean_variance();
    EXPECT_TRUE(std::get<0>(result)) << "Mean variance test normed input";
    EXPECT_TRUE(std::get<1>(result)) << "Mean variance test mean";
    EXPECT_TRUE(std::get<2>(result)) << "Mean variance test variance";

    result = bnt.test_gamma_beta();
    EXPECT_TRUE(std::get<0>(result)) << "Gamma beta test normed input";
    EXPECT_TRUE(std::get<1>(result)) << "Gamma beta test mean";
    EXPECT_TRUE(std::get<2>(result)) << "Gamma beta test variance";
}

NGRAPH_TEST(${BACKEND_NAME}, batch_norm_inference_parameters_duplication)
{
    auto input_shape = Shape{2, 2, 2, 1};
    auto input = make_shared<op::Parameter>(element::f32, input_shape);

    auto mvgb_shape = Shape{2};
    auto mvgb = make_shared<op::Parameter>(element::f32, mvgb_shape);

    double eps = 0.001;
    auto shape_r = Shape{2, 2, 2, 1};
    auto bn = make_shared<op::BatchNormInference>(input, mvgb, mvgb, mvgb, mvgb, eps);

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

NGRAPH_TEST(${BACKEND_NAME}, batch_norm_fprop_b1c2h2w2)
{
    auto input_shape = Shape{1, 2, 2, 2};
    auto input = make_shared<op::Parameter>(element::f32, input_shape);
    auto mean_shape = Shape{2};
    auto var_shape = Shape{2};
    auto gamma_shape = Shape{2};
    auto gamma = make_shared<op::Parameter>(element::f32, gamma_shape);
    auto beta_shape = Shape{2};
    auto beta = make_shared<op::Parameter>(element::f32, beta_shape);
    double eps = 0.001;
    auto shape_r = Shape{1, 2, 2, 2};
    auto bn = make_shared<op::BatchNormTraining>(input, gamma, beta, eps);

    auto output_rt = std::make_shared<op::GetOutputElement>(bn, 0);
    auto mean_rt = std::make_shared<op::GetOutputElement>(bn, 1);
    auto variance_rt = std::make_shared<op::GetOutputElement>(bn, 2);

    auto f = make_shared<Function>(NodeVector{output_rt, mean_rt, variance_rt},
                                   ParameterVector{input, gamma, beta});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto _input = backend->create_tensor(element::f32, Shape{1, 2, 2, 2});

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
    auto bn_output = backend->create_tensor(element::f32, shape_r);
    auto result_mean = backend->create_tensor(element::f32, mean_shape);
    auto result_variance = backend->create_tensor(element::f32, var_shape);

    vector<float> expected_result{-0.71498716f,
                                  1.48388731f,
                                  -0.00196938f,
                                  -0.76693159f,
                                  -0.91316032f,
                                  0.23943391f,
                                  -0.84090298f,
                                  1.51462936f};
    vector<float> expected_mean{0.602912f, 0.599727f};
    vector<float> expected_variance{0.00472505f, 0.0361782f};

    auto handle = backend->compile(f);
    handle->call_with_validate({bn_output, result_mean, result_variance}, {_input, _gamma, _beta});

    EXPECT_TRUE(test::all_close(expected_result, read_vector<float>(bn_output), 1e-5f, 1e-6f));
    EXPECT_TRUE(test::all_close(expected_mean, read_vector<float>(result_mean), 1e-5f, 1e-6f));
    EXPECT_TRUE(
        test::all_close(expected_variance, read_vector<float>(result_variance), 1e-5f, 1e-6f));
}

NGRAPH_TEST(${BACKEND_NAME}, batch_norm_fprop_b2c2h2w1)
{
    auto input_shape = Shape{2, 2, 2, 1};
    auto input = make_shared<op::Parameter>(element::f32, input_shape);
    auto mean_shape = Shape{2};
    auto var_shape = Shape{2};
    auto gamma_shape = Shape{2};
    auto gamma = make_shared<op::Parameter>(element::f32, gamma_shape);
    auto beta_shape = Shape{2};
    auto beta = make_shared<op::Parameter>(element::f32, beta_shape);
    double eps = 0.001;
    auto shape_r = Shape{2, 2, 2, 1};
    auto bn = make_shared<op::BatchNormTraining>(input, gamma, beta, eps);

    auto output_rt = std::make_shared<op::GetOutputElement>(bn, 0);
    auto mean_rt = std::make_shared<op::GetOutputElement>(bn, 1);
    auto variance_rt = std::make_shared<op::GetOutputElement>(bn, 2);

    auto f = make_shared<Function>(NodeVector{output_rt, mean_rt, variance_rt},
                                   ParameterVector{input, gamma, beta});
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
    auto bn_output = backend->create_tensor(element::f32, shape_r);
    auto result_mean = backend->create_tensor(element::f32, mean_shape);
    auto result_variance = backend->create_tensor(element::f32, var_shape);

    vector<float> expected_result{
        -0.30327f, 1.1561f, -0.0963782f, -0.434702f, -1.4011f, 0.548275f, -1.06187f, 1.59295f};
    vector<float> expected_mean{0.583388f, 0.619252f};
    vector<float> expected_variance{0.0119972f, 0.0282681f};
    auto handle = backend->compile(f);
    handle->call_with_validate({bn_output, result_mean, result_variance}, {_input, _gamma, _beta});

    EXPECT_TRUE(test::all_close(expected_result, read_vector<float>(bn_output)));
    EXPECT_TRUE(test::all_close(expected_mean, read_vector<float>(result_mean)));
    EXPECT_TRUE(
        test::all_close(expected_variance, read_vector<float>(result_variance), 1e-5f, 1e-6f));
}

NGRAPH_TEST(${BACKEND_NAME}, batch_norm_fprop_b2c2d2h1w1)
{
    auto input_shape = Shape{2, 2, 2, 1, 1};
    auto input = make_shared<op::Parameter>(element::f32, input_shape);
    auto mean_shape = Shape{2};
    auto var_shape = Shape{2};
    auto gamma_shape = Shape{2};
    auto gamma = make_shared<op::Parameter>(element::f32, gamma_shape);
    auto beta_shape = Shape{2};
    auto beta = make_shared<op::Parameter>(element::f32, beta_shape);
    double eps = 0.001;
    auto shape_r = Shape{2, 2, 2, 1, 1};
    auto bn = make_shared<op::BatchNormTraining>(eps, gamma, beta, input);

    auto output_rt = std::make_shared<op::GetOutputElement>(bn, 0);
    auto mean_rt = std::make_shared<op::GetOutputElement>(bn, 1);
    auto variance_rt = std::make_shared<op::GetOutputElement>(bn, 2);

    auto f = make_shared<Function>(NodeVector{output_rt, mean_rt, variance_rt},
                                   ParameterVector{input, gamma, beta});
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
    auto bn_output = backend->create_tensor(element::f32, shape_r);
    auto result_mean = backend->create_tensor(element::f32, mean_shape);
    auto result_variance = backend->create_tensor(element::f32, var_shape);

    vector<float> expected_result{
        -0.30327f, 1.1561f, -0.0963782f, -0.434702f, -1.4011f, 0.548275f, -1.06187f, 1.59295f};
    vector<float> expected_mean{0.583388f, 0.619252f};
    vector<float> expected_variance{0.0119972f, 0.0282681f};
    auto handle = backend->compile(f);
    handle->call_with_validate({bn_output, result_mean, result_variance}, {_input, _gamma, _beta});

    EXPECT_TRUE(test::all_close(expected_result, read_vector<float>(bn_output)));
    EXPECT_TRUE(test::all_close(expected_mean, read_vector<float>(result_mean)));
    EXPECT_TRUE(
        test::all_close(expected_variance, read_vector<float>(result_variance), 1e-5f, 1e-6f));
}

NGRAPH_TEST(${BACKEND_NAME}, batch_norm_bprop_n4c3h2w2)
{
    auto input_shape = Shape{4, 3, 2, 2};
    auto shape_mean = Shape{3};
    auto input = make_shared<op::Parameter>(element::f32, input_shape);
    auto mean_shape = Shape{3};
    auto mean = make_shared<op::Parameter>(element::f32, mean_shape);
    auto var_shape = Shape{3};
    auto var = make_shared<op::Parameter>(element::f32, var_shape);
    auto gamma_shape = Shape{3};
    auto gamma = make_shared<op::Parameter>(element::f32, gamma_shape);
    auto beta_shape = Shape{3};
    auto beta = make_shared<op::Parameter>(element::f32, beta_shape);
    double eps = 0.001;
    auto shape_r = Shape{4, 3, 2, 2};
    auto bn = make_shared<op::BatchNormTraining>(input, gamma, beta, eps);
    auto bn_dx = make_shared<op::GetOutputElement>(bn, 0);
    auto bn_dgamma = make_shared<op::GetOutputElement>(bn, 1);
    auto bn_dbeta = make_shared<op::GetOutputElement>(bn, 2);

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto _input = backend->create_tensor(element::f32, input_shape);
    vector<float> dataInput{
        10.76331902f, 11.51178265f, 10.31018162f, 12.2993021f,  14.17626667f, 14.63498497f,
        13.63494492f, 13.84248161f, 11.34602547f, 13.22014618f, 10.46686649f, 10.39842987f,
        12.94806862f, 11.71670246f, 14.94438076f, 13.13236618f, 13.40889645f, 12.76128387f,
        11.34430027f, 11.86629677f, 11.11464024f, 10.93221283f, 11.95324039f, 10.96581173f,
        13.05455494f, 14.41404247f, 13.11169434f, 11.26559448f, 10.89965153f, 14.08202171f,
        11.12685776f, 12.58428574f, 12.59247875f, 13.00187492f, 12.66310215f, 10.06655025f,
        12.62048626f, 14.47942352f, 13.84950638f, 10.61425877f, 11.47936344f, 13.06011772f,
        13.63069057f, 12.31748772f, 13.84555244f, 10.95815468f, 12.78933334f, 12.75389099f};
    copy_data(_input, dataInput);
    auto _mean = backend->create_tensor(element::f32, mean_shape);
    copy_data(_mean, vector<float>{12.56472874f, 12.80312157f, 11.81676865f});
    auto _var = backend->create_tensor(element::f32, var_shape);
    copy_data(_var, vector<float>{1.94557643f, 1.32772446f, 1.28163588f});

    auto _gamma = backend->create_tensor(element::f32, gamma_shape);
    copy_data(_gamma, vector<float>{2.0f, 2.0f, 2.0f});
    auto _beta = backend->create_tensor(element::f32, beta_shape);
    copy_data(_beta, vector<float>{1.0f, 1.0f, 1.0f});
    auto result = backend->create_tensor(element::f32, shape_r);

    shared_ptr<runtime::Tensor> _delta = backend->create_tensor(element::f32, shape_r);
    vector<float> deltaData(shape_size(shape_r), 20.0f);
    copy_data(_delta, deltaData);

    auto f = make_shared<Function>(NodeVector{bn_dx, bn_dgamma, bn_dbeta},
                                   ParameterVector{mean, var, input, gamma, beta});

    auto C = std::make_shared<op::Parameter>(element::f32, shape_r);

    auto zero = ngraph::make_zero(bn_dgamma->get_element_type(), bn_dgamma->get_shape());
    ngraph::autodiff::Adjoints adjoints(OutputVector{bn_dx, bn_dgamma, bn_dbeta},
                                        OutputVector{C, zero, zero});

    auto dinput = adjoints.backprop_output(input);
    auto dgamma = adjoints.backprop_output(gamma);
    auto dbeta = adjoints.backprop_output(beta);

    auto df = make_shared<Function>(OutputVector{dinput, dgamma, dbeta},
                                    ParameterVector{mean, var, input, gamma, beta, C});

#ifndef NGRAPH_JSON_DISABLE
    // roundtrip serialization
    string js = serialize(df, 4);
    istringstream in(js);
    df = deserialize(in);
#endif

    shared_ptr<runtime::Tensor> _dinput = backend->create_tensor(element::f32, shape_r);
    shared_ptr<runtime::Tensor> _dgamma = backend->create_tensor(element::f32, gamma_shape);
    shared_ptr<runtime::Tensor> _dbeta = backend->create_tensor(element::f32, beta_shape);

    auto handle = backend->compile(df);
    handle->call_with_validate({_dinput, _dgamma, _dbeta},
                               {_mean, _var, _input, _gamma, _beta, _delta});

    vector<float> expected_input{
        8.17051607e-06f,  4.77576657e-06f,  1.02257760e-05f,  1.20387525e-06f,  -1.73868522e-06f,
        3.84632768e-06f,  -1.07932050e-05f, -2.57458956e-06f, -2.22166714e-06f, -8.38779043e-06f,
        -2.48082982e-06f, 5.89238360e-06f,  -2.52895109e-07f, -8.68433445e-06f, -5.82726737e-06f,
        8.84659658e-06f,  3.03944108e-05f,  4.05480879e-05f,  1.84123158e-05f,  2.30061178e-05f,
        1.34087590e-05f,  -9.26072571e-07f, -3.22908454e-05f, -2.07365116e-05f, -4.21330941e-05f,
        2.83083100e-05f,  -3.71039101e-05f, -4.84390640e-06f, -2.93012376e-05f, 5.68858087e-06f,
        1.83181458e-05f,  -1.07494506e-05f, -2.32429103e-06f, 6.92914809e-06f,  -6.66512321e-06f,
        -7.00302840e-06f, -3.46675184e-06f, -4.36748381e-06f, 6.73822226e-07f,  -4.20158993e-06f,
        3.83005061e-06f,  5.85143729e-06f,  4.17875243e-06f,  -8.64167783e-06f, 1.00170803e-05f,
        -4.23939666e-06f, 4.80201680e-06f,  4.62702078e-06f};

    ASSERT_TRUE(ngraph::test::all_close(read_vector<float>(_dinput), expected_input, 1e-3f, 1e-4f));
    vector<float> expected_dgamma{7.06315041e-05f, -2.35289335e-04f, -5.06639481e-05f};
    ASSERT_TRUE(
        ngraph::test::all_close(read_vector<float>(_dgamma), expected_dgamma, 1e-2f, 1e-3f));
    vector<float> expected_dbeta{320.f, 320.f, 320.f};
    ASSERT_TRUE(ngraph::test::all_close(read_vector<float>(_dbeta), expected_dbeta, 1e-4f, 1e-8f));
}

NGRAPH_TEST(${BACKEND_NAME}, batch_norm_fprop_inference_b2c2h2w1)
{
    auto input_shape = Shape{2, 2, 2, 1};
    auto input = make_shared<op::Parameter>(element::f32, input_shape);
    auto mean_shape = Shape{2};
    auto mean = make_shared<op::Parameter>(element::f32, mean_shape);
    auto var_shape = Shape{2};
    auto var = make_shared<op::Parameter>(element::f32, var_shape);
    auto gamma_shape = Shape{2};
    auto gamma = make_shared<op::Parameter>(element::f32, gamma_shape);
    auto beta_shape = Shape{2};
    auto beta = make_shared<op::Parameter>(element::f32, beta_shape);
    double eps = 0.001;
    auto shape_r = Shape{2, 2, 2, 1};
    auto bn = make_shared<op::BatchNormInference>(input, gamma, beta, mean, var, eps);

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

NGRAPH_TEST(DISABLED_${BACKEND_NAME}, dyn_batch_norm_fprop_b1c2h2w2)
{
    // auto input_shape = Shape{1, 2, 2, 2};
    auto input = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto mean_shape = Shape{2};
    auto var_shape = Shape{2};
    auto gamma_shape = Shape{2};
    auto gamma = make_shared<op::Parameter>(element::f32, gamma_shape);
    auto beta_shape = Shape{2};
    auto beta = make_shared<op::Parameter>(element::f32, beta_shape);
    double eps = 0.001;
    auto shape_r = Shape{1, 2, 2, 2};
    auto bn = make_shared<op::BatchNormTraining>(input, gamma, beta, eps);

    auto output_rt = std::make_shared<op::GetOutputElement>(bn, 0);
    auto mean_rt = std::make_shared<op::GetOutputElement>(bn, 1);
    auto variance_rt = std::make_shared<op::GetOutputElement>(bn, 2);

    auto shapeof_mean_rt = std::make_shared<ngraph::op::ShapeOf>(mean_rt);
    auto rankof_mean_rt = std::make_shared<ngraph::op::ShapeOf>(shapeof_mean_rt);
    auto rank_scalar = std::make_shared<ngraph::op::Reshape>(
        rankof_mean_rt, ngraph::AxisVector{0}, ngraph::Shape{});
    auto range = std::make_shared<ngraph::op::Range>(
        ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0}),
        rank_scalar,
        ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{}, {1}));

    auto one_bcast = std::make_shared<ngraph::op::DynBroadcast>(
        ngraph::op::Constant::create(mean_rt->get_element_type(), ngraph::Shape{}, {1}),
        shapeof_mean_rt,
        range);
    auto mean_rt_multiplied = std::make_shared<ngraph::op::Multiply>(one_bcast, mean_rt);

    auto f = make_shared<Function>(NodeVector{output_rt, mean_rt_multiplied, variance_rt},
                                   ParameterVector{input, gamma, beta});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    // Create some tensors for input/output
    auto _input = backend->create_tensor(element::f32, Shape{1, 2, 2, 2});

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

    auto bn_output = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());
    auto result_mean = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());
    auto result_variance = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());

    vector<float> expected_result{-0.71498716f,
                                  1.48388731f,
                                  -0.00196938f,
                                  -0.76693159f,
                                  -0.91316032f,
                                  0.23943391f,
                                  -0.84090298f,
                                  1.51462936f};
    vector<float> expected_mean{0.602912f, 0.599727f};
    vector<float> expected_variance{0.00472505f, 0.0361782f};

    auto handle = backend->compile(f);
    handle->call_with_validate({bn_output, result_mean, result_variance}, {_input, _gamma, _beta});

    EXPECT_TRUE(test::all_close(expected_result, read_vector<float>(bn_output), 1e-5f, 1e-6f));
    EXPECT_TRUE(test::all_close(expected_mean, read_vector<float>(result_mean), 1e-5f, 1e-6f));
    EXPECT_TRUE(
        test::all_close(expected_variance, read_vector<float>(result_variance), 1e-5f, 1e-6f));
}
