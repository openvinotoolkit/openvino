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
#include "ngraph/pass/batch_fusion.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "runtime/backend.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/autodiff/numeric_compare.hpp"
#include "util/known_element_types.hpp"
#include "util/ndarray.hpp"
#include "util/random.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

// This test operates against the INTERPRETER backend as a reference, so it is
// disabled if INTERPRETER is disabled.
#if NGRAPH_INTERPRETER_ENABLE
NGRAPH_TEST(${BACKEND_NAME}, batch_mat_mul_forward)
{
    auto make_dot = [](ParameterVector& a_params, ParameterVector& b_params) {
        Shape shape_a{2, 3};
        Shape shape_b{3, 2};
        auto A = make_shared<op::Parameter>(element::f32, shape_a);
        auto B = make_shared<op::Parameter>(element::f32, shape_b);
        a_params.push_back(A);
        b_params.push_back(B);
        return make_shared<op::Dot>(A, B);
    };

    ParameterVector dot_a_params;
    ParameterVector dot_b_params;
    auto dot1 = make_dot(dot_a_params, dot_b_params);
    auto dot2 = make_dot(dot_a_params, dot_b_params);
    auto dot3 = make_dot(dot_a_params, dot_b_params);
    auto dot_concat = make_shared<op::Concat>(NodeVector{dot1, dot2, dot3}, 0);
    ParameterVector dot_params(dot_a_params);
    dot_params.insert(dot_params.end(), dot_b_params.begin(), dot_b_params.end());
    auto ref_f = make_shared<Function>(dot_concat, dot_params);

    auto make_batchmatmul = [](ParameterVector& params) {
        Shape shape_a{3, 2, 3};
        Shape shape_b{3, 3, 2};
        auto A = make_shared<op::Parameter>(element::f32, shape_a);
        auto B = make_shared<op::Parameter>(element::f32, shape_b);
        params.push_back(A);
        params.push_back(B);
        return make_shared<op::BatchMatMul>(A, B);
    };

    ParameterVector batchmatmul_params;
    auto batchmatmul = make_batchmatmul(batchmatmul_params);
    auto backend_f = make_shared<Function>(batchmatmul, batchmatmul_params);

    test::Uniform<float> dot_rng(-1.0f, 1.0f);
    vector<vector<float>> dot_args;
    for (shared_ptr<op::Parameter> param : dot_params)
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        dot_rng.initialize(tensor_val);
        dot_args.push_back(tensor_val);
    }

    test::Uniform<float> batchmatmul_rng(-1.0f, 1.0f);
    vector<vector<float>> batchmatmul_args;
    for (shared_ptr<op::Parameter> param : batchmatmul_params)
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        batchmatmul_rng.initialize(tensor_val);
        batchmatmul_args.push_back(tensor_val);
    }
    auto ref_results = execute(ref_f, dot_args, "INTERPRETER");
    auto backend_results = execute(backend_f, batchmatmul_args, "${BACKEND_NAME}");
    for (size_t i = 0; i < ref_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close_f(
            ref_results.at(i), backend_results.at(i), DEFAULT_FLOAT_TOLERANCE_BITS + 3));
    }
}

#ifndef NGRAPH_JSON_DISABLE
NGRAPH_TEST(${BACKEND_NAME}, fuse_batch_mat_mul_transpose_forward)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::BatchFusion>();

    const std::string file_name("mxnet/batch_dot_3.json");
    auto backend_f = make_function_from_file(file_name);
    auto int_f = make_function_from_file(file_name);
    pass_manager.run_passes(backend_f);
    test::Uniform<float> rng(0.0f, 1.0f);
    vector<vector<float>> args;

    for (shared_ptr<op::Parameter> param : int_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto backend_results = execute(backend_f, args, "${BACKEND_NAME}");
    for (size_t i = 0; i < int_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(backend_results.at(i), int_results.at(i), 1.0e-4f, 1.0e-4f));
    }
}

//#if defined(AUTODIFF_BACKEND_${BACKEND_NAME})
NGRAPH_TEST(${BACKEND_NAME}, backwards_batchmatmultranspose_tensor2_tensor2)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    const std::string file_name("mxnet/batch_dot_3.json");
    auto f = make_function_from_file(file_name);

    test::Uniform<float> rng(-1.0f, 1.0f);
    std::vector<std::shared_ptr<ngraph::runtime::Tensor>> args;
    for (shared_ptr<op::Parameter> param : f->get_parameters())
    {
        args.push_back(rng.initialize(backend->create_tensor<float>(param->get_shape())));
    }

    auto g = make_function_from_file(file_name);
    pass::Manager pass_manager;
    pass_manager.register_pass<ngraph::pass::BatchFusion>();
    pass_manager.run_passes(g);
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend.get(), f, g, args, .01f, .01f));
}
//#endif
#endif

#endif
