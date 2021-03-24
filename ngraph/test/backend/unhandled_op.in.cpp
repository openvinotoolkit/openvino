// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "runtime/backend.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

namespace
{
    class UnhandledOp : public ngraph::op::Op
    {
    public:
        UnhandledOp(const Output<Node>& arg)
            : Op({arg})
        {
            constructor_validate_and_infer_types();
        }
        shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override
        {
            return make_shared<UnhandledOp>(new_args[0]);
        }

    protected:
        void validate_and_infer_types() override
        {
            set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
        }

        static constexpr NodeTypeInfo type_info{"UnhandledOp", 0};
        const NodeTypeInfo& get_type_info() const override { return type_info; }
    };

    constexpr NodeTypeInfo UnhandledOp::type_info;
}

NGRAPH_TEST(${BACKEND_NAME}, unhandled_op)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto unhandled = make_shared<UnhandledOp>(A);
    auto f = make_shared<Function>(unhandled, ParameterVector{A});
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    shared_ptr<runtime::Tensor> a = backend->create_tensor<float>(shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor<float>(shape);
    ASSERT_THROW(auto handle = backend->compile(f);
                 handle->call_with_validate({result}, {a}), unsupported_op);
}
