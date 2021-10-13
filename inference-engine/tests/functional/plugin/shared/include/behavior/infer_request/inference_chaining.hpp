// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <chrono>
#include <initializer_list>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "base/ov_behavior_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/function.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/rank.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace test {

class OVInferenceChaining : public ov::test::BehaviorTestsBasic {
protected:
    static std::shared_ptr<ov::Function> getFirstStaticFunction(const ov::element::Type type,
                                                                const ov::PartialShape& shape = {3}) {
        auto params = ngraph::builder::makeDynamicParams(type, {shape, shape, shape});
        params[0]->get_output_tensor(0).set_names({"input_tensor_0"});
        params[0]->set_friendly_name("param_0");
        params[1]->get_output_tensor(0).set_names({"input_tensor_1"});
        params[1]->set_friendly_name("param_1");
        params[2]->get_output_tensor(0).set_names({"input_tensor_2"});
        params[2]->set_friendly_name("param_2");
        auto eltwise = ngraph::builder::makeEltwise(params[0], params[1], ngraph::helpers::EltwiseTypes::ADD);
        auto eltwise2 = ngraph::builder::makeEltwise(eltwise, params[2], ngraph::helpers::EltwiseTypes::ADD);
        eltwise2->get_output_tensor(0).set_names({"result_tensor_0"});
        eltwise2->set_friendly_name("result_0");

        return std::make_shared<ov::Function>(eltwise2, ov::ParameterVector(params));
    }

    static std::shared_ptr<ov::Function> getSecondStaticFunction(const ov::element::Type type,
                                                                 const ov::PartialShape& shape = {3}) {
        auto params = ngraph::builder::makeDynamicParams(type, {shape, shape});
        params[0]->get_output_tensor(0).set_names({"input_tensor_0"});
        params[0]->set_friendly_name("param_0");
        params[1]->get_output_tensor(0).set_names({"input_tensor_1"});
        params[1]->set_friendly_name("param_1");
        auto eltwise = ngraph::builder::makeEltwise(params[0], params[1], ngraph::helpers::EltwiseTypes::MULTIPLY);
        eltwise->get_output_tensor(0).set_names({"result_tensor_0"});
        eltwise->set_friendly_name("result_0");

        return std::make_shared<ov::Function>(eltwise, ov::ParameterVector(params));
    }

    static std::shared_ptr<ov::Function> getThirdStaticFunction(const ov::element::Type type,
                                                                const ov::PartialShape& shape = {3}) {
        auto params = ngraph::builder::makeDynamicParams(type, {shape, shape, shape, shape});
        params[0]->get_output_tensor(0).set_names({"input_tensor_0"});
        params[0]->set_friendly_name("param_0");
        params[1]->get_output_tensor(0).set_names({"input_tensor_1"});
        params[1]->set_friendly_name("param_1");
        params[2]->get_output_tensor(0).set_names({"input_tensor_2"});
        params[2]->set_friendly_name("param_2");
        params[3]->get_output_tensor(0).set_names({"input_tensor_3"});
        params[3]->set_friendly_name("param_3");
        auto eltwise = ngraph::builder::makeEltwise(params[0], params[1], ngraph::helpers::EltwiseTypes::ADD);
        auto eltwise2 = ngraph::builder::makeEltwise(eltwise, params[2], ngraph::helpers::EltwiseTypes::ADD);
        auto eltwise3 = ngraph::builder::makeEltwise(eltwise2, params[3], ngraph::helpers::EltwiseTypes::MULTIPLY);
        eltwise3->get_output_tensor(0).set_names({"result_tensor_0"});
        eltwise3->set_friendly_name("result_0");

        return std::make_shared<ov::Function>(eltwise3, ov::ParameterVector(params));
    }

    template <typename T>
    ov::runtime::Tensor tensor(const std::vector<T>& v) {
        auto type = ov::element::from<T>();
        ov::runtime::Tensor tensor(type, {v.size()});
        std::memcpy(tensor.data(), v.data(), v.size() * type.size());

        return tensor;
    }

    std::shared_ptr<ov::Function> function0;
    std::shared_ptr<ov::Function> function1;
    std::shared_ptr<ov::Function> function2;

    bool outputToInput = true;

public:
    void Run() {
        ov::runtime::ExecutableNetwork execNet0, execNet1, execNet2;
        ASSERT_NO_THROW(execNet0 = ie->compile_model(function0, targetDevice, configuration));
        ASSERT_NO_THROW(execNet1 = ie->compile_model(function1, targetDevice, configuration));
        ASSERT_NO_THROW(execNet2 = ie->compile_model(function2, targetDevice, configuration));

        ov::runtime::InferRequest r0, r1, r2;
        ASSERT_NO_THROW(r0 = execNet0.create_infer_request());
        ASSERT_NO_THROW(r1 = execNet1.create_infer_request());
        ASSERT_NO_THROW(r2 = execNet2.create_infer_request());

        // perform inference chaining
        if (outputToInput) {
            ASSERT_NO_THROW(r1.set_tensor("input_tensor_0", r0.get_tensor("result_tensor_0")));
        } else {
            ASSERT_NO_THROW(r0.set_tensor("result_tensor_0", r1.get_tensor("input_tensor_0")));
        }

        // create input tensors
        ov::runtime::Tensor t0 = tensor(std::vector<float>{1.0f, 2.0f, 3.0f});
        ov::runtime::Tensor t1 = tensor(std::vector<float>{4.0f, 5.0f, 6.0f});
        ov::runtime::Tensor t2 = tensor(std::vector<float>{7.0f, 8.0f, 9.0f});
        ov::runtime::Tensor t3 = tensor(std::vector<float>{2.0f, 3.0f, 2.0f});

        ASSERT_NO_THROW(r0.set_tensor("input_tensor_0", t0));
        ASSERT_NO_THROW(r0.set_tensor("input_tensor_1", t1));
        ASSERT_NO_THROW(r0.set_tensor("input_tensor_2", t2));
        ASSERT_NO_THROW(r1.set_tensor("input_tensor_1", t3));

        ASSERT_NO_THROW(r2.set_tensor("input_tensor_0", t0));
        ASSERT_NO_THROW(r2.set_tensor("input_tensor_1", t1));
        ASSERT_NO_THROW(r2.set_tensor("input_tensor_2", t2));
        ASSERT_NO_THROW(r2.set_tensor("input_tensor_3", t3));

        ASSERT_NO_THROW(r0.infer());
        ASSERT_NO_THROW(r1.infer());
        ASSERT_NO_THROW(r2.infer());

        // check results
        std::vector<float> reference1 = {12.0f, 15.0f, 18.0f};
        std::vector<float> reference2 = {24.0f, 45.0f, 36.0f};

        auto rti = r0.get_tensor("result_tensor_0");
        auto rt0 = r1.get_tensor("result_tensor_0");
        auto rt1 = r2.get_tensor("result_tensor_0");

        for (size_t i = 0; i < reference1.size(); ++i) {
            EXPECT_EQ(reference1[i], rti.data<float>()[i]);
            EXPECT_EQ(reference2[i], rt0.data<float>()[i]);
            EXPECT_EQ(reference2[i], rt1.data<float>()[i]);
        }
    }
};

TEST_P(OVInferenceChaining, StaticOutputToStaticInput) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    function0 = getFirstStaticFunction(elementType);
    function1 = getSecondStaticFunction(elementType);
    function2 = getThirdStaticFunction(elementType);

    Run();
}

TEST_P(OVInferenceChaining, StaticOutputToDynamicInput) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    const auto dynamic = ov::PartialShape::dynamic(ov::Rank(1));
    function0 = getFirstStaticFunction(elementType);
    function1 = getSecondStaticFunction(elementType, dynamic);
    function2 = getThirdStaticFunction(elementType, dynamic);

    Run();
}

TEST_P(OVInferenceChaining, DynamicOutputToDynamicInput) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    const auto dynamic = ov::PartialShape::dynamic();
    function0 = getFirstStaticFunction(elementType, dynamic);
    function1 = getSecondStaticFunction(elementType, dynamic);
    function2 = getThirdStaticFunction(elementType, dynamic);

    Run();
}

TEST_P(OVInferenceChaining, DynamicInputToDynamicOutput) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    this->outputToInput = false;

    const auto dynamic = ov::PartialShape::dynamic();
    function0 = getFirstStaticFunction(elementType, dynamic);
    function1 = getSecondStaticFunction(elementType, dynamic);
    function2 = getThirdStaticFunction(elementType, dynamic);

    Run();
}

}  // namespace test
}  // namespace ov
