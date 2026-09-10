// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "behavior/ov_infer_request/infer_request_dynamic.hpp"
#include "common_test_utils/test_constants.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace test {
namespace behavior {

class InferRequestDynamicTests : public OVInferRequestDynamicTests {
public:
    static std::shared_ptr<ov::Model> getFunction() {
        const ov::Shape inputShape = {1, 10, 12};

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape)};
        params.front()->get_output_tensor(0).set_names({"Parameter_1"});

        auto relu = std::make_shared<ov::op::v0::Relu>(params[0]);
        relu->get_output_tensor(0).set_names({"Relu_2"});

        return std::make_shared<ov::Model>(relu, params, "SimpleActivation");
    }

protected:
    static bool exceedsUpperBounds(const ov::Shape& shape, const ov::PartialShape& bounds) {
        const auto maxShape = bounds.get_max_shape();
        for (size_t i = 0; i < maxShape.size(); i++) {
            if (shape[i] > maxShape[i]) {
                return true;
            }
        }
        return false;
    }

    void checkOutputFP16(const ov::Tensor& in, const ov::Tensor& actual) {
        auto net = ie->compile_model(function, ov::test::utils::DEVICE_TEMPLATE);
        ov::InferRequest req;
        req = net.create_infer_request();
        auto tensor = req.get_tensor(function->inputs().back().get_any_name());
        tensor.set_shape(in.get_shape());
        for (size_t i = 0; i < in.get_size(); i++) {
            tensor.data<ov::element_type_traits<ov::element::f32>::value_type>()[i] =
                in.data<ov::element_type_traits<ov::element::f32>::value_type>()[i];
        }
        req.infer();
        OVInferRequestDynamicTests::checkOutput(actual, req.get_output_tensor(0));
    }
};

}  // namespace behavior
}  // namespace test
}  // namespace ov
