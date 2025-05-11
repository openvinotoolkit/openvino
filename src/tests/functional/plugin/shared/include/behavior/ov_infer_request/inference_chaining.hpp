// Copyright (C) 2018-2025 Intel Corporation
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
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/model.hpp"
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
namespace behavior {

class OVInferenceChaining : public OVInferRequestTests {
protected:
    static std::shared_ptr<ov::Model> getFirstStaticFunction(const ov::PartialShape &shape = {3});

    static std::shared_ptr<ov::Model> getSecondStaticFunction(const ov::PartialShape &shape = {3});

    static std::shared_ptr<ov::Model> getThirdStaticFunction(const ov::PartialShape &shape = {3});

    template<typename T>
    ov::Tensor tensor(const std::vector<T> &v) {
        auto type = ov::element::from<T>();
        ov::Tensor tensor(type, {v.size()});
        std::memcpy(tensor.data(), v.data(), v.size() * type.size());

        return tensor;
    }

    std::shared_ptr<ov::Model> function0;
    std::shared_ptr<ov::Model> function1;
    std::shared_ptr<ov::Model> function2;

    bool outputToInput = true;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<InferRequestParams>& obj);

    void Run();
};
using OVInferenceChainingStatic = OVInferenceChaining;

}  // namespace behavior
}  // namespace test
}  // namespace ov
