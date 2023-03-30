// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/transpose_sinking/ts_squeeze.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include "gtest/gtest.h"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/manager.hpp"
#include "ts_test_case.hpp"
#include "ts_test_utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::element;
using namespace ov::opset10;
using namespace ov::pass::transpose_sinking;
using namespace transpose_sinking::testing::utils;

namespace transpose_sinking {
namespace testing {
namespace squeeze {

class SqueezeFactory : public IFactory {
public:
    explicit SqueezeFactory(const std::string& type_name) : IFactory(type_name) {}
    NodePtr create(const OutputVector& parent_nodes) const override {
        if (parent_nodes.size() == 2) {
            return std::make_shared<Squeeze>(parent_nodes[0], parent_nodes[1]);
        } else if (parent_nodes.size() == 1) {
            return std::make_shared<Squeeze>(parent_nodes[0]);
        }
        OPENVINO_ASSERT(false, "Unexpected number of inputs to Squeeze operation.");
    }
};

FactoryPtr CreateSqueezeFactory(const std::string& type_name) {
    return std::make_shared<SqueezeFactory>(type_name);
}
// ----------------------------------------------------------------------------

#undef CREATE_SQUEEZE_FACTORY
#define CREATE_SQUEEZE_FACTORY(type_name) CreateSqueezeFactory(#type_name)
// ----------------------------------------------------------------------------

shared_ptr<ov::Model> create_model(size_t main_node_idx,
                                   const ModelDescription& model_desc,
                                   size_t num_ops,
                                   const OutputVector& inputs_to_main) {
    auto new_inputs = model_desc.preprocess_inputs_to_main.apply(inputs_to_main);
    auto main_node = create_main_node(new_inputs, num_ops, model_desc.main_op[main_node_idx]);
    auto outputs = model_desc.preprocess_outputs_of_main.apply(main_node->outputs());
    return make_shared<ov::Model>(outputs, filter_parameters(inputs_to_main));
}

struct SqueezeForwardArguments {
    OutputVector inputs_to_main;
    Output<Node> new_input_to_squeeze_1;
};

auto test_forward_squeeze = [](const SqueezeForwardArguments& test_arguments) {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSSqueezeForward);
    test_case.num_main_ops = {1};
    test_case.inputs_to_main = test_arguments.inputs_to_main;

    // Test model description:
    test_case.model.preprocess_inputs_to_main = {{set_transpose_for}, {{0}}};
    test_case.model.main_op = {CREATE_SQUEEZE_FACTORY(Squeeze)};
    test_case.model.model_template = create_model;

    // Reference model description:
    auto new_constant = [&](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        OutputVector new_out_vec(out_vec.size());
        new_out_vec[0] = out_vec[0];
        if (out_vec.size() > 1) {
            new_out_vec[1] = test_arguments.new_input_to_squeeze_1;
        }
        return new_out_vec;
    };
    test_case.model_ref.preprocess_inputs_to_main = {{new_constant}, {{1}}};
    test_case.model_ref.main_op = {CREATE_SQUEEZE_FACTORY(Squeeze)};
    test_case.model_ref.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model_ref.model_template = create_model;

    return wrapper(test_case);
};

vector<SqueezeForwardArguments> tests_arguments{
    {{parameter(f32, {1, 2}), constant<int64_t>(i32, {1}, {1})}, constant<int64_t>(i32, {1}, {0})},
    {{parameter(f32, {1, 2, 1}), constant<int64_t>(i32, {2}, {0, 2})}, constant<int64_t>(i32, {2}, {2, 0})},
    {{parameter(f32, {1, 1, 2, 1}), constant<int64_t>(i32, {3}, {0, 2, 3})}, constant<int64_t>(i32, {3}, {3, 1, 0})},
    {{constant<int64_t>(i32, {1, 2}, {1}), constant<int64_t>(i32, {1}, {1})}, constant<int64_t>(i32, {1}, {0})},
    {{parameter(f32, {1, 2})}},
};

INSTANTIATE_TEST_SUITE_P(TSCommonSqueezeForward_0, TSTestFixture, test_forward_squeeze(tests_arguments[0]));
INSTANTIATE_TEST_SUITE_P(TSCommonSqueezeForward_1, TSTestFixture, test_forward_squeeze(tests_arguments[1]));
INSTANTIATE_TEST_SUITE_P(TSCommonSqueezeForward_2, TSTestFixture, test_forward_squeeze(tests_arguments[2]));
INSTANTIATE_TEST_SUITE_P(TSCommonSqueezeForward_3, TSTestFixture, test_forward_squeeze(tests_arguments[3]));
INSTANTIATE_TEST_SUITE_P(TSCommonSqueezeForward_4, TSTestFixture, test_forward_squeeze(tests_arguments[4]));

}  // namespace squeeze
}  // namespace testing
}  // namespace transpose_sinking
