// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

#include <optional>

#include "openvino/core/model.hpp"
#include "openvino/core/model_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"

namespace ov::test {

using op::v0::Parameter, op::v1::Add, op::v1::Multiply, op::v0::Result;

class ModelUtilTest : public testing::Test {
protected:
    static auto make_model_without_tensor_names() {
        auto input_1 = std::make_shared<Parameter>(element::f32, Shape{1, 3});
        auto input_2 = std::make_shared<Parameter>(element::f32, Shape{3, 3});
        auto input_3 = std::make_shared<Parameter>(element::f32, Shape{3, 1});

        auto add = std::make_shared<Add>(input_1, input_2);
        auto mul = std::make_shared<Multiply>(add, input_3);
        auto add_final = std::make_shared<Add>(mul, add);

        auto add_result = std::make_shared<Result>(add);
        auto mul_result = std::make_shared<Result>(mul);
        auto final_result = std::make_shared<Result>(add_final);

        auto model = std::make_shared<Model>(ResultVector{add_result, mul_result, final_result},
                                             ParameterVector{input_1, input_2, input_3},
                                             "ModelWithoutTensorNames");
        return model;
    }

    static auto make_model_with_named_nodes() {
        auto model = make_model_without_tensor_names();

        const auto& inputs = model->inputs();
        for (size_t i = 0; i < inputs.size(); ++i) {
            inputs[i].get_node()->set_friendly_name("input_" + std::to_string(i + 1));
        }

        auto results = model->get_results();
        results[0]->set_friendly_name("add_result");
        results[1]->set_friendly_name("mul_result");
        results[2]->set_friendly_name("final_result");
        model->set_friendly_name("ModelWithNamedNodes");
        return model;
    }

    static auto compare_tensor_names(const OutputVector& outputs, const TensorNamesMap& expected) {
        std::optional<std::string> mismatch_err;

        for (const auto& [port, expected_names] : expected) {
            if (const auto& names = outputs[port].get_names(); expected_names != names) {
                using testing::PrintToString;
                mismatch_err.emplace("Tensor names mismatch on port " + PrintToString(port) + "\n Expected: " +
                                     PrintToString(expected_names) + "\n Actual:   " + PrintToString(names));
            }
        }
        return mismatch_err;
    }
};

TEST_F(ModelUtilTest, manual_set_all_input_tensors_names) {
    const auto inputs_names = TensorNamesMap{{0, {"input_1"}}, {1, {"input_2"}}, {2, {"input_3"}}};

    auto model = make_model_without_tensor_names();
    util::set_input_tensors_names(*model, inputs_names);

    ASSERT_EQ(model->inputs().size(), inputs_names.size());
    for (const auto& [port, names] : inputs_names) {
        EXPECT_EQ(model->input(port).get_names(), names) << "Names not match for input port " << port;
    }
}
TEST_F(ModelUtilTest, manual_set_some_input_tensors_names) {
    const auto inputs_names = TensorNamesMap{{2, {"input_2", "mul_input"}}, {0, {"add_input"}}};
    auto expected_names = inputs_names;
    expected_names.emplace(1, TensorNames{});

    auto model = make_model_without_tensor_names();
    util::set_input_tensors_names(*model, inputs_names);

    ASSERT_EQ(model->inputs().size(), expected_names.size());
    const auto mismatch_error = compare_tensor_names(model->inputs(), expected_names);
    EXPECT_FALSE(mismatch_error) << *mismatch_error;
}

TEST_F(ModelUtilTest, auto_set_all_input_tensors_names) {
    const auto expected_names = TensorNamesMap{{0, {"input_1"}}, {1, {"input_2"}}, {2, {"input_3"}}};

    auto model = make_model_with_named_nodes();
    util::set_input_tensors_names(AUTO, *model);

    ASSERT_EQ(model->inputs().size(), expected_names.size());
    const auto mismatch_error = compare_tensor_names(model->inputs(), expected_names);
    EXPECT_FALSE(mismatch_error) << *mismatch_error;
}

TEST_F(ModelUtilTest, auto_set_missing_input_tensors_names) {
    const auto inputs_names = TensorNamesMap{{2, {"input_2", "mul_input"}}, {0, {"add_input"}}};
    auto expected_names = inputs_names;
    expected_names.emplace(1, TensorNames{"input_2"});

    auto model = make_model_with_named_nodes();
    util::set_input_tensors_names(AUTO, *model, inputs_names);

    ASSERT_EQ(model->inputs().size(), expected_names.size());
    const auto mismatch_error = compare_tensor_names(model->inputs(), expected_names);
    EXPECT_FALSE(mismatch_error) << *mismatch_error;
}

TEST_F(ModelUtilTest, auto_set_all_io_tensors_names) {
    const auto exp_inputs_names = TensorNamesMap{{2, {"input_3"}}, {0, {"input_1"}}, {1, {"input_2"}}};
    const auto exp_outputs_names = TensorNamesMap{{0, {"add_result"}}, {1, {"mul_result"}}, {2, {"final_result"}}};

    auto model = make_model_with_named_nodes();
    util::set_tensors_names(AUTO, *model);

    ASSERT_EQ(model->inputs().size(), exp_inputs_names.size());
    auto mismatch_error = compare_tensor_names(model->inputs(), exp_inputs_names);
    EXPECT_FALSE(mismatch_error) << *mismatch_error;

    ASSERT_EQ(model->outputs().size(), exp_outputs_names.size());
    mismatch_error = compare_tensor_names(model->outputs(), exp_outputs_names);
    EXPECT_FALSE(mismatch_error) << *mismatch_error;
}

TEST_F(ModelUtilTest, manual_set_all_io_tensors_names) {
    const auto inputs_names = TensorNamesMap{{0, {"input_1"}}, {1, {"input_2"}}, {2, {"input_3"}}};
    const auto outputs_names = TensorNamesMap{{0, {"add_result"}}, {1, {"mul_result"}}, {2, {"final_result"}}};

    auto model = make_model_without_tensor_names();
    util::set_tensors_names(*model, inputs_names, outputs_names);

    ASSERT_EQ(model->inputs().size(), inputs_names.size());
    auto mismatch_error = compare_tensor_names(model->inputs(), inputs_names);
    EXPECT_FALSE(mismatch_error) << *mismatch_error;

    ASSERT_EQ(model->outputs().size(), outputs_names.size());
    mismatch_error = compare_tensor_names(model->outputs(), outputs_names);
    EXPECT_FALSE(mismatch_error) << *mismatch_error;
}

TEST_F(ModelUtilTest, manual_set_some_io_tensors_names) {
    const auto inputs_names = TensorNamesMap{{0, {"input_1"}}, {2, {"input_3"}}};
    auto expected_input_names = inputs_names;
    expected_input_names.emplace(1, TensorNames{});

    const auto outputs_names = TensorNamesMap{{1, {"mul_result"}}};
    auto expected_output_names = outputs_names;
    expected_output_names.emplace(0, TensorNames{});
    expected_output_names.emplace(2, TensorNames{});

    auto model = make_model_without_tensor_names();
    util::set_tensors_names(*model, inputs_names, outputs_names);

    ASSERT_EQ(model->inputs().size(), expected_input_names.size());
    auto mismatch_error = compare_tensor_names(model->inputs(), expected_input_names);
    EXPECT_FALSE(mismatch_error) << *mismatch_error;

    ASSERT_EQ(model->outputs().size(), expected_output_names.size());
    mismatch_error = compare_tensor_names(model->outputs(), expected_output_names);
    EXPECT_FALSE(mismatch_error) << *mismatch_error;
}

TEST_F(ModelUtilTest, manual_set_all_output_tensors_names) {
    const auto outputs_names = TensorNamesMap{{0, {"add_result"}}, {1, {"mul_result"}}, {2, {"final_result"}}};

    auto model = make_model_with_named_nodes();
    util::set_output_tensor_names(*model, outputs_names);

    ASSERT_EQ(model->outputs().size(), outputs_names.size());
    const auto mismatch_error = compare_tensor_names(model->outputs(), outputs_names);
    EXPECT_FALSE(mismatch_error) << *mismatch_error;
}

TEST_F(ModelUtilTest, manual_set_some_output_tensors_names) {
    const auto outputs_names = TensorNamesMap{{1, {"mul_result"}}, {2, {"final_result"}}};
    auto expected_names = outputs_names;
    expected_names.emplace(0, TensorNames{});

    auto model = make_model_with_named_nodes();
    util::set_output_tensor_names(*model, outputs_names);

    ASSERT_EQ(model->outputs().size(), expected_names.size());
    const auto mismatch_error = compare_tensor_names(model->outputs(), expected_names);
    EXPECT_FALSE(mismatch_error) << *mismatch_error;
}

TEST_F(ModelUtilTest, auto_set_all_output_tensors_names) {
    const auto expected_names = TensorNamesMap{{0, {"add_result"}}, {1, {"mul_result"}}, {2, {"final_result"}}};

    auto model = make_model_with_named_nodes();
    util::set_output_tensor_names(AUTO, *model);

    ASSERT_EQ(model->outputs().size(), expected_names.size());
    const auto mismatch_error = compare_tensor_names(model->outputs(), expected_names);
    EXPECT_FALSE(mismatch_error) << *mismatch_error;
}

TEST_F(ModelUtilTest, auto_set_missing_output_tensors_names) {
    const auto outputs_names = TensorNamesMap{{1, {"mul_result"}}, {2, {"final_result"}}};
    auto expected_names = outputs_names;
    expected_names.emplace(0, TensorNames{"add_result"});

    auto model = make_model_with_named_nodes();
    util::set_output_tensor_names(AUTO, *model, outputs_names);

    ASSERT_EQ(model->outputs().size(), expected_names.size());
    const auto mismatch_error = compare_tensor_names(model->outputs(), expected_names);
    EXPECT_FALSE(mismatch_error) << *mismatch_error;
}

}  // namespace ov::test
