// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_common.hpp"
#include "conversion_with_reference.hpp"
#include "gtest/gtest.h"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/subtract.hpp"
#include "tf_utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;
using namespace ov::frontend::tensorflow::tests;

TEST_F(FrontEndConversionWithReferenceTestsF, SavedModelProgramOnly) {
    {
        model = convert_model("saved_model_program-only");

        // check tensor names in the resulted model
        unordered_set<std::string> input_tensor_names = {"y"};
        unordered_set<std::string> output_tensor_names = {"z"};
        ASSERT_EQ(model->get_results().size(), 1);
        ASSERT_TRUE(model->get_results()[0]->input_value(0).get_names() == output_tensor_names);
        ASSERT_EQ(model->get_parameters().size(), 1);
        ASSERT_TRUE(model->get_parameters()[0]->output(0).get_names() == input_tensor_names);

        // check Parameter and Result node names
        ASSERT_TRUE(model->get_parameters()[0]->get_friendly_name() == "y");
        ASSERT_TRUE(model->get_results()[0]->get_friendly_name() == "z");
    }
    {
        // create a reference graph
        auto x = make_shared<v0::Constant>(element::f32, Shape{2, 3}, vector<float>{1, 2, 3, 3, 2, 1});
        auto y = make_shared<v0::Parameter>(element::f32, Shape{1});
        auto add = make_shared<v1::Add>(x, y);

        model_ref = make_shared<Model>(OutputVector{add}, ParameterVector{y});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, SavedModelVariables) {
    { model = convert_model("saved_model_variables"); }
    {
        // create a reference graph
        auto x = make_shared<v0::Parameter>(element::f32, Shape{1});
        auto y = make_shared<v0::Constant>(element::f32, Shape{}, vector<float>{123});
        auto multiply = make_shared<v1::Multiply>(x, y);

        model_ref = make_shared<Model>(OutputVector{multiply}, ParameterVector{x});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, SavedModelWithInputIntegerType) {
    {
        model = convert_model("saved_model_with_gather",
                              nullptr,
                              {"params", "indices"},
                              {},
                              {PartialShape{10, 5}, PartialShape{3}});

        // check tensor names in the resulted model
        unordered_set<std::string> input_tensor_name1 = {"params"};
        unordered_set<std::string> input_tensor_name2 = {"indices"};
        unordered_set<std::string> output_tensor_names = {"test_output_name"};
        ASSERT_EQ(model->get_results().size(), 1);
        ASSERT_TRUE(model->get_results()[0]->input_value(0).get_names() == output_tensor_names);
        ASSERT_EQ(model->get_parameters().size(), 2);
        ASSERT_TRUE(model->get_parameters()[0]->output(0).get_names() == input_tensor_name1);
        ASSERT_TRUE(model->get_parameters()[1]->output(0).get_names() == input_tensor_name2);

        // check Parameter and Result node names
        ASSERT_TRUE(model->get_parameters()[0]->get_friendly_name() == "params");
        ASSERT_TRUE(model->get_parameters()[1]->get_friendly_name() == "indices");
        ASSERT_TRUE(model->get_results()[0]->get_friendly_name() == "test_output_name");
    }
    {
        // create a reference graph
        auto params = make_shared<v0::Parameter>(element::f32, Shape{10, 5});
        auto indices = make_shared<v0::Parameter>(element::i32, Shape{3});
        auto gather_axis = make_shared<v0::Constant>(element::i32, Shape{}, 0);
        auto gather = make_shared<v8::Gather>(params, indices, gather_axis);

        auto const_mul = make_shared<v0::Constant>(element::f32, Shape{}, 5);
        auto mul = make_shared<v1::Multiply>(gather, const_mul);

        model_ref = make_shared<Model>(OutputVector{mul}, ParameterVector{params, indices});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, SavedModelMultipleTensorNames) {
    // The test aims to check tensor names of input and output tensors
    // it checks that TF FE preserved user specific names for input and output tensor
    // and exclude internal names
    {
        model = convert_model("saved_model_parameter_result");

        // check tensor names in the resulted model
        unordered_set<std::string> tensor_names = {"params", "test_output_name"};
        ASSERT_EQ(model->get_results().size(), 1);
        ASSERT_TRUE(model->get_results()[0]->input_value(0).get_names() == tensor_names);
        ASSERT_EQ(model->get_parameters().size(), 1);
        ASSERT_TRUE(model->get_parameters()[0]->output(0).get_names() == tensor_names);

        // check Parameter and Result node names
        ASSERT_TRUE(model->get_parameters()[0]->get_friendly_name() == "params");
        ASSERT_TRUE(model->get_results()[0]->get_friendly_name() == "test_output_name");
    }
    {
        // create a reference graph
        auto x = make_shared<v0::Parameter>(element::f32, Shape{20, 5});
        auto result = make_shared<v0::Result>(x);
        model_ref = make_shared<Model>(OutputVector{result}, ParameterVector{x});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, SavedModelBroadcastIssue) {
    { model = convert_model("saved_model_broadcast_issue"); }
    {
        // create a reference graph
        auto x = make_shared<v0::Constant>(element::i64, Shape{2, 2}, vector<int64_t>{1, 2, -1, -1});

        model_ref = make_shared<Model>(OutputVector{x}, ParameterVector{});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, SavedModelMultiGraph) {
    // The test verifies loading of MetaGraph with empty tags as default
    // And verifies loading variables with no corresponding RestoreV2
    { model = convert_model("saved_model_multi-graph"); }
    {
        // create a reference graph
        auto x = make_shared<v0::Constant>(element::f32, Shape{2, 3}, vector<float>{1, 2, 3, 3, 2, 1});
        auto y = make_shared<v0::Parameter>(element::f32, Shape{1});
        auto add = make_shared<v1::Add>(x, y);

        model_ref = make_shared<Model>(OutputVector{add}, ParameterVector{y});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, SavedModelWithIntermediateOutput) {
    // The test aims to check that output from intermediate layers presented in the model signature
    // must be preserved
    {
        model = convert_model("saved_model_intermediate_output");
        ASSERT_TRUE(model->get_results().size() == 2);
    }
    {
        // create a reference graph
        auto input1 = make_shared<v0::Parameter>(element::f32, Shape{2});
        auto input2 = make_shared<v0::Parameter>(element::f32, Shape{2});
        auto add = make_shared<v1::Add>(input1, input2);
        auto sub = make_shared<v1::Subtract>(input2, add);
        auto result1 = make_shared<v0::Result>(add);
        auto result2 = make_shared<v0::Result>(sub);
        model_ref = make_shared<Model>(OutputVector{result1, result2}, ParameterVector{input1, input2});
    }
}

TEST_F(FrontEndConversionWithReferenceTestsF, SavedModelMMAPCompare) {
    { model = convert_model("saved_model_variables"); }
    { model_ref = convert_model("saved_model_variables", nullptr, {}, {}, {}, {}, {}, true); }
}

TEST_F(FrontEndConversionWithReferenceTestsF, SavedModelWithNumericalNames) {
    comparator.enable(FunctionsComparator::CmpValues::TENSOR_NAMES);
    // The test aims to check that model with only numerical names for operation
    // is successfully converted
    // it is a tricky case because collision between naming input and output ports may occur
    { model = convert_model("saved_model_with_numerical_names"); }
    {
        // create a reference graph
        auto x = make_shared<v0::Parameter>(element::f32, Shape{1});
        x->output(0).set_names({"0"});
        auto y = make_shared<v0::Parameter>(element::f32, Shape{1});
        y->output(0).set_names({"1"});
        auto z = make_shared<v0::Parameter>(element::f32, Shape{1});
        z->output(0).set_names({"2"});
        auto add = make_shared<v1::Add>(x, y);
        add->output(0).set_names({"3:0"});
        auto sub = make_shared<v1::Subtract>(add, z);
        sub->output(0).set_names({"4"});
        auto result = make_shared<v0::Result>(sub);
        result->output(0).set_names({"4"});
        model_ref = make_shared<Model>(ResultVector{result}, ParameterVector{x, y, z});
    }
}

// Test that a crafted SavedModel with integer overflow in BundleEntryProto
// offset/size fields produces a clean exception, not a SIGSEGV.
// The malicious variables.index has entry.offset = 0x7FFFFFFFFFFFFFF0 and entry.size = 16.
// The sum overflows signed int64 to INT64_MIN, which would bypass the old bounds check.
TEST(FrontEndConvertModelTest, SavedModelMaliciousOverflowOffset) {
    shared_ptr<Model> model = nullptr;
    // Test with mmap enabled (default) — triggers crash at variables_index.cpp CKOG path
    try {
        model = convert_model("saved_model_malicious_overflow");
        FAIL() << "Loading a malicious SavedModel with overflow offset should throw an exception.";
    } catch (const ov::Exception& error) {
        string error_message = error.what();
        EXPECT_TRUE(error_message.find("entry") != string::npos || error_message.find("offset") != string::npos ||
                    error_message.find("bounds") != string::npos || error_message.find("negative") != string::npos ||
                    error_message.find("size") != string::npos)
            << "Unexpected error message: " << error_message;
        EXPECT_EQ(model, nullptr);
    } catch (const std::exception& e) {
        FAIL() << "Unexpected exception type: " << e.what();
    } catch (...) {
        FAIL() << "Unexpected non-std exception thrown";
    }
}

// Crafted SavedModel where AssignVariableOp input references "save/RestoreV2:999"
// but the Const(tensor_names) has only 1 string_val entry → OOB positive index.
TEST(FrontEndConvertModelTest, SavedModelOobPositiveIndex) {
    try {
        convert_model("saved_model_oob_pos_index");
        FAIL() << "Expected exception for OOB positive RestoreV2 output index";
    } catch (const ov::Exception& e) {
        EXPECT_TRUE(string(e.what()).find("out of range") != string::npos) << "Unexpected error message: " << e.what();
    } catch (const std::exception& e) {
        FAIL() << "Unexpected std::exception: " << e.what();
    } catch (...) {
        FAIL() << "Unexpected non-std exception";
    }
}

// Crafted SavedModel where AssignVariableOp input references "save/RestoreV2:-1" → negative OOB.
TEST(FrontEndConvertModelTest, SavedModelOobNegativeIndex) {
    try {
        convert_model("saved_model_oob_neg_index");
        FAIL() << "Expected exception for negative RestoreV2 output index";
    } catch (const ov::Exception& e) {
        EXPECT_TRUE(string(e.what()).find("out of range") != string::npos) << "Unexpected error message: " << e.what();
    } catch (const std::exception& e) {
        FAIL() << "Unexpected std::exception: " << e.what();
    } catch (...) {
        FAIL() << "Unexpected non-std exception";
    }
}

// Crafted SavedModel where AssignVariableOp input references "save/RestoreV2:0" but
// Const(tensor_names) has zero string_val entries → upper-bound guard fires at index 0.
// This exercises the security check on the implicit-0 code path.
TEST(FrontEndConvertModelTest, SavedModelOobEmptyTensorNames) {
    try {
        convert_model("saved_model_oob_empty_names");
        FAIL() << "Expected exception for OOB index into empty tensor_names";
    } catch (const ov::Exception& e) {
        EXPECT_TRUE(string(e.what()).find("out of range") != string::npos) << "Unexpected error message: " << e.what();
    } catch (const std::exception& e) {
        FAIL() << "Unexpected std::exception: " << e.what();
    } catch (...) {
        FAIL() << "Unexpected non-std exception";
    }
}

// Crafted TF1-style SavedModel where Assign.input(1) = "save/RestoreV2:999" but
// Const(tensor_names) has only 1 string_val entry → exercises the Assign code path
// in map_assignvariable() (distinct from the AssignVariableOp path in other OOB tests).
TEST(FrontEndConvertModelTest, SavedModelOobAssignPath) {
    try {
        convert_model("saved_model_oob_assign_path");
        FAIL() << "Expected exception for OOB RestoreV2 output index in Assign path";
    } catch (const ov::Exception& e) {
        EXPECT_TRUE(string(e.what()).find("out of range") != string::npos) << "Unexpected error message: " << e.what();
    } catch (const std::exception& e) {
        FAIL() << "Unexpected std::exception: " << e.what();
    } catch (...) {
        FAIL() << "Unexpected non-std exception";
    }
}

TEST(FrontEndConvertModelTest, SavedModelOobRestoreV2ShortInputs) {
    try {
        convert_model("saved_model_oob_restorev2_short_inputs");
        FAIL() << "Expected exception when RestoreV2 tensor_names input is missing";
    } catch (const ov::Exception& e) {
        EXPECT_TRUE(string(e.what()).find("missing tensor_names input") != string::npos)
            << "Unexpected error message: " << e.what();
    } catch (const std::exception& e) {
        FAIL() << "Unexpected std::exception: " << e.what();
    } catch (...) {
        FAIL() << "Unexpected non-std exception";
    }
}

// Crafted SavedModel where RestoreV2.input(1) names "tensor_names" but no such
// node exists in the GraphDef.  shape_and_slices is a Const that occupies the
// inputs[1] slot in the compacted PtrNode::inputs vector — a buggy
// implementation that resolves tensor_names via inputs[1] would silently use
// shape_and_slices.  Resolving by matching node->name() across rv2_node->inputs
// makes the missing input explicit.
TEST(FrontEndConvertModelTest, SavedModelOobRestoreV2WrongInputAtPort1) {
    try {
        convert_model("saved_model_oob_wrong_input_at_port1");
        FAIL() << "Expected exception when RestoreV2 tensor_names input node is absent";
    } catch (const ov::Exception& e) {
        EXPECT_TRUE(string(e.what()).find("not found among RestoreV2 inputs") != string::npos)
            << "Unexpected error message: " << e.what();
    } catch (const std::exception& e) {
        FAIL() << "Unexpected std::exception: " << e.what();
    } catch (...) {
        FAIL() << "Unexpected non-std exception";
    }
}

// Crafted SavedModel where RestoreV2.input(1) is a control dep (`^bogus`) and
// `bogus` is a Const with non-empty `string_val`.  parse_node_name strips the
// `^`, and associate_node has already linked `bogus` into rv2_node->inputs, so
// a buggy implementation that pulls the candidate from node->input(1) without
// checking for the `^` prefix would silently bind the variable to bogus's
// first string_val.  The fix iterates node->input() and skips control inputs,
// finds only one data input ('prefix'), and throws.
TEST(FrontEndConvertModelTest, SavedModelOobRestoreV2ControlDepAtPort1) {
    try {
        convert_model("saved_model_oob_control_dep_at_port1");
        FAIL() << "Expected exception when RestoreV2.input(1) is a control dep";
    } catch (const ov::Exception& e) {
        EXPECT_TRUE(string(e.what()).find("missing tensor_names input") != string::npos)
            << "Unexpected error message: " << e.what();
    } catch (const std::exception& e) {
        FAIL() << "Unexpected std::exception: " << e.what();
    } catch (...) {
        FAIL() << "Unexpected non-std exception";
    }
}

// Same test with mmap disabled to verify the stream code path is also protected
TEST(FrontEndConvertModelTest, SavedModelMaliciousOverflowOffsetNoMmap) {
    shared_ptr<Model> model = nullptr;
    try {
        model = convert_model("saved_model_malicious_overflow", nullptr, {}, {}, {}, {}, {}, true /* disable_mmap */);
        FAIL() << "Loading a malicious SavedModel with overflow offset should throw an exception.";
    } catch (const ov::Exception& error) {
        string error_message = error.what();
        EXPECT_TRUE(error_message.find("entry") != string::npos || error_message.find("offset") != string::npos ||
                    error_message.find("bounds") != string::npos || error_message.find("negative") != string::npos ||
                    error_message.find("size") != string::npos)
            << "Unexpected error message: " << error_message;
        EXPECT_EQ(model, nullptr);
    } catch (const std::exception& e) {
        FAIL() << "Unexpected exception type: " << e.what();
    } catch (...) {
        FAIL() << "Unexpected non-std exception thrown";
    }
}
