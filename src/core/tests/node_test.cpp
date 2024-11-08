// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/node.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"

using namespace testing;

class TestNode : public ov::Node {
public:
    TestNode() : Node() {}

    static const type_info_t& get_type_info_static() {
        static const type_info_t info{"TestNode", ""};
        info.hash();
        return info;
    }

    const type_info_t& get_type_info() const override {
        return get_type_info_static();
    }

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector&) const override {
        return std::make_shared<TestNode>();
    }
};

class NodeValidationFailureTest : public Test {
protected:
    TestNode test_node;
};

TEST_F(NodeValidationFailureTest, node_failure_message) {
    OV_EXPECT_THROW(NODE_VALIDATION_CHECK(&test_node, false, "Test message"),
                    ov::NodeValidationFailure,
                    HasSubstr("':\nTest message"));
}

TEST_F(NodeValidationFailureTest, node_shape_infer_failure_message) {
    const auto input_shapes = std::vector<ov::PartialShape>{{1, 2, 3}, {1}};

    OV_EXPECT_THROW(NODE_SHAPE_INFER_CHECK(&test_node, input_shapes, false, "Test message"),
                    ov::NodeValidationFailure,
                    HasSubstr("':\nShape inference input shapes {[1,2,3],[1]}\nTest message"));
}

TEST_F(NodeValidationFailureTest, create_node_validation_failure) {
    constexpr int line = 145;
    constexpr char test_file[] = "src/test_file.cpp";
    constexpr char check_string[] = "value != 0";
    const std::string explanation = "test error message";
    const std::string ctx_info = "My context";
    const std::string exp_error_msg1 = "Check 'value != 0' failed at src/test_file.cpp:145:\nWhile validating node";
    const std::string exp_error_msg2 = ":\ntest error message\n";

    OV_EXPECT_THROW(ov::NodeValidationFailure::create(test_file, line, check_string, &test_node, explanation),
                    ov::NodeValidationFailure,
                    AllOf(HasSubstr(exp_error_msg1), HasSubstr(exp_error_msg2)));

    OPENVINO_SUPPRESS_DEPRECATED_START
    OV_EXPECT_THROW(ov::NodeValidationFailure::create({test_file, line, check_string}, &test_node, explanation),
                    ov::NodeValidationFailure,
                    AllOf(HasSubstr(exp_error_msg1), HasSubstr(exp_error_msg2)));
    OPENVINO_SUPPRESS_DEPRECATED_END
}

TEST_F(NodeValidationFailureTest, create_node_validation_failure_with_input_shapes) {
    constexpr int line = 145;
    constexpr char test_file[] = "src/test_file.cpp";
    constexpr char check_string[] = "value != 0";
    const std::string explanation = "test error message";
    const std::string ctx_info = "My context";
    const std::string exp_error_msg1 = "Check 'value != 0' failed at src/test_file.cpp:145:\nWhile validating node";
    const std::string exp_error_msg2 = "Shape inference input shapes {[1,2],[4]}\ntest error message\n";
    const auto input_shapes = std::vector<ov::PartialShape>{{1, 2}, {4}};

    OV_EXPECT_THROW(
        ov::NodeValidationFailure::create(test_file,
                                          line,
                                          check_string,
                                          std::make_pair(static_cast<const ov::Node*>(&test_node), &input_shapes),
                                          explanation),
        ov::NodeValidationFailure,
        AllOf(HasSubstr(exp_error_msg1), HasSubstr(exp_error_msg2)));

    OPENVINO_SUPPRESS_DEPRECATED_START
    OV_EXPECT_THROW(
        ov::NodeValidationFailure::create({test_file, line, check_string},
                                          std::make_pair(static_cast<const ov::Node*>(&test_node), &input_shapes),
                                          explanation),
        ov::NodeValidationFailure,
        AllOf(HasSubstr(exp_error_msg1), HasSubstr(exp_error_msg2)));
    OPENVINO_SUPPRESS_DEPRECATED_END
}
