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
