// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_eagle3_extension.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "openvino/op/parameter.hpp"

namespace {

using Ports = std::unordered_map<std::string, ov::Output<const ov::Node>>;

class Eagle3ExtensionRoleTest : public ::testing::Test {
protected:
    Ports ports(const std::vector<std::string>& names) {
        Ports result;
        for (const auto& name : names) {
            auto node = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
            node->output(0).set_names({name});
            m_nodes.push_back(node);
            result.emplace(name, ov::Output<const ov::Node>(node, 0));
        }
        return result;
    }

private:
    std::vector<std::shared_ptr<ov::Node>> m_nodes;
};

TEST_F(Eagle3ExtensionRoleTest, TargetWithLastHiddenState) {
    ov::npuw::Eagle3Extension extension;
    extension.initialize(true,
                         ports({"input_ids", "eagle_tree_mask"}),
                         ports({"logits", ov::npuw::Eagle3LayerNames::last_hidden_state}));
    EXPECT_TRUE(extension.is_eagle3_model());
    EXPECT_NE(extension.get_sampling_state(), nullptr);
}

TEST_F(Eagle3ExtensionRoleTest, TargetWithTreeMaskOnly) {
    ov::npuw::Eagle3Extension extension;
    extension.initialize(true, ports({"input_ids", ov::npuw::Eagle3LayerNames::eagle_tree_mask}), ports({"logits"}));
    EXPECT_TRUE(extension.is_eagle3_model());
    EXPECT_NE(extension.get_sampling_state(), nullptr);
}

TEST_F(Eagle3ExtensionRoleTest, DraftModel) {
    ov::npuw::Eagle3Extension extension;
    extension.initialize(true,
                         ports({"input_ids", ov::npuw::Eagle3LayerNames::hidden_states}),
                         ports({"logits", ov::npuw::Eagle3LayerNames::last_hidden_state}));
    EXPECT_TRUE(extension.is_eagle3_model());
}

TEST_F(Eagle3ExtensionRoleTest, RejectsModelWithoutTreeMaskOrLastHiddenState) {
    ov::npuw::Eagle3Extension extension;
    EXPECT_THROW(extension.initialize(true, ports({"input_ids"}), ports({"logits"})), ov::Exception);
}

TEST_F(Eagle3ExtensionRoleTest, RejectsDraftWithoutLastHiddenState) {
    ov::npuw::Eagle3Extension extension;
    EXPECT_THROW(extension.initialize(true,
                                      ports({"input_ids",
                                             ov::npuw::Eagle3LayerNames::hidden_states,
                                             ov::npuw::Eagle3LayerNames::eagle_tree_mask}),
                                      ports({"logits"})),
                 ov::Exception);
}

TEST_F(Eagle3ExtensionRoleTest, NotEagleWhenDisabled) {
    ov::npuw::Eagle3Extension extension;
    extension.initialize(false, ports({"input_ids", ov::npuw::Eagle3LayerNames::eagle_tree_mask}), ports({"logits"}));
    EXPECT_FALSE(extension.is_eagle3_model());
    EXPECT_EQ(extension.get_sampling_state(), nullptr);
}

}  // namespace
