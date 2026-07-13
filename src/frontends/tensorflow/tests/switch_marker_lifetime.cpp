// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Regression test for a memory leak in the TensorFlow frontend: a Switch node used to store a
// shared_ptr to itself inside its own rt_info conditional-flow marker (CfMarkerType), creating an
// ownership cycle. Such a Switch was never freed and transitively kept the whole TF GraphDef alive
// (via FrameworkNode -> decoder -> GraphDef), which LeakSanitizer reported against generated
// protobuf. The fix stores Switch nodes as weak_ptr, so the marker no longer owns the node.
//
// This test reproduces the exact ownership pattern that propagate_conditional_flow builds for a
// Switch node (see src/tf_utils.cpp) and asserts, via the weak_ptr/expired idiom, that dropping all
// external strong references frees the Switch. With the pre-fix shared_ptr marker the node would
// stay alive (expired() == false); with the weak_ptr marker it is released (expired() == true).

#include <memory>

#include "gtest/gtest.h"
#include "helper_ops/switch.hpp"
#include "openvino/op/parameter.hpp"
#include "tf_utils.hpp"

using namespace ov;
using namespace ov::frontend::tensorflow;

TEST(TFSwitchMarkerLifetime, SwitchSelfMarkerDoesNotLeak) {
    std::weak_ptr<ov::Node> weak_switch;
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{});
        auto pred = std::make_shared<ov::op::v0::Parameter>(element::boolean, PartialShape{});

        const uint32_t switch_marker = 0;
        auto switch_node = std::make_shared<Switch>(data->output(0), pred->output(0), switch_marker);

        // Reproduce the marker that propagate_conditional_flow attaches to a Switch node: the node's
        // own rt_info marker references the Switch via its new_markers set. This is the exact
        // self-reference that used to leak when the set held a shared_ptr.
        CfMarkerType marker;
        marker.new_markers[switch_marker].insert(switch_node);
        set_cf_marker(marker, switch_node);

        weak_switch = switch_node;

        // Sanity: while a strong reference is held the node is obviously alive.
        ASSERT_FALSE(weak_switch.expired());
    }

    // All external strong references are gone. If the marker still owned the Switch (shared_ptr),
    // the self-cycle would keep it alive and this would fail.
    EXPECT_TRUE(weak_switch.expired())
        << "Switch node was not released: its conditional-flow marker still owns it (leak).";
}
