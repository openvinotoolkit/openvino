// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "openvino/core/core.hpp"

namespace {

class QDQStrippingTest : virtual public ov::test::SubgraphBaseStaticTest {
public:

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        function = core->read_model("/home/guest/golubevv/openvino/bin/intel64/RelWithDebInfo/1_Conv.onnx");
    }
};

TEST_F(QDQStrippingTest, Inference) {
    run();
}

}  // namespace
