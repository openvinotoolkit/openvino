// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/remote_tensor.hpp"

using namespace cldnn;
using namespace ov::intel_gpu;
using namespace ::tests;

class RemoteTensorStridesTest : public ::testing::Test {
protected:
    void SetUp() override {
#ifdef OV_GPU_WITH_SYCL_RT
        GTEST_SKIP() << "RemoteTensor is not supported with SYCL runtime yet";
#endif
    }
};

TEST_F(RemoteTensorStridesTest, get_strides_byte_granular_type) {
    auto& engine = get_test_engine();
    auto context = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});

    auto tensor = std::make_shared<RemoteTensorImpl>(context, ov::Shape{2, 3}, ov::element::f32);

    EXPECT_EQ(tensor->get_strides(), ov::Strides({12, 4}));
}

TEST_F(RemoteTensorStridesTest, get_strides_dense_sub_byte_type) {
    auto& engine = get_test_engine();
    auto context = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});

    // u4: 100 elements per row, byte-aligned (100 * 4 bits == 50 bytes).
    auto tensor = std::make_shared<RemoteTensorImpl>(context, ov::Shape{2, 100}, ov::element::u4);

    EXPECT_EQ(tensor->get_strides(), ov::Strides({50, 1}));
}

TEST_F(RemoteTensorStridesTest, get_strides_unaligned_sub_byte_shape_is_empty) {
    auto& engine = get_test_engine();
    auto context = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});

    // u4 with an odd last-dimension count (3 * 4 bits == 12 bits) is not byte-aligned, so byte
    // strides are not representable.
    auto tensor = std::make_shared<RemoteTensorImpl>(context, ov::Shape{2, 3}, ov::element::u4);

    EXPECT_TRUE(tensor->get_strides().empty());
}
