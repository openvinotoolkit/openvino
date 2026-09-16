// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/remote_tensor.hpp"
#include "openvino/runtime/make_tensor.hpp"

using namespace cldnn;
using namespace ov::intel_gpu;
using namespace ::tests;

// IRemoteTensor::data() always throws, so if the fast-path `return` is ever dropped again and
// execution falls through to the fallback path, this call throws instead of silently passing.
TEST(convert_and_copy_test, remote_tensor_fast_path_does_not_fall_through) {
    auto& engine = get_test_engine();
    auto& stream = get_test_stream();

    auto context = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});

    const ov::Shape shape{1, 2, 2, 2};
    const ov::element::Type et = ov::element::f32;

    auto src_remote = std::make_shared<RemoteTensorImpl>(context, shape, et);
    auto src_mem = src_remote->get_original_memory();
    std::vector<float> src_values{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
    set_values(src_mem, src_values);

    layout dst_layout{shape, et, format::bfyx};
    auto dst_mem = engine.allocate_memory(dst_layout);

    OV_ASSERT_NO_THROW(convert_and_copy(src_remote.get(), dst_mem, stream, dst_layout, false));

    cldnn::mem_lock<float, mem_lock_type::read> dst_ptr(dst_mem, stream);
    for (size_t i = 0; i < src_values.size(); ++i) {
        ASSERT_EQ(dst_ptr[i], src_values[i]);
    }
}

TEST(convert_and_copy_test, boolean_padded_source) {
    auto& engine = get_test_engine();
    auto& stream = get_test_stream();

    const ov::Shape shape{1, 1, 2, 2};
    bool src_values[] = {false, false, true, false, false, true};
    auto src_tensor = ov::make_tensor(ov::element::boolean, shape, src_values);

    layout src_layout{shape,
                      data_types::boolean,
                      format::bfyx,
                      padding{{0, 0, 1, 0}, {0, 0, 0, 0}}};
    auto dst_mem = engine.allocate_memory(layout{shape, data_types::boolean, format::bfyx});

    OV_ASSERT_NO_THROW(convert_and_copy(src_tensor.get(), dst_mem, stream, src_layout, false));

    cldnn::mem_lock<uint8_t, mem_lock_type::read> dst_ptr(dst_mem, stream);
    const std::vector<uint8_t> expected{1, 0, 0, 1};
    ASSERT_EQ(std::vector<uint8_t>(dst_ptr.begin(), dst_ptr.end()), expected);
}

TEST(convert_and_copy_test, boolean_transposed_source) {
    auto& engine = get_test_engine();
    auto& stream = get_test_stream();

    const ov::Shape src_shape{1, 1, 2, 3};
    bool src_values[] = {true, false, true, false, true, false};
    auto src_tensor = ov::make_tensor(ov::element::boolean, src_shape, src_values);

    auto dst_mem = engine.allocate_memory(layout{{1, 1, 3, 2}, data_types::boolean, format::bfyx});
    layout src_layout{src_shape, data_types::boolean, format::bfyx};

    OV_ASSERT_NO_THROW(convert_and_copy(src_tensor.get(), dst_mem, stream, src_layout, true));

    cldnn::mem_lock<uint8_t, mem_lock_type::read> dst_ptr(dst_mem, stream);
    const std::vector<uint8_t> expected{1, 0, 0, 1, 1, 0};
    ASSERT_EQ(std::vector<uint8_t>(dst_ptr.begin(), dst_ptr.end()), expected);
}
