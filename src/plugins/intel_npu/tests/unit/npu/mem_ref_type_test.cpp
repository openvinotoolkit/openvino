#include "intel_npu/utils/vm/mem_ref_type.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "openvino/runtime/make_tensor.hpp"

namespace {

class NpuMemRefTypeTest : public testing::TestWithParam<ov::element::Type> {};

TEST_P(NpuMemRefTypeTest, SetsContiguousElementStrides) {
    const auto tensor = ov::make_tensor(GetParam(), ov::Shape{2, 3});
    if (GetParam().bitwidth() < 8) {
        EXPECT_THROW(tensor->get_strides(), ov::Exception);
    }

    intel_npu::MemRefType memref;
    ASSERT_NO_THROW(memref.set(tensor->data(), 0, tensor));
    EXPECT_EQ(memref._sizes, (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(memref._strides, (std::vector<int64_t>{3, 1}));
    EXPECT_EQ(memref._data, tensor->data());
}

INSTANTIATE_TEST_SUITE_P(
    HostCompile,
    NpuMemRefTypeTest,
    testing::Values(ov::element::u4, ov::element::i4, ov::element::nf4, ov::element::f16));

TEST(NpuMemRefStridedTensorTest, PreservesNonContiguousF16Strides) {
    std::vector<uint16_t> storage(16);
    const auto tensor = ov::make_tensor(ov::element::f16, ov::Shape{2, 3}, storage.data(), ov::Strides{16, 4});
    intel_npu::MemRefType memref;
    ASSERT_NO_THROW(memref.set(tensor->data(), 0, tensor));
    EXPECT_EQ(memref._strides, (std::vector<int64_t>{8, 2}));
    EXPECT_EQ(memref._data, storage.data());
}

}  // namespace