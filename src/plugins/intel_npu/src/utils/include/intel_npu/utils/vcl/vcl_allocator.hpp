// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "intel_npu/utils/utils.hpp"
#include "intel_npu/utils/vcl/vcl.h"
#include "openvino/core/except.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"

namespace intel_npu {

struct vcl_allocator_3 : vcl_allocator2_t {
    vcl_allocator_3() : vcl_allocator2_t{allocate, deallocate}, m_allocator(intel_npu::utils::STANDARD_PAGE_SIZE) {}

    ~vcl_allocator_3() {
        for (auto& item : m_info) {
            if (item.first) {
                m_allocator.deallocate(item.first, item.second, intel_npu::utils::STANDARD_PAGE_SIZE);
            }
        }
        m_info.clear();
    }

    static uint8_t* allocate(vcl_allocator2_t* allocator, size_t size) noexcept {
        vcl_allocator_3* vclAllocator = static_cast<vcl_allocator_3*>(allocator);
        size_t alignedSize = intel_npu::utils::align_size_to_standard_page_size(size);

        // Prevent integer wraparound on extremely large allocation sizes
        if (alignedSize < size) {
            return nullptr;
        }

        uint8_t* allocatedPtr = nullptr;
        try {
            allocatedPtr = static_cast<uint8_t*>(
                vclAllocator->m_allocator.allocate(alignedSize, intel_npu::utils::STANDARD_PAGE_SIZE));

            if (allocatedPtr == nullptr) {
                return nullptr;
            }
            std::memset(allocatedPtr + size, 0, alignedSize - size);

            vclAllocator->m_info.emplace_back(std::make_pair(allocatedPtr, alignedSize));
            return allocatedPtr;
        } catch (...) {
            if (allocatedPtr != nullptr) {
                vclAllocator->m_allocator.deallocate(allocatedPtr, alignedSize, intel_npu::utils::STANDARD_PAGE_SIZE);
            }
            return nullptr;
        }
    }

    static void deallocate(vcl_allocator2_t* allocator, uint8_t* ptr) noexcept {
        if (ptr == nullptr) {
            return;
        }
        vcl_allocator_3* vclAllocator = static_cast<vcl_allocator_3*>(allocator);

        auto it = std::find_if(vclAllocator->m_info.begin(),
                               vclAllocator->m_info.end(),
                               [ptr](const std::pair<uint8_t*, size_t>& item) {
                                   return item.first == ptr;
                               });
        if (it != vclAllocator->m_info.end()) {
            vclAllocator->m_info.erase(it);
        }

        // 1 is the placeholder value, as size is not needed in deallocate
        vclAllocator->m_allocator.deallocate(ptr, 1, intel_npu::utils::STANDARD_PAGE_SIZE);
    }
    intel_npu::utils::AlignedAllocator m_allocator;
    std::vector<std::pair<uint8_t*, size_t>> m_info;
};

inline ov::Tensor make_tensor_from_aligned_addr(uint8_t* allocated,
                                                size_t size,
                                                std::shared_ptr<vcl_allocator_3> sourceAllocator) {
    auto tensor = ov::Tensor(ov::element::u8, ov::Shape{size}, allocated);
    auto impl = ov::get_tensor_impl(std::move(tensor));
    std::shared_ptr<void> ptr(allocated, [sourceAllocator](uint8_t* p) noexcept {
        if (p == nullptr) {
            return;
        }
        vcl_allocator_3::deallocate(sourceAllocator.get(), p);
    });
    impl._so = std::move(ptr);
    return ov::make_tensor(impl);
}

}  // namespace intel_npu
