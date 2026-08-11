// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <system_error>

#include "openvino/util/demand_pager.hpp"
#include "openvino/util/memory.hpp"

namespace ov::util {

// TODO: implement fault delegation on top of AddVectoredExceptionHandler, see win_mmap_object.cpp for prior art.
// Until then every region has to be populated up front.
struct DemandPager::Impl {};

DemandPager::DemandPager() = default;

DemandPager::~DemandPager() = default;

bool DemandPager::is_available() const noexcept {
    return false;
}

DemandPager::pointer_type DemandPager::reserve(size_type size) noexcept {
    std::error_code ec;
    auto* addr = vm_reserve(size, ec);
    if (addr != nullptr) {
        vm_commit(addr, size, ec);
        if (ec) {
            vm_release(addr, size);
            addr = nullptr;
        }
    }
    return addr;
}

void DemandPager::release(pointer_type addr, size_type size) noexcept {
    vm_release(addr, size);
}

bool DemandPager::register_region(callback_type, void*, pointer_type, size_type) noexcept {
    return false;
}

void DemandPager::unregister_region(pointer_type) noexcept {}

void DemandPager::update_user_data(pointer_type, void*) noexcept {}

bool DemandPager::populate(pointer_type, size_type, const void*) noexcept {
    return false;
}

void DemandPager::evict(pointer_type, size_type) noexcept {}

}  // namespace ov::util
