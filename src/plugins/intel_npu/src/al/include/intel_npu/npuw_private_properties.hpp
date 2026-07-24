// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "intel_npu/config/npuw.hpp"
#include "openvino/runtime/file_handle.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

namespace ov::intel_npu::detail {

template <typename Option>
using NPUWProperty = ov::Property<typename ::intel_npu::NPUWOptionMeta<Option>::PropertyValueType>;

template <typename Option>
inline constexpr NPUWProperty<Option> npuw_property{::intel_npu::NPUWOptionMeta<Option>::key().data()};

}  // namespace ov::intel_npu::detail

namespace ov::intel_npu::npuw {

inline constexpr ov::Property<ov::FileHandleProvider> weights_handle_provider{"NPUW_WEIGHTS_HANDLE_PROVIDER"};

// Sub-region of the handle returned by NPUW_WEIGHTS_HANDLE_PROVIDER that holds
// the weights pool. When the size is non-zero, NPUW maps only
// [offset, offset+size) out of the handle instead of the whole file, so the
// mapped base coincides with the pool start and per-constant descriptor offsets
// (which are pool-relative) resolve as mapped->data() + offset. See fd-backed
// weight sharing (Option B): the pool is a region embedded inside a larger
// model file. Two scalar properties (rather than a struct) so callers outside
// this plugin can set them without depending on a private type.
inline constexpr ov::Property<std::size_t> weights_handle_region_offset{"NPUW_WEIGHTS_HANDLE_REGION_OFFSET"};
inline constexpr ov::Property<std::size_t> weights_handle_region_size{"NPUW_WEIGHTS_HANDLE_REGION_SIZE"};

// Buffer-backed weight sharing (fd == -1): the weights pool is an already
// resident host region rather than a file to mmap. The caller supplies the
// pool base pointer, its size, and a keep-alive shared_ptr that guarantees the
// bytes outlive the compiled model. NPUW wraps this as a MappedMemory
// (HostRegionMemory) and feeds it through the same weightless import as the fd
// path. The pointer is carried as a uintptr_t so it round-trips through ov::Any
// on any platform. host_region_size == 0 disables this source.
//
// These are mutually exclusive with the fd handle provider; the pointer already
// addresses the pool start, so no region offset is needed here.
inline constexpr ov::Property<std::uintptr_t> weights_host_region_ptr{"NPUW_WEIGHTS_HOST_REGION_PTR"};
inline constexpr ov::Property<std::size_t> weights_host_region_size{"NPUW_WEIGHTS_HOST_REGION_SIZE"};
inline constexpr ov::Property<std::shared_ptr<void>> weights_host_region_keepalive{
    "NPUW_WEIGHTS_HOST_REGION_KEEPALIVE"};

}  // namespace ov::intel_npu::npuw

#define INTEL_NPU_NPUW_DECLARE_PROPERTY_ALIAS(OPT, NS_PATH, NAME)                                 \
    namespace NS_PATH {                                                                           \
    inline constexpr const auto& NAME = ::ov::intel_npu::detail::npuw_property<::intel_npu::OPT>; \
    }

#define INTEL_NPU_NPUW_IF_BUILD_ALL(OPT, NS_PATH, NAME) INTEL_NPU_NPUW_DECLARE_PROPERTY_ALIAS(OPT, NS_PATH, NAME)
#ifdef NPU_PLUGIN_DEVELOPER_BUILD
#    define INTEL_NPU_NPUW_IF_BUILD_DEV(OPT, NS_PATH, NAME) INTEL_NPU_NPUW_DECLARE_PROPERTY_ALIAS(OPT, NS_PATH, NAME)
#else
#    define INTEL_NPU_NPUW_IF_BUILD_DEV(OPT, NS_PATH, NAME)
#endif

#define INTEL_NPU_NPUW_SIMPLE_OPT(OPT, TYPE, DEFAULT, NS_PATH, NAME, KEY, GROUP, SURFACE, CACHING, BUILD) \
    INTEL_NPU_NPUW_IF_BUILD_##BUILD(OPT, NS_PATH, NAME)
#define INTEL_NPU_NPUW_STRING_ENUM_OPT(OPT, TYPE, TRAITS, NS_PATH, NAME, KEY, GROUP, SURFACE, CACHING, BUILD) \
    INTEL_NPU_NPUW_IF_BUILD_##BUILD(OPT, NS_PATH, NAME)
#define INTEL_NPU_NPUW_ANYMAP_OPT(OPT, NS_PATH, NAME, KEY, GROUP, SURFACE, CACHING, BUILD) \
    INTEL_NPU_NPUW_IF_BUILD_##BUILD(OPT, NS_PATH, NAME)
#include "intel_npu/config/npuw_option_defs.inc"
#undef INTEL_NPU_NPUW_ANYMAP_OPT
#undef INTEL_NPU_NPUW_STRING_ENUM_OPT
#undef INTEL_NPU_NPUW_SIMPLE_OPT
#undef INTEL_NPU_NPUW_IF_BUILD_DEV
#undef INTEL_NPU_NPUW_IF_BUILD_ALL
#undef INTEL_NPU_NPUW_DECLARE_PROPERTY_ALIAS
