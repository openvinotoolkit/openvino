// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

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

}  // namespace ov::intel_npu::npuw

#define INTEL_NPU_NPUW_DECLARE_PROPERTY_ALIAS(OPT, NS_PATH, NAME) \
    namespace NS_PATH {                                           \
    inline constexpr const auto& NAME = ::ov::intel_npu::detail::npuw_property<::intel_npu::OPT>; \
    }

#define INTEL_NPU_NPUW_IF_BUILD_ALL(OPT, NS_PATH, NAME) INTEL_NPU_NPUW_DECLARE_PROPERTY_ALIAS(OPT, NS_PATH, NAME)
#ifdef NPU_PLUGIN_DEVELOPER_BUILD
#define INTEL_NPU_NPUW_IF_BUILD_DEV(OPT, NS_PATH, NAME) INTEL_NPU_NPUW_DECLARE_PROPERTY_ALIAS(OPT, NS_PATH, NAME)
#else
#define INTEL_NPU_NPUW_IF_BUILD_DEV(OPT, NS_PATH, NAME)
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
