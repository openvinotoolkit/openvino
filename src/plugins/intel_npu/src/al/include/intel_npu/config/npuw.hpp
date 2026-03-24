// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <thread>

#include "intel_npu/config/options.hpp"
#include "intel_npu/npu_private_properties.hpp"

namespace ov {
namespace npuw {
namespace s11n {
// FIXME: likely shouldn't be here as it was initially a part of npuw::s11n
// but we need to somehow serialize AnyMap right here for several properties.
enum class AnyType : int {
    STRING = 0,
    CHARS,
    INT,
    UINT32,
    INT64,
    UINT64,
    SIZET,
    FLOAT,
    BOOL,
    CACHE_MODE,
    ELEMENT_TYPE,
    ANYMAP,
    PERFMODE
};

std::string anyToString(const ov::Any& var);
ov::Any stringToAny(const std::string& var);
std::string anyMapToString(const ov::AnyMap& var);
ov::AnyMap stringToAnyMap(const std::string& var);
}  // namespace s11n
}  // namespace npuw
}  // namespace ov

namespace intel_npu {

void registerNPUWOptions(OptionsDesc& desc);
void registerNPUWLLMOptions(OptionsDesc& desc);
void registerNPUWKokoroOptions(OptionsDesc& desc);

#define DEFINE_NPUW_SIMPLE_OPT(Name, Type, DefaultValue, KeyLiteral) \
    struct Name final : OptionBase<Name, Type> {                     \
        static std::string_view key() {                              \
            return KeyLiteral;                                       \
        }                                                            \
                                                                     \
        static Type defaultValue() {                                 \
            return DefaultValue;                                     \
        }                                                            \
                                                                     \
        static OptionMode mode() {                                   \
            return OptionMode::RunTime;                              \
        }                                                            \
    };

#define DEFINE_NPUW_ANYMAP_OPT(Name, KeyLiteral)                     \
    struct Name final : OptionBase<Name, ov::AnyMap> {               \
        static std::string_view key() {                              \
            return KeyLiteral;                                       \
        }                                                            \
                                                                     \
        static constexpr std::string_view getTypeName() {            \
            return "ov::AnyMap";                                     \
        }                                                            \
                                                                     \
        static ov::AnyMap defaultValue() {                           \
            return {};                                               \
        }                                                            \
                                                                     \
        static ov::AnyMap parse(std::string_view val) {              \
            return ov::npuw::s11n::stringToAnyMap(std::string(val)); \
        }                                                            \
                                                                     \
        static std::string toString(const ov::AnyMap& val) {         \
            return ov::npuw::s11n::anyMapToString(val);              \
        }                                                            \
                                                                     \
        static OptionMode mode() {                                   \
            return OptionMode::RunTime;                              \
        }                                                            \
                                                                     \
        static bool isPublic() {                                     \
            return false;                                            \
        }                                                            \
    };

namespace npuw {
namespace llm {
enum class PrefillHint { DYNAMIC, STATIC };
enum class GenerateHint { FAST_COMPILE, BEST_PERF };
enum class AttentionHint { DYNAMIC, STATIC, PYRAMID, HFA };
enum class MoEHint { DENSE, HOST_ROUTED, DEVICE_ROUTED };
}  // namespace llm
}  // namespace npuw

enum class NPUWOptionGroup { ROOT, LLM, KOKORO };
enum class NPUWOptionSurface { EXPOSED, HIDDEN };
enum class NPUWOptionCaching { CACHED, UNCACHED };
enum class NPUWOptionBuild { ALL, DEV };

template <typename Option>
struct NPUWOptionMeta;

namespace detail {
template <typename T>
struct TypeTag {
    using type = T;
};
}  // namespace detail

#ifdef NPU_PLUGIN_DEVELOPER_BUILD
inline constexpr bool kIsNPUWDeveloperBuild = true;
#else
inline constexpr bool kIsNPUWDeveloperBuild = false;
#endif

template <NPUWOptionBuild Build>
inline constexpr bool kIsNPUWBuildEnabled = Build == NPUWOptionBuild::ALL || kIsNPUWDeveloperBuild;

template <typename EnumType>
struct NPUWStringEnumOptionTraits;

template <>
struct NPUWStringEnumOptionTraits<::intel_npu::npuw::llm::PrefillHint> {
    using ValueType = ::intel_npu::npuw::llm::PrefillHint;

    static constexpr std::string_view getTypeName() {
        return "::intel_npu::npuw::llm::PrefillHint";
    }

    static ValueType defaultValue() {
        return ValueType::DYNAMIC;
    }

    static ValueType parse(std::string_view val) {
        if (val == "DYNAMIC") {
            return ValueType::DYNAMIC;
        } else if (val == "STATIC") {
            return ValueType::STATIC;
        }
        OPENVINO_THROW("Unsupported \"PREFILL_HINT\" provided: ", val);
    }

    static std::string toString(const ValueType& val) {
        switch (val) {
        case ValueType::DYNAMIC:
            return "DYNAMIC";
        case ValueType::STATIC:
            return "STATIC";
        default:
            OPENVINO_THROW("Can't convert provided \"PREFILL_HINT\" : ", int(val), " to string.");
        }
    }
};


template <>
struct NPUWStringEnumOptionTraits<::intel_npu::npuw::llm::GenerateHint> {
    using ValueType = ::intel_npu::npuw::llm::GenerateHint;

    static constexpr std::string_view getTypeName() {
        return "::intel_npu::npuw::llm::GenerateHint";
    }

    static ValueType defaultValue() {
        return ValueType::FAST_COMPILE;
    }

    static ValueType parse(std::string_view val) {
        if (val == "FAST_COMPILE") {
            return ValueType::FAST_COMPILE;
        } else if (val == "BEST_PERF") {
            return ValueType::BEST_PERF;
        }
        OPENVINO_THROW("Unsupported \"GENERATE_HINT\" provided: ",
                       val,
                       ". Please select either \"FAST_COMPILE\" or \"BEST_PERF\".");
    }

    static std::string toString(const ValueType& val) {
        switch (val) {
        case ValueType::FAST_COMPILE:
            return "FAST_COMPILE";
        case ValueType::BEST_PERF:
            return "BEST_PERF";
        default:
            OPENVINO_THROW("Can't convert provided \"GENERATE_HINT\" : ", int(val), " to string.");
        }
    }
};

template <>
struct NPUWStringEnumOptionTraits<::intel_npu::npuw::llm::AttentionHint> {
    using ValueType = ::intel_npu::npuw::llm::AttentionHint;

    static constexpr std::string_view getTypeName() {
        return "::intel_npu::npuw::llm::AttentionHint";
    }

    static ValueType defaultValue() {
        return ValueType::STATIC;
    }

    static ValueType parse(std::string_view val) {
        if (val == "DYNAMIC") {
            return ValueType::DYNAMIC;
        } else if (val == "STATIC") {
            return ValueType::STATIC;
        } else if (val == "PYRAMID") {
            return ValueType::PYRAMID;
        } else if (val == "HFA") {
            return ValueType::HFA;
        }
        OPENVINO_THROW("Unsupported attention hint provided: ", val);
    }

    static std::string toString(const ValueType& val) {
        switch (val) {
        case ValueType::DYNAMIC:
            return "DYNAMIC";
        case ValueType::STATIC:
            return "STATIC";
        case ValueType::PYRAMID:
            return "PYRAMID";
        case ValueType::HFA:
            return "HFA";
        default:
            OPENVINO_THROW("Can't convert provided attention hint : ", int(val), " to string.");
        }
    }
};

struct PrefillAttentionHintOptionTraits : NPUWStringEnumOptionTraits<::intel_npu::npuw::llm::AttentionHint> {
    static ValueType defaultValue() {
        return ValueType::PYRAMID;
    }
};

struct GenerateAttentionHintOptionTraits : NPUWStringEnumOptionTraits<::intel_npu::npuw::llm::AttentionHint> {
    static ValueType defaultValue() {
        return ValueType::STATIC;
    }
};

template <>
struct NPUWStringEnumOptionTraits<::intel_npu::npuw::llm::MoEHint> {
    using ValueType = ::intel_npu::npuw::llm::MoEHint;

    static constexpr std::string_view getTypeName() {
        return "::intel_npu::npuw::llm::MoEHint";
    }

    static ValueType defaultValue() {
        return ValueType::HOST_ROUTED;
    }

    static ValueType parse(std::string_view val) {
        if (val == "DENSE") {
            return ValueType::DENSE;
        } else if (val == "HOST_ROUTED") {
            return ValueType::HOST_ROUTED;
        } else if (val == "DEVICE_ROUTED") {
            return ValueType::DEVICE_ROUTED;
        }
        OPENVINO_THROW("Unsupported MoE hint provided: ", val);
    }

    static std::string toString(const ValueType& val) {
        switch (val) {
        case ValueType::DENSE:
            return "DENSE";
        case ValueType::HOST_ROUTED:
            return "HOST_ROUTED";
        case ValueType::DEVICE_ROUTED:
            return "DEVICE_ROUTED";
        default:
            OPENVINO_THROW("Can't convert provided MoE hint : ", int(val), " to string.");
        }
    }
};

template <class ActualOpt, class Traits>
struct NPUWStringEnumOptionBase : OptionBase<ActualOpt, typename Traits::ValueType> {
    using ValueType = typename Traits::ValueType;

    static constexpr std::string_view getTypeName() {
        return Traits::getTypeName();
    }

    static ValueType defaultValue() {
        return Traits::defaultValue();
    }

    static ValueType parse(std::string_view val) {
        return Traits::parse(val);
    }

    static std::string toString(const ValueType& val) {
        return Traits::toString(val);
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }

    static bool isPublic() {
        return false;
    }
};

#define DEFINE_NPUW_STRING_ENUM_OPT(Name, Traits, KeyLiteral)    \
    struct Name final : NPUWStringEnumOptionBase<Name, Traits> { \
        static std::string_view key() {                          \
            return KeyLiteral;                                   \
        }                                                        \
    };

#define INTEL_NPU_NPUW_SIMPLE_OPT(OPT, TYPE, DEFAULT, NS, VARNAME, KEY, GROUP, SURFACE, CACHING, BUILD) \
    DEFINE_NPUW_SIMPLE_OPT(OPT, TYPE, DEFAULT, KEY)
#define INTEL_NPU_NPUW_STRING_ENUM_OPT(OPT, TYPE, TRAITS, NS, VARNAME, KEY, GROUP, SURFACE, CACHING, BUILD) \
    DEFINE_NPUW_STRING_ENUM_OPT(OPT, TRAITS, KEY)
#define INTEL_NPU_NPUW_ANYMAP_OPT(OPT, NS, VARNAME, KEY, GROUP, SURFACE, CACHING, BUILD) \
    DEFINE_NPUW_ANYMAP_OPT(OPT, KEY)
#include "intel_npu/config/npuw_option_defs.inc"
#undef INTEL_NPU_NPUW_SIMPLE_OPT
#undef INTEL_NPU_NPUW_STRING_ENUM_OPT
#undef INTEL_NPU_NPUW_ANYMAP_OPT
#undef DEFINE_NPUW_STRING_ENUM_OPT
#undef DEFINE_NPUW_ANYMAP_OPT
#undef DEFINE_NPUW_SIMPLE_OPT

#define INTEL_NPU_NPUW_SIMPLE_OPT(OPT, TYPE, DEFAULT, NS, VARNAME, KEY, GROUP, SURFACE, CACHING, BUILD) \
    template <>                                                                                         \
    struct NPUWOptionMeta<OPT> {                                                                        \
        using ConfigValueType = TYPE;                                                                   \
        using PropertyValueType = TYPE;                                                                 \
        static constexpr auto group = NPUWOptionGroup::GROUP;                                           \
        static constexpr auto surface = NPUWOptionSurface::SURFACE;                                     \
        static constexpr auto caching = NPUWOptionCaching::CACHING;                                     \
        static constexpr auto build = NPUWOptionBuild::BUILD;                                           \
        static constexpr std::string_view key() {                                                       \
            return KEY;                                                                                 \
        }                                                                                               \
    };
#define INTEL_NPU_NPUW_STRING_ENUM_OPT(OPT, TYPE, TRAITS, NS, VARNAME, KEY, GROUP, SURFACE, CACHING, BUILD) \
    template <>                                                                                             \
    struct NPUWOptionMeta<OPT> {                                                                            \
        using ConfigValueType = TYPE;                                                                       \
        using PropertyValueType = std::string;                                                              \
        static constexpr auto group = NPUWOptionGroup::GROUP;                                               \
        static constexpr auto surface = NPUWOptionSurface::SURFACE;                                         \
        static constexpr auto caching = NPUWOptionCaching::CACHING;                                         \
        static constexpr auto build = NPUWOptionBuild::BUILD;                                               \
        static constexpr std::string_view key() {                                                           \
            return KEY;                                                                                     \
        }                                                                                                   \
    };
#define INTEL_NPU_NPUW_ANYMAP_OPT(OPT, NS, VARNAME, KEY, GROUP, SURFACE, CACHING, BUILD) \
    template <>                                                                          \
    struct NPUWOptionMeta<OPT> {                                                         \
        using ConfigValueType = ov::AnyMap;                                              \
        using PropertyValueType = ov::AnyMap;                                            \
        static constexpr auto group = NPUWOptionGroup::GROUP;                            \
        static constexpr auto surface = NPUWOptionSurface::SURFACE;                      \
        static constexpr auto caching = NPUWOptionCaching::CACHING;                      \
        static constexpr auto build = NPUWOptionBuild::BUILD;                            \
        static constexpr std::string_view key() {                                        \
            return KEY;                                                                  \
        }                                                                                \
    };
#include "intel_npu/config/npuw_option_defs.inc"
#undef INTEL_NPU_NPUW_ANYMAP_OPT
#undef INTEL_NPU_NPUW_STRING_ENUM_OPT
#undef INTEL_NPU_NPUW_SIMPLE_OPT

template <NPUWOptionGroup Group, typename Fn>
void for_each_npuw_option(Fn&& fn) {
#define INTEL_NPU_NPUW_SIMPLE_OPT(OPT, TYPE, DEFAULT, NS, VARNAME, KEY, ENTRY_GROUP, SURFACE, CACHING, BUILD) \
    if constexpr (Group == NPUWOptionGroup::ENTRY_GROUP && kIsNPUWBuildEnabled<NPUWOptionBuild::BUILD>) {     \
        fn(detail::TypeTag<OPT>{});                                                                           \
    }
#define INTEL_NPU_NPUW_STRING_ENUM_OPT(OPT, TYPE, TRAITS, NS, VARNAME, KEY, ENTRY_GROUP, SURFACE, CACHING, BUILD) \
    if constexpr (Group == NPUWOptionGroup::ENTRY_GROUP && kIsNPUWBuildEnabled<NPUWOptionBuild::BUILD>) {         \
        fn(detail::TypeTag<OPT>{});                                                                               \
    }
#define INTEL_NPU_NPUW_ANYMAP_OPT(OPT, NS, VARNAME, KEY, ENTRY_GROUP, SURFACE, CACHING, BUILD)            \
    if constexpr (Group == NPUWOptionGroup::ENTRY_GROUP && kIsNPUWBuildEnabled<NPUWOptionBuild::BUILD>) { \
        fn(detail::TypeTag<OPT>{});                                                                       \
    }
#include "intel_npu/config/npuw_option_defs.inc"
#undef INTEL_NPU_NPUW_ANYMAP_OPT
#undef INTEL_NPU_NPUW_STRING_ENUM_OPT
#undef INTEL_NPU_NPUW_SIMPLE_OPT
}

template <typename Fn>
void for_each_exposed_npuw_option(Fn&& fn) {
#define INTEL_NPU_NPUW_SIMPLE_OPT(OPT, TYPE, DEFAULT, NS, VARNAME, KEY, GROUP, SURFACE, CACHING, BUILD) \
    if constexpr (NPUWOptionSurface::SURFACE == NPUWOptionSurface::EXPOSED &&                           \
                  kIsNPUWBuildEnabled<NPUWOptionBuild::BUILD>) {                                        \
        fn(detail::TypeTag<OPT>{});                                                                     \
    }
#define INTEL_NPU_NPUW_STRING_ENUM_OPT(OPT, TYPE, TRAITS, NS, VARNAME, KEY, GROUP, SURFACE, CACHING, BUILD) \
    if constexpr (NPUWOptionSurface::SURFACE == NPUWOptionSurface::EXPOSED &&                               \
                  kIsNPUWBuildEnabled<NPUWOptionBuild::BUILD>) {                                            \
        fn(detail::TypeTag<OPT>{});                                                                         \
    }
#define INTEL_NPU_NPUW_ANYMAP_OPT(OPT, NS, VARNAME, KEY, GROUP, SURFACE, CACHING, BUILD) \
    if constexpr (NPUWOptionSurface::SURFACE == NPUWOptionSurface::EXPOSED &&            \
                  kIsNPUWBuildEnabled<NPUWOptionBuild::BUILD>) {                         \
        fn(detail::TypeTag<OPT>{});                                                      \
    }
#include "intel_npu/config/npuw_option_defs.inc"
#undef INTEL_NPU_NPUW_ANYMAP_OPT
#undef INTEL_NPU_NPUW_STRING_ENUM_OPT
#undef INTEL_NPU_NPUW_SIMPLE_OPT
}

template <typename Fn>
void for_each_cached_npuw_option(Fn&& fn) {
#define INTEL_NPU_NPUW_SIMPLE_OPT(OPT, TYPE, DEFAULT, NS, VARNAME, KEY, GROUP, SURFACE, CACHING, BUILD) \
    if constexpr (NPUWOptionCaching::CACHING == NPUWOptionCaching::CACHED &&                            \
                  kIsNPUWBuildEnabled<NPUWOptionBuild::BUILD>) {                                        \
        fn(detail::TypeTag<OPT>{});                                                                     \
    }
#define INTEL_NPU_NPUW_STRING_ENUM_OPT(OPT, TYPE, TRAITS, NS, VARNAME, KEY, GROUP, SURFACE, CACHING, BUILD) \
    if constexpr (NPUWOptionCaching::CACHING == NPUWOptionCaching::CACHED &&                                \
                  kIsNPUWBuildEnabled<NPUWOptionBuild::BUILD>) {                                            \
        fn(detail::TypeTag<OPT>{});                                                                         \
    }
#define INTEL_NPU_NPUW_ANYMAP_OPT(OPT, NS, VARNAME, KEY, GROUP, SURFACE, CACHING, BUILD) \
    if constexpr (NPUWOptionCaching::CACHING == NPUWOptionCaching::CACHED &&             \
                  kIsNPUWBuildEnabled<NPUWOptionBuild::BUILD>) {                         \
        fn(detail::TypeTag<OPT>{});                                                      \
    }
#include "intel_npu/config/npuw_option_defs.inc"
#undef INTEL_NPU_NPUW_ANYMAP_OPT
#undef INTEL_NPU_NPUW_STRING_ENUM_OPT
#undef INTEL_NPU_NPUW_SIMPLE_OPT
}

}  // namespace intel_npu

#include "intel_npu/npuw_private_properties.hpp"
