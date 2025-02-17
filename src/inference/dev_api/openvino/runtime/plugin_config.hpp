// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <filesystem>
#include <map>
#include <string_view>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/properties.hpp"

#define OV_CONFIG_DECLARE_OPTION(PropertyNamespace, PropertyVar, Visibility, ...)                           \
public:                                                                                                     \
    const decltype(PropertyNamespace::PropertyVar)::value_type& get_##PropertyVar() const {                 \
        if (m_is_finalized) {                                                                               \
            return m_##PropertyVar.value;                                                                   \
        } else {                                                                                            \
            if (m_user_properties.find(PropertyNamespace::PropertyVar.name()) != m_user_properties.end()) { \
                return m_user_properties.at(PropertyNamespace::PropertyVar.name())                          \
                    .as<decltype(PropertyNamespace::PropertyVar)::value_type>();                            \
            } else {                                                                                        \
                return m_##PropertyVar.value;                                                               \
            }                                                                                               \
        }                                                                                                   \
    }                                                                                                       \
                                                                                                            \
private:                                                                                                    \
    ConfigOption<decltype(PropertyNamespace::PropertyVar)::value_type, Visibility> m_##PropertyVar{         \
        this,                                                                                               \
        PropertyNamespace::PropertyVar.name(),                                                              \
        #PropertyNamespace "::" #PropertyVar,                                                               \
        __VA_ARGS__};

#define OV_CONFIG_RELEASE_OPTION(PropertyNamespace, PropertyVar, ...) \
    OV_CONFIG_DECLARE_OPTION(PropertyNamespace, PropertyVar, OptionVisibility::RELEASE, __VA_ARGS__)

#define OV_CONFIG_RELEASE_INTERNAL_OPTION(PropertyNamespace, PropertyVar, ...) \
    OV_CONFIG_DECLARE_OPTION(PropertyNamespace, PropertyVar, OptionVisibility::RELEASE_INTERNAL, __VA_ARGS__)

#ifdef ENABLE_DEBUG_CAPS
#    define OV_CONFIG_DEBUG_GLOBAL_OPTION(PropertyNamespace, PropertyVar, ...)                              \
    public:                                                                                                 \
        static const decltype(PropertyNamespace::PropertyVar)::value_type& get_##PropertyVar() {            \
            static PluginConfig::GlobalOptionInitializer init_helper(PropertyNamespace::PropertyVar.name(), \
                                                                     m_allowed_env_prefix,                  \
                                                                     m_##PropertyVar);                      \
            return init_helper.m_option.value;                                                              \
        }                                                                                                   \
                                                                                                            \
    private:                                                                                                \
        static inline ConfigOption<decltype(PropertyNamespace::PropertyVar)::value_type,                    \
                                   OptionVisibility::DEBUG_GLOBAL>                                          \
            m_##PropertyVar{nullptr,                                                                        \
                            PropertyNamespace::PropertyVar.name(),                                          \
                            #PropertyNamespace "::" #PropertyVar,                                           \
                            __VA_ARGS__};                                                                   \
        OptionRegistrationHelper m_##PropertyVar##_rh{this, PropertyNamespace::PropertyVar.name(), &m_##PropertyVar};

#    define OV_CONFIG_DEBUG_OPTION(PropertyNamespace, PropertyVar, ...) \
        OV_CONFIG_DECLARE_OPTION(PropertyNamespace, PropertyVar, OptionVisibility::DEBUG, __VA_ARGS__)
#else
#    define OV_CONFIG_DEBUG_GLOBAL_OPTION(...)
#    define OV_CONFIG_DEBUG_OPTION(...)
#endif

namespace ov {
enum class OptionVisibility : uint8_t {
    RELEASE = 1 << 0,  // Option can be set for any build type via public interface, environment and config file
    RELEASE_INTERNAL = 1 << 1,  // Option can be set for any build type via environment and config file only
    DEBUG = 1 << 2,             // Option can be set for debug builds only via environment and config file
    DEBUG_GLOBAL = 1 << 3,      // Global option can be set for debug builds only via environment and config file
    ANY = 0xFF,                 // Any visibility is valid
};

inline OptionVisibility operator&(OptionVisibility a, OptionVisibility b) {
    using T = std::underlying_type_t<OptionVisibility>;
    return static_cast<OptionVisibility>(static_cast<T>(a) & static_cast<T>(b));
}

inline OptionVisibility operator|(OptionVisibility a, OptionVisibility b) {
    using T = std::underlying_type_t<OptionVisibility>;
    return static_cast<OptionVisibility>(static_cast<T>(a) | static_cast<T>(b));
}

inline OptionVisibility operator~(OptionVisibility a) {
    using T = std::underlying_type_t<OptionVisibility>;
    return static_cast<OptionVisibility>(~static_cast<T>(a));
}

inline std::ostream& operator<<(std::ostream& os, const OptionVisibility& visibility) {
    switch (visibility) {
    case OptionVisibility::RELEASE:
        os << "RELEASE";
        break;
    case OptionVisibility::RELEASE_INTERNAL:
        os << "RELEASE_INTERNAL";
        break;
    case OptionVisibility::DEBUG:
        os << "DEBUG";
        break;
    case OptionVisibility::DEBUG_GLOBAL:
        os << "DEBUG_GLOBAL";
        break;
    case OptionVisibility::ANY:
        os << "ANY";
        break;
    default:
        os << "UNKNOWN";
        break;
    }

    return os;
}

struct ConfigOptionBase {
    ConfigOptionBase(std::string_view prop_name, std::string_view desc) : property_name(prop_name), description(desc) {}
    virtual ~ConfigOptionBase() = default;

    virtual void set_any(const ov::Any& any) = 0;
    virtual ov::Any get_any() const = 0;
    virtual bool is_valid_value(const ov::Any& val) const = 0;
    virtual OptionVisibility get_visibility() const = 0;

    std::string_view property_name;
    std::string_view description;
};

// Base class for configuration of plugins
// Implementation should provide a list of properties with default values and validators (optional)
// and prepare a map string property name -> ConfigOptionBase pointer
// For the sake of efficiency, we expect that plugin properties are defined as class members of the derived class
// and accessed directly in the plugin's code (i.e. w/o get_property()/set_property() calls)
// get/set property members are provided to handle external property access
// The class provides a helpers to read the properties from configuration file and from environment variables
//
// Expected order of properties resolution:
// 1. Assign default value for each property per device
// 2. Save user properties passed via Core::set_property() call to user_properties
// 3. Save user properties passed via Core::compile_model() call to user_properties
// 4. Apply RT info properties to user_properties if they were not set by user
// 5. Read and apply properties from the config file as user_properties
// 6. Read and apply properties from the the environment variables as user_properties
// 7. Apply user_properties to actual plugin properties
// 8. Update dependant properties if they were not set by user either way
class OPENVINO_RUNTIME_API PluginConfig {
public:
    PluginConfig() {}
    virtual ~PluginConfig() = default;

    // Disable copy and move as we need to setup m_options_map properly and ensure that
    // values are a part of current config object
    PluginConfig(const PluginConfig& other) = delete;
    PluginConfig& operator=(const PluginConfig& other) = delete;
    PluginConfig(PluginConfig&& other) = delete;
    PluginConfig& operator=(PluginConfig&& other) = delete;

    void set_property(const ov::AnyMap& properties);
    void set_user_property(const ov::AnyMap& properties, OptionVisibility allowed_visibility = OptionVisibility::ANY);
    Any get_property(const std::string& name, OptionVisibility allowed_visibility = OptionVisibility::ANY) const;

    template <typename... Properties>
    util::EnableIfAllStringAny<void, Properties...> set_property(Properties&&... properties) {
        set_property(ov::AnyMap{std::forward<Properties>(properties)...});
    }
    template <typename... Properties>
    util::EnableIfAllStringAny<void, Properties...> set_user_property(Properties&&... properties) {
        set_user_property(ov::AnyMap{std::forward<Properties>(properties)...});
    }

    std::string to_string() const;

    void finalize(const IRemoteContext* context, const ov::Model* model);

    bool visit_attributes(ov::AttributeVisitor& visitor);

    void register_option(const std::string& name, ConfigOptionBase* ptr) {
        m_options_map.emplace(name, ptr);
    }

protected:
    template <typename OptionType>
    class GlobalOptionInitializer {
    public:
        GlobalOptionInitializer(const std::string& name, std::string_view prefix, OptionType& option)
            : m_option(option) {
            auto val = PluginConfig::read_env(name, prefix, &option);
            if (!val.empty()) {
                std::cout << "Non default global config value for " << name << " = " << val.template as<std::string>()
                          << std::endl;
                option.set_any(val);
            }
        }

        OptionType& m_option;
    };

    virtual void apply_model_specific_options(const IRemoteContext* context, const ov::Model& model) {}
    void apply_env_options();
    void apply_config_options(std::string_view device_name, std::filesystem::path config_path = "");
    virtual void finalize_impl(const IRemoteContext* context) {}

    template <typename T, PropertyMutability mutability>
    bool is_set_by_user(const ov::Property<T, mutability>& property) const {
        return m_user_properties.find(property.name()) != m_user_properties.end();
    }

    ConfigOptionBase* get_option_ptr(const std::string& name) const {
        auto it = m_options_map.find(name);
        OPENVINO_ASSERT(it != m_options_map.end(), "Option not found: ", name);
        OPENVINO_ASSERT(it->second != nullptr, "Option is invalid: ", name);

        return it->second;
    }

    template <typename T, PropertyMutability mutability>
    void apply_rt_info_property(const ov::Property<T, mutability>& property, const ov::RTMap& rt_info) {
        if (!is_set_by_user(property)) {
            auto rt_info_val = rt_info.find(property.name());
            if (rt_info_val != rt_info.end()) {
                set_user_property({property(rt_info_val->second.template as<T>())}, OptionVisibility::RELEASE);
            }
        }
    }

    ov::AnyMap read_config_file(std::filesystem::path filename, std::string_view target_device_name) const;
    ov::AnyMap read_env() const;
    static ov::Any read_env(const std::string& option_name, std::string_view prefix, const ConfigOptionBase* option);
    void cleanup_unsupported(ov::AnyMap& config) const;

    std::map<std::string, ConfigOptionBase*> m_options_map;
    using OptionMapEntry = decltype(m_options_map)::value_type;

    // List of properties explicitly set by user via Core::set_property() or Core::compile_model() or ov::Model's
    // runtime info
    ov::AnyMap m_user_properties;

    std::string_view get_help_message(const std::string& name = "") const;
    void print_help() const;

    bool m_is_finalized = false;

    inline static const std::string_view m_allowed_env_prefix = "OV_";
};

class OptionRegistrationHelper {
public:
    OptionRegistrationHelper(PluginConfig* config, std::string_view name, ConfigOptionBase* option) {
        if (config)
            config->register_option(std::string{name}, option);
    }
};

template <typename T>
struct TypedOption : public ConfigOptionBase {
    TypedOption(const T& default_val, std::string_view prop_name, std::string_view desc)
        : ConfigOptionBase(prop_name, desc),
          value(default_val) {}
    T value;
};

template <typename T, OptionVisibility visibility_>
struct ConfigOption : public TypedOption<T> {
    ConfigOption(PluginConfig* config,
                 std::string_view name,
                 std::string_view prop_name,
                 const T& default_val,
                 std::string_view desc,
                 std::function<bool(T)> validator = nullptr)
        : TypedOption<T>(default_val, prop_name, desc),
          validator(validator) {
        OptionRegistrationHelper option(config, name, this);
    }
    constexpr static const auto visibility = visibility_;

    void set_any(const ov::Any& any) override {
        if (validator)
            OPENVINO_ASSERT(validator(any.as<T>()), "Invalid value: ", any.as<std::string>());
        this->value = any.as<T>();
    }

    ov::Any get_any() const override {
        return ov::Any(this->value);
    }

    bool is_valid_value(const ov::Any& val) const override {
        try {
            auto v = val.as<T>();
            return validator ? validator(v) : true;
        } catch (std::exception&) {
            return false;
        }
    }

    OptionVisibility get_visibility() const override {
        return visibility;
    }

    operator T() const {
        return this->value;
    }

    ConfigOption& operator=(const T& val) {
        this->value = val;
        return *this;
    }

    template <typename U, typename = std::enable_if_t<std::is_convertible_v<U, T>>>
    bool operator==(const U& val) const {
        return this->value == static_cast<T>(val);
    }

    template <typename U, typename = std::enable_if_t<std::is_convertible_v<U, T>>>
    bool operator!=(const U& val) const {
        return !(*this == val);
    }

private:
    std::function<bool(T)> validator;
};

template <>
class OPENVINO_RUNTIME_API AttributeAdapter<ConfigOptionBase*> : public DirectValueAccessor<ConfigOptionBase*> {
public:
    AttributeAdapter(ConfigOptionBase*& value) : DirectValueAccessor<ConfigOptionBase*>(value) {}

    OPENVINO_RTTI("AttributeAdapter<ConfigOptionBase*>");
};

template <>
class OPENVINO_RUNTIME_API AttributeAdapter<ov::AnyMap> : public DirectValueAccessor<ov::AnyMap> {
public:
    AttributeAdapter(ov::AnyMap& value) : DirectValueAccessor<ov::AnyMap>(value) {}

    OPENVINO_RTTI("AttributeAdapter<ov::AnyMap>");
};

template <typename OStreamType>
class OstreamAttributeVisitor : public ov::AttributeVisitor {
    OStreamType& os;

public:
    OstreamAttributeVisitor(OStreamType& os) : os(os) {}

    void on_adapter(const std::string& name, ov::ValueAccessor<bool>& adapter) override {
        os << adapter.get();
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<void>& adapter) override {
        if (auto a = ov::as_type<ov::AttributeAdapter<ConfigOptionBase*>>(&adapter)) {
            return handle_option(a->get());
        } else if (auto a = ov::as_type<ov::AttributeAdapter<ov::AnyMap>>(&adapter)) {
            const auto& props = a->get();
            os << props.size();
            for (auto& kv : props) {
                os << kv.first << kv.second.as<std::string>();
            }
        } else {
            OPENVINO_THROW("Attribute ", name, " can't be processed\n");
        }
    }

    void handle_option(ConfigOptionBase* option) {
        if (option->get_visibility() == OptionVisibility::RELEASE ||
            option->get_visibility() == OptionVisibility::RELEASE_INTERNAL)
            os << option->get_any().as<std::string>();
    }
};

template <typename IStreamType>
class IstreamAttributeVisitor : public ov::AttributeVisitor {
    IStreamType& is;

public:
    IstreamAttributeVisitor(IStreamType& is) : is(is) {}

    void on_adapter(const std::string& name, ov::ValueAccessor<void>& adapter) override {
        if (auto a = ov::as_type<ov::AttributeAdapter<ConfigOptionBase*>>(&adapter)) {
            return handle_option(a->get());
        } else if (auto a = ov::as_type<ov::AttributeAdapter<ov::AnyMap>>(&adapter)) {
            size_t size;
            is >> size;
            ov::AnyMap props;
            for (size_t i = 0; i < size; i++) {
                std::string name, val;
                is >> name;
                is >> val;
                props[name] = val;
            }
            a->set(props);
        } else {
            OPENVINO_THROW("Attribute ", name, " can't be processed\n");
        }
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<bool>& adapter) override {
        bool val;
        is >> val;
        adapter.set(val);
    }

    void handle_option(ConfigOptionBase* option) {
        if (option->get_visibility() == OptionVisibility::RELEASE ||
            option->get_visibility() == OptionVisibility::RELEASE_INTERNAL) {
            std::string s;
            is >> s;
            if (option->is_valid_value(s))
                option->set_any(s);
        }
    }
};

}  // namespace ov
