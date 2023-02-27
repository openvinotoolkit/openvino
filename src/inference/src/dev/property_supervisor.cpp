// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/runtime/property_supervisor.hpp"

#include "ie_metric_helpers.hpp"
#include "ie_plugin_config.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/util/common_util.hpp"

namespace {

static const std::string& copy_property_name(const std::string&, const std::string& sub_str) {
    return sub_str;
}

static ov::PropertyName copy_property_name(const ov::PropertyName& property_name, const std::string& sub_str) {
    return {sub_str, property_name.is_mutable() ? ov::PropertyMutability::RW : ov::PropertyMutability::RO};
}

template <typename T>
static std::vector<T> flatten_supported(const std::vector<T>& properties, bool full_properties_routs = false) {
    std::vector<T> result;
    for (T lhs_property : properties) {
        auto reversed_matches = [](const T& lhs, const T& rhs) {
            if (lhs == rhs)
                return false;
            auto lhs_vec = ov::util::split(lhs, '.');
            auto rhs_vec = ov::util::split(rhs, '.');
            auto mismatch = std::mismatch(lhs_vec.size() < rhs_vec.size() ? lhs_vec.rbegin() : rhs_vec.rbegin(),
                                          lhs_vec.size() < rhs_vec.size() ? lhs_vec.rend() : rhs_vec.rend(),
                                          lhs_vec.size() >= rhs_vec.size() ? lhs_vec.rbegin() : rhs_vec.rbegin());

            return (mismatch.first != (lhs_vec.size() < rhs_vec.size() ? lhs_vec.rbegin() : rhs_vec.rbegin())) &&
                   (mismatch.second != (lhs_vec.size() >= rhs_vec.size() ? lhs_vec.rbegin() : rhs_vec.rbegin()));
        };
        bool any_reversed_matches = std::any_of(properties.begin(), properties.end(), [&](const T& rhs_property) {
            return reversed_matches(lhs_property, rhs_property);
        });
        if (any_reversed_matches) {
            result.emplace_back(std::move(lhs_property));
        } else {
            auto last = lhs_property.find_last_of('.');
            if (last == std::string::npos) {
                result.emplace_back(lhs_property);
            } else {
                if (full_properties_routs) {
                    result.emplace_back(lhs_property);
                    for (auto pos = lhs_property.find('.', 0); pos != std::string::npos;
                         pos = lhs_property.find('.', pos + 1)) {
                        result.emplace_back(copy_property_name(lhs_property, lhs_property.substr(pos + 1)));
                    }
                } else {
                    result.emplace_back(copy_property_name(lhs_property, lhs_property.substr(last + 1)));
                }
            }
        }
    }
    return result;
}

static std::vector<std::string> check_found_pathes(const std::vector<std::vector<std::string>>& found_pathes,
                                                   const std::string& rout,
                                                   bool skip_unsupported = false) {
    if (found_pathes.empty()) {
        if (!skip_unsupported) {
            OPENVINO_UNREACHABLE("Property ", rout, " was not found");
        } else {
            return {};
        }
    } else if (found_pathes.size() > 1) {
        std::stringstream strm;
        strm << "Found ambiguous property names:" << std::endl;
        for (auto&& path : found_pathes) {
            strm << '\t';
            for (auto&& part : path) {
                strm << part << ".";
            }
            strm << std::endl;
        }
        if (!skip_unsupported) {
            OPENVINO_UNREACHABLE(strm.str());
        } else {
            return {};
        }
    }
    return found_pathes.front();
}

static ov::AnyMap flatten_special(const ov::AnyMap& any_map) {
    ov::AnyMap result;
    for (auto&& value : any_map) {
        if (value.second.is<ov::AnyMap>()) {
            auto special_properties = {ov::common_property, ov::legacy_property, ov::internal_property};
            auto is_specail_property_set = std::any_of(std::begin(special_properties),
                                                       std::end(special_properties),
                                                       [&](const ov::NamedProperties& property_set) {
                                                           return property_set.name() == value.first;
                                                       });
            if (is_specail_property_set) {
                for (auto&& value : flatten_special(value.second.as<ov::AnyMap>())) {
                    result.insert(value);
                }
            } else {
                result.emplace(value.first, flatten_special(value.second.as<ov::AnyMap>()));
            }
        } else {
            result.insert(value);
        }
    }
    return result;
}

}  // namespace

/************************************************
 * Private API
 ************************************************/
struct ov::PropertySupervisor::SubAccess : public Access {
    SubAccess(PropertySupervisor property_access, const std::shared_ptr<void>& so)
        : m_property_access{std::move(property_access)},
          m_so{so} {}
    ~SubAccess() {
        m_property_access = {};
        m_so = {};
    }

    PropertySupervisor* sub_access_ptr() override {
        return &m_property_access;
    }
    Any get(const AnyMap&) const override {
        return m_property_access.get();
    }
    // void ro() override {
    //     property_access.ro();
    // }
    PropertySupervisor m_property_access;
    std::shared_ptr<void> m_so;
};

ov::PropertySupervisor::Access::Ptr& ov::PropertySupervisor::find_or_create(const std::string& name) {
    auto in_path = ov::util::split(name, '.');
    std::vector<std::string> path;
    if (!name.empty() && (in_path.front() == name)) {
        path = {in_path.begin() + 1, in_path.end()};
    } else {
        path = in_path;
    }
    if (path.size() > 1) {
        auto it_access = m_accesses.find(path.front());
        if (it_access == m_accesses.end()) {
            add(path.front(), PropertySupervisor{});
        }
        return m_accesses.at(path.front())
            ->sub_access()
            .find_or_create(ov::util::join(std::vector<std::string>{path.begin() + 1, path.end()}, "."));
    } else {
        return m_accesses[in_path.back()];
    }
};

std::vector<ov::PropertyName> ov::PropertySupervisor::get_supported() const {
    std::vector<ov::PropertyName> property_names;
    for (auto&& access : m_accesses) {
        if (access.second->is_sub_access()) {
            auto sub_property_names = access.second->sub_access().get_supported();
            for (auto&& sub_property_name : sub_property_names) {
                property_names.emplace_back(
                    access.first + "." + sub_property_name,
                    sub_property_name.is_mutable() ? PropertyMutability::RW : PropertyMutability::RO);
            }
        } else {
            property_names.emplace_back(access.first,
                                        access.second->is_mutable() ? PropertyMutability::RW : PropertyMutability::RO);
        }
    }
    return property_names;
}

const void* ov::PropertySupervisor::find_access(const std::vector<std::string>& in_path) const {
    std::vector<std::string> path;
    const void* result = nullptr;
    if (!m_name.empty() && (in_path.front() == m_name)) {
        result = this;
        path = {std::next(in_path.begin()), in_path.end()};
    } else {
        path = in_path;
    }
    if (!path.empty()) {
        auto it_access = m_accesses.find(path.front());
        if (it_access == m_accesses.end()) {
            bool found = false;
            for (auto&& property_set : {ov::common_property, ov::legacy_property, ov::internal_property}) {
                auto it_special_access = m_accesses.find(property_set.name());
                if (it_special_access != m_accesses.end()) {
                    it_access = it_special_access->second->sub_access().m_accesses.find(path.front());
                    if (it_access != it_special_access->second->sub_access().m_accesses.end()) {
                        found = true;
                        break;
                    }
                }
            }
            if (!found) {
                return nullptr;
            }
        }
        if (it_access->second->is_sub_access()) {
            if (path.size() > 1) {
                result = it_access->second->sub_access().find_access({std::next(path.begin()), path.end()});
            } else {
                result = it_access->second.get();
            }
        } else {
            result = it_access->second.get();
        }
    }
    return result;
}

void* ov::PropertySupervisor::find_access(const std::vector<std::string>& path) {
    return const_cast<void*>(const_cast<const PropertySupervisor*>(this)->find_access(path));
}

ov::PropertySupervisor* ov::PropertySupervisor::find_property_access(const std::vector<std::string>& in_path) {
    std::vector<std::string> path;
    PropertySupervisor* result = nullptr;
    if (!m_name.empty() && (in_path.front() == m_name)) {
        result = this;
        path = {in_path.begin() + 1, in_path.end()};
    } else {
        path = in_path;
    }
    if (path.size() > 1) {
        auto it_access = m_accesses.find(path.front());
        if (it_access != m_accesses.end()) {
            result = it_access->second->sub_access().find_property_access({path.begin() + 1, path.end()});
        } else {
            result = nullptr;
        }
    } else {
        result = this;
    }
    return result;
};

std::vector<std::vector<std::string>> ov::PropertySupervisor::get_all_pathes() const {
    std::vector<std::vector<std::string>> pathes;
    for (auto&& access : m_accesses) {
        std::vector<std::string> path;
        if (!m_name.empty()) {
            path.push_back(m_name);
        }
        if (access.second->is_sub_access()) {
            auto special_properties = {ov::common_property, ov::legacy_property, ov::internal_property};
            auto is_specail_property_set = std::any_of(std::begin(special_properties),
                                                       std::end(special_properties),
                                                       [&](const NamedProperties& property_set) {
                                                           return property_set.name() == access.first;
                                                       });
            if (!is_specail_property_set) {
                path.push_back(access.first);
            }
            auto sub_pathes = access.second->sub_access().get_all_pathes();
            for (auto&& sub_path : sub_pathes) {
                pathes.emplace_back(path);
                std::move(sub_path.begin(), sub_path.end(), std::back_inserter(pathes.back()));
            }
        } else {
            path.push_back(access.first);
            pathes.emplace_back(std::move(path));
        }
    }
    return pathes;
}

std::vector<std::vector<std::string>> ov::PropertySupervisor::find_property(
    const std::vector<std::string>& rout) const {
    std::vector<std::vector<std::string>> found_pathes;
    for (auto&& path : get_all_pathes()) {
        auto reversed_matches = [&] {
            auto mismatch = std::mismatch(rout.size() < path.size() ? rout.crbegin() : path.crbegin(),
                                          rout.size() < path.size() ? rout.crend() : path.crend(),
                                          rout.size() >= path.size() ? rout.crbegin() : path.crbegin());
            return (mismatch.first != (rout.size() < path.size() ? rout.crbegin() : path.crbegin())) &&
                   (mismatch.second != (rout.size() >= path.size() ? rout.crbegin() : path.crbegin())) &&
                   ((mismatch.first == (rout.size() < path.size() ? rout.crend() : path.crend())) ||
                    (mismatch.second == (rout.size() >= path.size() ? rout.crend() : path.crend())));
        }();
        if (reversed_matches) {
            found_pathes.emplace_back(path);
        }
    }
    if (found_pathes.empty()) {
        if (find_access(rout) != nullptr) {
            found_pathes.push_back(rout);
        }
    }
    return found_pathes;
}

/************************************************
 * Public API
 ************************************************/

ov::PropertySupervisor::PropertySupervisor() {
    add(ov::supported_properties, [this](const AnyMap& args) {
        auto supported_properties = get_supported();
        auto flattened = flatten_supported(supported_properties, util::contains(args, "OV_FULL_PROPERTIES_ROUTS"));
        std::vector<PropertyName> properties;
        for (auto property : flattened) {
            for (std::string str : {std::string{'.'} + ov::legacy_property.name() + '.',
                                    std::string{ov::legacy_property.name()} + '.',
                                    std::string{'.'} + ov::common_property.name() + '.',
                                    std::string{ov::common_property.name()} + '.'}) {
                for (auto i = property.find(str); i != std::string::npos; i = property.find(str)) {
                    property.erase(i, str.size());
                }
            }
            bool skip_internal = false;
            for (std::string str : {std::string{'.'} + ov::internal_property.name() + '.',
                                    std::string{ov::internal_property.name()} + '.'}) {
                for (auto i = property.find(str); i != std::string::npos; i = property.find(str)) {
                    skip_internal = true;
                }
            }
            if (skip_internal) {
                continue;
            }
            properties.emplace_back(std::move(property));
        }
        return properties;
    });

    auto get_legacy_properties = [this](bool mutability) {
        std::vector<std::string> property_names;
        for (auto&& properties_set : {ov::legacy_property, ov::common_property}) {
            auto it_supported = m_accesses.find(properties_set.name());
            if (it_supported != m_accesses.end()) {
                OPENVINO_ASSERT(it_supported->second->is_sub_access());
                for (auto&& supporeted_property : it_supported->second->sub_access().get_supported()) {
                    if (mutability == supporeted_property.is_mutable()) {
                        property_names.emplace_back(supporeted_property);
                    }
                }
            }
        }
        if (!mutability) {
            property_names.emplace_back(METRIC_KEY(SUPPORTED_METRICS));
            property_names.emplace_back(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        }
        return flatten_supported(property_names);
    };
    add(METRIC_KEY(SUPPORTED_METRICS), [get_legacy_properties] {
        return get_legacy_properties(false);
    });
    add(METRIC_KEY(SUPPORTED_CONFIG_KEYS), [get_legacy_properties] {
        return get_legacy_properties(true);
    });
}

bool ov::PropertySupervisor::empty() const {
    auto supported_properties_only = std::all_of(
        m_accesses.begin(),
        m_accesses.end(),
        [](const std::map<std::string, std::shared_ptr<Access>>::value_type& value) {
            return value.first == ov::supported_properties || value.first == METRIC_KEY(SUPPORTED_METRICS) ||
                   value.first == METRIC_KEY(SUPPORTED_CONFIG_KEYS);
        });
    return m_accesses.empty() || supported_properties_only;
}

bool ov::PropertySupervisor::has(const std::string name) const {
    return find_property(util::split(name, '.')).size() == 1;
}

ov::Any ov::PropertySupervisor::get(const std::string& name, const AnyMap& args) const {
    auto names = util::split(name, '.');
    auto access_ptr = find_access(check_found_pathes(find_property(names), name));
    if (access_ptr == this) {
        OPENVINO_ASSERT(names.size() > 1);
        return get(util::join(std::vector<std::string>{names.begin() + 1, names.end()}, "."), args);
    } else {
        if (access_ptr == nullptr)
            OPENVINO_ASSERT(access_ptr != nullptr);
        auto access = static_cast<const Access*>(access_ptr);
        return access->get(args);
    }
}

ov::AnyMap ov::PropertySupervisor::get(const PropertyMutability mutability, bool intially_mutable) const {
    AnyMap result;
    for (auto&& access : m_accesses) {
        if (access.second->is_sub_access()) {
            auto any_map = access.second->sub_access().get(mutability, intially_mutable);
            if (!any_map.empty()) {
                result.emplace(access.first, any_map);
            }
        } else if ((mutability == PropertyMutability::RW &&
                    (intially_mutable ? access.second->is_intially_mutable() : access.second->is_mutable())) ||
                   (mutability == PropertyMutability::RO)) {
            auto any = access.second->get({});
            result.emplace(access.first, any);
        }
    }
    return flatten_special(result);
}

ov::PropertySupervisor& ov::PropertySupervisor::set(const std::string& name,
                                                    const Any& property,
                                                    bool skip_unsupported) {
    auto names = util::split(name, '.');
    auto access_ptr = find_access(check_found_pathes(find_property(names), name, skip_unsupported));
    if (access_ptr == this) {
        return set(util::join(std::vector<std::string>{names.begin() + 1, names.end()}, "."),
                   property,
                   skip_unsupported);
    } else {
        if (skip_unsupported && access_ptr == nullptr) {
            return *this;
        }
        OPENVINO_ASSERT(access_ptr != nullptr);
        auto access = static_cast<Access*>(access_ptr);
        if (access->is_sub_access() && property.is<AnyMap>()) {
            access->sub_access().PropertySupervisor::set(property.as<AnyMap>());
        } else if ((!access->is_sub_access() && property.is<AnyMap>()) ||
                   (access->is_sub_access() && !property.is<AnyMap>())) {
            OPENVINO_UNREACHABLE("Type dose not matches");
        } else {
            if (!access->is_mutable()) {
                OPENVINO_ASSERT(access->is_mutable(), "Property ", name, " is not writeable");
            }
            access->precondition(property);
            access->set(property);
        }
    }
    return *this;
}

ov::PropertySupervisor& ov::PropertySupervisor::set(const AnyMap& properties, bool skip_unsupported) {
    for (auto&& value : properties) {
        set(value.first, value.second, skip_unsupported);
    }
    return *this;
}

ov::AnyMap ov::PropertySupervisor::merge(const AnyMap& properties,
                                         const PropertyMutability mutability,
                                         bool intially_mutable) const {
    AnyMap result = get(mutability, intially_mutable);
    for (auto&& property : properties) {
        auto names = util::split(property.first, '.');
        auto found_properties = find_property(names);
        if (found_properties.size() == 1) {
            auto access_ptr = find_access(found_properties.front());
            if ((access_ptr == this) && property.second.is<AnyMap>()) {
                for (auto&& v : merge(property.second.as<AnyMap>(), mutability, intially_mutable)) {
                    result.emplace(v.first, util::to_string(v.second));
                }
            } else if ((access_ptr == this) && !property.second.is<AnyMap>()) {
                OPENVINO_UNREACHABLE("Could not merge unsupported types");
            } else {
                OPENVINO_ASSERT(access_ptr != nullptr);
                auto access = static_cast<const Access*>(access_ptr);
                if (access->is_sub_access() && property.second.is<AnyMap>()) {
                    auto any_map =
                        access->sub_access().merge(property.second.as<AnyMap>(), mutability, intially_mutable);
                    if (!any_map.empty()) {
                        result[property.first] = any_map;
                    }
                } else if ((!access->is_sub_access() && property.second.is<AnyMap>()) ||
                           (access->is_sub_access() && !property.second.is<AnyMap>())) {
                    OPENVINO_UNREACHABLE("Could not merge unsupported types");
                } else {
                    OPENVINO_ASSERT(intially_mutable ? access->is_intially_mutable() : access->is_mutable(),
                                    "Could not merge not writeable property: ",
                                    property.first);
                    access->precondition(property.second);
                    result[property.first] = property.second;
                }
            }
        } else {
            result.emplace(property.first, property.second);
        }
    }
    return result;
}

ov::PropertySupervisor& ov::PropertySupervisor::remove(const std::string& name) {
    auto names = util::split(name, '.');
    auto property_access = find_property_access(check_found_pathes(find_property(names), name, "skip_unsupported"));
    if (property_access == this) {
        if (names.size() > 1) {
            remove(util::join(std::vector<std::string>{names.begin() + 1, names.end()}, "."));
        } else {
            property_access->m_accesses.erase(names.back());
            return *this;
        }
    } else if (property_access != nullptr) {
        property_access->m_accesses.erase(names.back());
    }
    return *this;
}

ov::PropertySupervisor& ov::PropertySupervisor::add(PropertySupervisor sub_accesses) {
    sub_accesses.remove(ov::supported_properties)
        .remove(METRIC_KEY(SUPPORTED_METRICS))
        .remove(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
    for (auto&& sub_access : sub_accesses.m_accesses) {
        if (sub_access.second->is_sub_access()) {
            add(sub_access.first, sub_access.second->sub_access());
        } else {
            m_accesses.emplace(std::move(sub_access.first), std::move(sub_access.second));
        }
    }
    return *this;
}

ov::PropertySupervisor& ov::PropertySupervisor::add(const std::string& name,
                                                    PropertySupervisor sub_accesses_,
                                                    const std::shared_ptr<void>& so) {
    auto& access = m_accesses[name];
    sub_accesses_.remove(ov::supported_properties)
        .remove(METRIC_KEY(SUPPORTED_METRICS))
        .remove(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
    if (access == nullptr) {
        access = std::make_shared<SubAccess>(std::move(sub_accesses_), so);
    } else {
        access->sub_access().add(std::move(sub_accesses_));
    }
    return *this;
}
