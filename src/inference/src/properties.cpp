// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "properties.hpp"

#include <algorithm>

#include "cpp_interfaces/interface/internal_properties.hpp"
#include "ie_metric_helpers.hpp"
#include "ie_plugin_config.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {
struct PropertyAccess::SubAccess : public Access {
    SubAccess(PropertyAccess property_access_, const std::shared_ptr<void>& so_)
        : property_access{std::move(property_access_)},
          so{so_} {}
    ~SubAccess() {
        property_access = {};
        so = {};
    }

    PropertyAccess* sub_access_ptr() override {
        return &property_access;
    }
    Any get(const AnyMap&) const override {
        return property_access.get();
    }
    void ro() override {
        property_access.ro();
    }
    PropertyAccess property_access;
    std::shared_ptr<void> so;
};

static const std::string& copy_property_name(const std::string&, const std::string& sub_str) {
    return sub_str;
}

static PropertyName copy_property_name(const PropertyName& property_name, const std::string& sub_str) {
    return {sub_str, property_name.is_mutable() ? PropertyMutability::RW : PropertyMutability::RO};
}

template <typename T>
static std::vector<T> flatten_supported(const std::vector<T>& properties, bool full_properties_routs = false) {
    std::vector<T> result;
    for (T lhs_property : properties) {
        auto reversed_matches = [](const T& lhs, const T& rhs) {
            if (lhs == rhs)
                return false;
            auto lhs_vec = util::split(lhs, '.');
            auto rhs_vec = util::split(rhs, '.');
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

PropertyAccess::PropertyAccess() {
    add(ov::supported_properties, [this](const AnyMap& args) {
        auto supporeted_properties = get_supported();
        auto flattened = flatten_supported(supporeted_properties, util::contains(args, "OV_FULL_PROPERTIES_ROUTS"));
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
            auto it_supported = accesses.find(properties_set.name());
            if (it_supported != accesses.end()) {
                OPENVINO_ASSERT(it_supported->second->is_sub_access());
                for (auto&& supporeted_property : it_supported->second->sub_access().get_supported()) {
                    if (mutability == supporeted_property.is_mutable()) {
                        property_names.emplace_back(supporeted_property);
                    }
                }
            }
        }
        return flatten_supported(property_names);
    };
    add(METRIC_KEY(SUPPORTED_METRICS), [this, get_legacy_properties] {
        return get_legacy_properties(false);
    });
    add(METRIC_KEY(SUPPORTED_CONFIG_KEYS), [this, get_legacy_properties] {
        return get_legacy_properties(true);
    });
}

PropertyAccess& PropertyAccess::add(PropertyAccess sub_accesses) {
    sub_accesses.remove(ov::supported_properties)
        .remove(METRIC_KEY(SUPPORTED_METRICS))
        .remove(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
    for (auto&& sub_access : sub_accesses.accesses) {
        if (sub_access.second->is_sub_access()) {
            add(sub_access.first, sub_access.second->sub_access());
        } else {
            accesses.emplace(std::move(sub_access.first), std::move(sub_access.second));
        }
    }
    return *this;
}

PropertyAccess& PropertyAccess::add(const std::string& name,
                                    PropertyAccess sub_accesses_,
                                    const std::shared_ptr<void>& so) {
    auto& access = accesses[name];
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

PropertyAccess& PropertyAccess::add(const NamedProperties& named_properties,
                                    PropertyAccess sub_accesses,
                                    const std::shared_ptr<void>& so) {
    return add(named_properties.name(), std::move(sub_accesses), so);
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
            strm.seekp(-1, strm.cur);
        }
        if (!skip_unsupported) {
            OPENVINO_UNREACHABLE(strm.str());
        } else {
            return {};
        }
    }
    return found_pathes.front();
}

PropertyAccess& PropertyAccess::remove(const std::string& name) {
    auto names = util::split(name, '.');
    auto property_access = find_property_access(check_found_pathes(find_property(names), name, "skip_unsupported"));
    if (property_access == this) {
        if (names.size() > 1) {
            remove(util::join(std::vector<std::string>{names.begin() + 1, names.end()}, "."));
        } else {
            property_access->accesses.erase(names.back());
            return *this;
        }
    } else if (property_access != nullptr) {
        property_access->accesses.erase(names.back());
    }
    return *this;
}

PropertyAccess& PropertyAccess::ro() {
    for (auto&& access : accesses) {
        if (access.second->is_sub_access()) {
            access.second->sub_access().ro();
        } else {
            access.second->ro();
        }
    }
    return *this;
}

PropertyAccess& PropertyAccess::ro(const std::string& name) {
    auto access_ptr = find_access(check_found_pathes(find_property(util::split(name, '.')), name));
    if (access_ptr == this) {
        return ro();
    } else {
        OPENVINO_ASSERT(access_ptr != nullptr);
        static_cast<Access*>(access_ptr)->ro();
        return *this;
    }
}

bool PropertyAccess::has(const std::string name) const {
    return find_property(util::split(name, '.')).size() == 1;
}

std::vector<std::vector<std::string>> PropertyAccess::get_all_pathes() const {
    std::vector<std::vector<std::string>> pathes;
    for (auto&& access : accesses) {
        std::vector<std::string> path;
        if (!name.empty()) {
            path.push_back(name);
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

PropertyAccess::Access::Ptr& PropertyAccess::find_or_create(const std::string& name) {
    auto in_path = util::split(name, '.');
    std::vector<std::string> path;
    if (!name.empty() && (in_path.front() == name)) {
        path = {in_path.begin() + 1, in_path.end()};
    } else {
        path = in_path;
    }
    if (path.size() > 1) {
        auto it_access = accesses.find(path.front());
        if (it_access == accesses.end()) {
            add(path.front(), PropertyAccess{});
        }
        return accesses.at(path.front())
            ->sub_access()
            .find_or_create(util::join(std::vector<std::string>{path.begin() + 1, path.end()}, "."));
    } else {
        return accesses[in_path.back()];
    }
};

PropertyAccess* PropertyAccess::find_property_access(const std::vector<std::string>& in_path) {
    std::vector<std::string> path;
    PropertyAccess* result = nullptr;
    if (!name.empty() && (in_path.front() == name)) {
        result = this;
        path = {in_path.begin() + 1, in_path.end()};
    } else {
        path = in_path;
    }
    if (path.size() > 1) {
        auto it_access = accesses.find(path.front());
        if (it_access != accesses.end()) {
            result = it_access->second->sub_access().find_property_access({path.begin() + 1, path.end()});
        } else {
            result = nullptr;
        }
    } else {
        result = this;
    }
    return result;
};

const void* PropertyAccess::find_access(const std::vector<std::string>& in_path) const {
    std::vector<std::string> path;
    const void* result = nullptr;
    if (!name.empty() && (in_path.front() == name)) {
        result = this;
        path = {std::next(in_path.begin()), in_path.end()};
    } else {
        path = in_path;
    }
    if (!path.empty()) {
        auto it_access = accesses.find(path.front());
        if (it_access == accesses.end()) {
            bool found = false;
            for (auto&& property_set : {ov::common_property, ov::legacy_property, ov::internal_property}) {
                auto it_special_access = accesses.find(property_set.name());
                if (it_special_access != accesses.end()) {
                    it_access = it_special_access->second->sub_access().accesses.find(path.front());
                    if (it_access != it_special_access->second->sub_access().accesses.end()) {
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
};

void* PropertyAccess::find_access(const std::vector<std::string>& path) {
    return const_cast<void*>(const_cast<const PropertyAccess*>(this)->find_access(path));
}

std::vector<std::vector<std::string>> PropertyAccess::find_property(const std::vector<std::string>& rout) const {
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

std::vector<PropertyName> PropertyAccess::get_supported() const {
    std::vector<PropertyName> property_names;
    for (auto&& access : accesses) {
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

static AnyMap flatten_special(const AnyMap& any_map) {
    AnyMap result;
    for (auto&& value : any_map) {
        if (value.second.is<AnyMap>()) {
            auto special_properties = {ov::common_property, ov::legacy_property, ov::internal_property};
            auto is_specail_property_set = std::any_of(std::begin(special_properties),
                                                       std::end(special_properties),
                                                       [&](const NamedProperties& property_set) {
                                                           return property_set.name() == value.first;
                                                       });
            if (is_specail_property_set) {
                for (auto&& value : flatten_special(value.second.as<AnyMap>())) {
                    result.insert(value);
                }
            } else {
                result.emplace(value.first, flatten_special(value.second.as<AnyMap>()));
            }
        } else {
            result.insert(value);
        }
    }
    return result;
};

AnyMap PropertyAccess::get(const PropertyMutability mutability, bool intially_mutable) const {
    AnyMap result;
    for (auto&& access : accesses) {
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

Any PropertyAccess::get(const std::string& name, const AnyMap& args) const {
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

PropertyAccess& PropertyAccess::set(const std::string& name, const Any& property, bool skip_unsupported) {
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
            access->sub_access().PropertyAccess::set(property.as<AnyMap>());
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

PropertyAccess& PropertyAccess::set(const AnyMap& properties, bool skip_unsupported) {
    for (auto&& value : properties) {
        set(value.first, value.second, skip_unsupported);
    }
    return *this;
}

static Any& find_ref(const std::string& name, AnyMap& any_map, const std::vector<std::string>& path_) {
    auto path = path_;
    if (path.front() == name) {
        path = {path.begin() + 1, path.end()};
    }
    if (path.size() > 1) {
        auto it_any = any_map.find(path.front());
        if (it_any == any_map.end())
            OPENVINO_ASSERT(it_any != any_map.end());
        return find_ref(name, it_any->second->as<AnyMap>(), {path.begin() + 1, path.end()});
    } else if (path.size() == 1) {
        auto it_any = any_map.find(path.front());
        if (it_any == any_map.end())
            OPENVINO_ASSERT(it_any != any_map.end());
        return it_any->second;
    } else {
        OPENVINO_UNREACHABLE("Path not found");
    }
};

PropertyAccess& PropertyAccess::set(const std::map<std::string, std::string>& properties, bool skip_unsupported) {
    auto current_properties = flatten_special(get(PropertyMutability::RW, !"intially_mutable"));
    for (auto&& property : properties) {
        auto found_properties = find_property(util::split(property.first, '.'));
        if (found_properties.size() == 1) {
            auto& any = find_ref(name, current_properties, found_properties.front());
            std::stringstream strm{property.second};
            any.read(strm);
        } else if (!skip_unsupported) {
            OPENVINO_UNREACHABLE("Property with name was not found: ", property.first);
        }
    }
    return set(current_properties, skip_unsupported);
}

AnyMap PropertyAccess::merge(const AnyMap& properties,
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

std::map<std::string, std::string> PropertyAccess::merge(const std::map<std::string, std::string>& properties,
                                                         const PropertyMutability mutability,
                                                         bool intially_mutable) const {
    std::map<std::string, std::string> result;
    auto current_properties = get(mutability, intially_mutable);
    for (auto&& property : properties) {
        auto found_properties = find_property(util::split(property.first, '.'));
        if (found_properties.size() == 1) {
            auto& any = find_ref(name, current_properties, found_properties.front());
            std::stringstream strm{property.second};
            any.read(strm);
        } else {
            result.emplace(property.first, property.second);
        }
    }
    for (auto value : current_properties) {
        auto str = value.second.as<std::string>();
        if (!str.empty()) {
            result.emplace(value.first, value.second.as<std::string>());
        }
    }
    return result;
}

bool PropertyAccess::empty() const {
    auto supported_properties_only =
        std::all_of(accesses.begin(), accesses.end(), [](const AccessMap::value_type& value) {
            return value.first == ov::supported_properties || value.first == METRIC_KEY(SUPPORTED_METRICS) ||
                   value.first == METRIC_KEY(SUPPORTED_CONFIG_KEYS);
        });
    return accesses.empty() || supported_properties_only;
}

}  // namespace ov