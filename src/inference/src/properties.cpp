// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "properties.hpp"

#include <algorithm>

#include "ie_metric_helpers.hpp"
#include "ie_plugin_config.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {
struct PropertyAccess::SubAccess : public Access {
    SubAccess(PropertyAccess property_access_, const std::shared_ptr<void>& so_) :
    property_access{std::move(property_access_)},
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
                        result.emplace_back(lhs_property.substr(pos + 1));
                    }
                } else {
                    result.emplace_back(lhs_property.substr(last + 1));
                }
            }
        }
    }
    return result;
}

PropertyAccess::PropertyAccess() {
    add(ov::supported_properties, [this](const AnyMap& args) {
        auto supporeted_properties = get_supported();
        return flatten_supported(supporeted_properties, util::contains(args, "OV_FULL_PROPERTIES_ROUTS"));
    });
    add(METRIC_KEY(SUPPORTED_METRICS), [this] {
        std::vector<std::string> property_names;
        for (auto&& supporeted_property : get_supported()) {
            if (!supporeted_property.is_mutable()) {
                property_names.emplace_back(supporeted_property);
            }
        }
        return flatten_supported(property_names);
    });
    add(METRIC_KEY(SUPPORTED_CONFIG_KEYS), [this] {
        std::vector<std::string> property_names;
        for (auto&& supporeted_property : get_supported()) {
            if (supporeted_property.is_mutable()) {
                property_names.emplace_back(supporeted_property);
            }
        }
        auto flattedned_supported = flatten_supported(property_names);
        return flatten_supported(property_names);
    });
}

PropertyAccess& PropertyAccess::add(PropertyAccess sub_accesses) {
    sub_accesses.remove(ov::supported_properties)
        .remove(METRIC_KEY(SUPPORTED_METRICS))
        .remove(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
    for (auto&& access : sub_accesses.accesses) {
        accesses.emplace(std::move(access.first), std::move(access.second));
    }
    return *this;
}

PropertyAccess& PropertyAccess::add(const std::string& name, PropertyAccess sub_accesses_, const std::shared_ptr<void>& so) {
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

PropertyAccess& PropertyAccess::add(const NamedProperties& named_properties, PropertyAccess sub_accesses, const std::shared_ptr<void>& so) {
    return add(named_properties.name(), std::move(sub_accesses), so);
}

PropertyAccess& PropertyAccess::remove(const std::string& name) {
    accesses.erase(name);
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

static std::vector<std::string> check_found_pathes(const std::vector<std::vector<std::string>>& found_pathes,
                                                   const std::string& rout) {
    if (found_pathes.empty()) {
        OPENVINO_UNREACHABLE("Property ", rout, " was not found");
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
        OPENVINO_UNREACHABLE(strm.str());
    }
    return found_pathes.front();
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
        path.push_back(access.first);
        if (access.second->is_sub_access()) {
            auto sub_pathes = access.second->sub_access().get_all_pathes();
            for (auto&& sub_path : sub_pathes) {
                pathes.emplace_back(path);
                std::move(sub_path.begin(), sub_path.end(), std::back_inserter(pathes.back()));
            }
        } else {
            pathes.emplace_back(std::move(path));
        }
    }
    return pathes;
}

const void* PropertyAccess::find_access(const std::vector<std::string>& in_path) const {
    std::vector<std::string> path;
    const void* result = nullptr;
    if (!name.empty() && (in_path.front() == name)) {
        result = this;
        path = {in_path.begin() + 1, in_path.end()};
    } else {
        path = in_path;
    }
    if (!path.empty()) {
        auto it_access = accesses.find(path.front());
        if (it_access == accesses.end())
            return nullptr;
        if (it_access->second->is_sub_access()) {
            if ((path.begin() + 1) != path.end()) {
                result = it_access->second->sub_access().find_access({path.begin() + 1, path.end()});
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

AnyMap PropertyAccess::get(const PropertyMutability mutability) const {
    AnyMap result;
    for (auto&& access : accesses) {
        if (access.second->is_sub_access()) {
            auto any_map = access.second->sub_access().PropertyAccess::get(mutability);
            if (!any_map.empty()) {
                result.emplace(access.first, any_map);
            }
        } else if ((mutability == PropertyMutability::RW && access.second->is_mutable()) ||
                   (mutability == PropertyMutability::RO)) {
            auto any = access.second->get({});
            result.emplace(access.first, any);
        }
    }
    return result;
}

Any PropertyAccess::get(const std::string& name, const AnyMap& args) const {
    auto names = util::split(name, '.');
    auto access_ptr = find_access(check_found_pathes(find_property(names), name));
    if (access_ptr == this) {
        OPENVINO_ASSERT(names.size() > 1);
        return get(util::join(std::vector<std::string>{names.begin() + 1, names.end()}, "."), args);
    } else {
        OPENVINO_ASSERT(access_ptr != nullptr);
        auto access = static_cast<const Access*>(access_ptr);
        return access->get(args);
    }
}

PropertyAccess& PropertyAccess::set(const std::string& name, const Any& property) {
    auto names = util::split(name, '.');
    auto access_ptr = find_access(check_found_pathes(find_property(names), name));
    if (access_ptr == this) {
        return set(util::join(std::vector<std::string>{names.begin() + 1, names.end()}, "."), property);
    } else {
        OPENVINO_ASSERT(access_ptr != nullptr);
        auto access = static_cast<Access*>(access_ptr);
        if (access->is_sub_access() && property.is<AnyMap>()) {
            access->sub_access().PropertyAccess::set(property.as<AnyMap>());
        } else if ((!access->is_sub_access() && property.is<AnyMap>()) ||
                   (access->is_sub_access() && !property.is<AnyMap>())) {
            OPENVINO_UNREACHABLE("Could not merge unsupported types");
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

PropertyAccess& PropertyAccess::set(const AnyMap& properties) {
    for (auto&& value : properties) {
        set(value.first, value.second);
    }
    return *this;
}

static AnyMap flatten(const AnyMap& properties) {
    AnyMap result;
    for (auto&& property : properties) {
        if (property.second.is<AnyMap>()) {
            for (auto&& sub_property : flatten(property.second.as<AnyMap>())) {
                result.emplace(property.first + "." + sub_property.first, sub_property.second);
            }
        } else {
            result.emplace(property);
        }
    }
    return result;
}

PropertyAccess& PropertyAccess::set(const std::map<std::string, std::string>& properties) {
    for (auto&& property : properties) {
        auto names = util::split(property.first, '.');
        auto access_ptr = find_access(check_found_pathes(find_property(names), property.first));
        auto set_all = [&](PropertyAccess& property_access) {
            auto any_map = property_access.get();
            std::stringstream strm{property.second};
            util::Read<AnyMap>{}(strm, any_map);
            property_access.set(any_map);
        };
        if (access_ptr == this) {
            set_all(*this);
        } else {
            OPENVINO_ASSERT(access_ptr != nullptr);
            auto access = static_cast<Access*>(access_ptr);
            if (access->is_sub_access()) {
                set_all(access->sub_access());
            } else {
                OPENVINO_ASSERT(access->is_mutable(), "Could not merge read only property: ", property.first);
                auto any = access->get({});
                std::stringstream strm{property.second};
                any.read(strm);
                access->precondition(any);
                access->set(any);
            }
        }
    }
    return *this;
}

AnyMap PropertyAccess::merge(const AnyMap& properties, const PropertyMutability mutability) const {
    AnyMap result = get(mutability);
    for (auto&& property : properties) {
        auto names = util::split(property.first, '.');
        auto found_properties = find_property(names);
        if (found_properties.size() == 1) {
            auto access_ptr = find_access(found_properties.front());
            if ((access_ptr == this) && property.second.is<AnyMap>()) {
                for (auto&& v : merge(property.second.as<AnyMap>(), mutability)) {
                    result.emplace(v.first, util::to_string(v.second));
                }
            } else if ((access_ptr == this) && !property.second.is<AnyMap>()) {
                OPENVINO_UNREACHABLE("Could not merge unsupported types");
            } else {
                OPENVINO_ASSERT(access_ptr != nullptr);
                auto access = static_cast<const Access*>(access_ptr);
                if (access->is_sub_access() && property.second.is<AnyMap>()) {
                    auto any_map = access->sub_access().merge(property.second.as<AnyMap>(), mutability);
                    if (!any_map.empty()) {
                        result[property.first] = any_map;
                    }
                } else if ((!access->is_sub_access() && property.second.is<AnyMap>()) ||
                           (access->is_sub_access() && !property.second.is<AnyMap>())) {
                    OPENVINO_UNREACHABLE("Could not merge unsupported types");
                } else {
                    OPENVINO_ASSERT(access->is_mutable(), "Could not merge not writeable property: ", property.first);
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
                                                         const PropertyMutability mutability) const {
    std::map<std::string, std::string> result;
    for (auto&& property : properties) {
        auto found_properties = find_property(util::split(property.first, '.'));
        if (found_properties.size() == 1) {
            auto access_ptr = find_access(found_properties.front());
            if (access_ptr == this) {
                auto any_map = get(mutability);
                std::stringstream strm{property.second};
                util::Read<AnyMap>{}(strm, any_map);
                for (auto&& v : merge(any_map, mutability)) {
                    result.emplace(v.first, util::to_string(v.second));
                }
            } else {
                auto access = static_cast<const Access*>(access_ptr);
                if (access->is_sub_access()) {
                    auto any_map = access->sub_access().PropertyAccess::get(mutability);
                    std::stringstream strm{property.second};
                    util::Read<AnyMap>{}(strm, any_map);
                    any_map = access->sub_access().merge(any_map, mutability);
                    if (!any_map.empty()) {
                        result.emplace(property.first, util::to_string(any_map));
                    }
                } else {
                    auto any = access->get({});
                    OPENVINO_ASSERT(access->is_mutable(), "Could not merge read only property: ", property.first);
                    std::stringstream strm{property.second};
                    any.read(strm);
                    access->precondition(any);
                    result.emplace(property.first, any.as<std::string>());
                }
            }
        } else {
            result.emplace(property.first, property.second);
        }
    }
    for (auto value : get(mutability)) {
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