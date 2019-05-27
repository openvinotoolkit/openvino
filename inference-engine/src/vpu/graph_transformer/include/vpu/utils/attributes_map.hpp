// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>

#include <vpu/utils/any.hpp>
#include <vpu/utils/io.hpp>
#include <vpu/utils/dot_io.hpp>

namespace vpu {

//
// AttributesMap
//

class AttributesMap final {
public:
    using BaseMap = std::map<std::string, Any>;

    using iterator = BaseMap::iterator;
    using const_iterator = BaseMap::const_iterator;

    AttributesMap() = default;
    AttributesMap(const AttributesMap&) = default;
    AttributesMap(AttributesMap&&) = default;
    ~AttributesMap() = default;
    AttributesMap& operator=(const AttributesMap&) = default;
    AttributesMap& operator=(AttributesMap&&) = default;

    bool empty() const { return _tbl.empty(); }

    bool has(const std::string& name) const { return _tbl.count(name) != 0; }

    template <typename T>
    void set(const std::string& name, const T& val) { _tbl[name] = Any(val); }

    void erase(const std::string& name) {  _tbl.erase(name); }

    template <typename T>
    const T& get(const std::string& name) const {
        auto it = _tbl.find(name);
        IE_ASSERT(it != _tbl.end());
        return it->second.cast<T>();
    }

    template <typename T>
    T& get(const std::string& name) {
        auto it = _tbl.find(name);
        IE_ASSERT(it != _tbl.end());
        return it->second.cast<T>();
    }

    template <typename T>
    const T& getOrDefault(const std::string& name, const T& def) const {
        auto it = _tbl.find(name);
        if (it != _tbl.end())
            return it->second.cast<T>();
        return def;
    }

    template <typename T>
    T& getOrSet(const std::string& name, const T& def) {
        auto it = _tbl.find(name);
        if (it != _tbl.end())
            return it->second.cast<T>();
        set(name, def);
        return get<T>(name);
    }

    iterator begin() { return _tbl.begin(); }
    iterator end() { return _tbl.end(); }

    const_iterator begin() const { return _tbl.begin(); }
    const_iterator end() const { return _tbl.end(); }

    const_iterator cbegin() const { return _tbl.cbegin(); }
    const_iterator cend() const { return _tbl.cend(); }

    void copyFrom(const AttributesMap& other) {
        for (const auto& p : other._tbl) {
            _tbl[p.first] = p.second;
        }
    }

    void printImpl(std::ostream& os) const {
        printTo(os, _tbl);
    }

    void printImpl(DotLabel& lbl) const {
        printTo(lbl, _tbl);
    }

private:
    BaseMap _tbl;
};

//
// EnableCustomAttributes
//

class EnableCustomAttributes {
public:
    const AttributesMap& attrs() const { return _attrs; }
    AttributesMap& attrs() { return _attrs; }

protected:
    EnableCustomAttributes() = default;
    EnableCustomAttributes(const EnableCustomAttributes&) = default;
    EnableCustomAttributes(EnableCustomAttributes&&) = default;
    ~EnableCustomAttributes() = default;
    EnableCustomAttributes& operator=(const EnableCustomAttributes&) = default;
    EnableCustomAttributes& operator=(EnableCustomAttributes&&) = default;

private:
    AttributesMap _attrs;
};

}  // namespace vpu
