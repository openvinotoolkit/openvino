// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <utility>
#include <set>
#include <type_traits>

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

    inline AttributesMap() = default;

    inline AttributesMap(const AttributesMap&) = default;
    inline AttributesMap& operator=(const AttributesMap&) = default;

    inline AttributesMap(AttributesMap&&) = default;
    inline AttributesMap& operator=(AttributesMap&&) = default;

    inline ~AttributesMap() = default;

    inline bool empty() const {
        return _tbl.empty();
    }

    inline bool has(const std::string& name) const {
        return _tbl.count(name) != 0;
    }

    inline void erase(const std::string& name) {
        _tbl.erase(name);
    }

    template <typename T>
    inline void set(const std::string& name, const T& val) {
        _tbl[name].set(val);
    }
    template <
        typename T,
        typename _Check = typename std::enable_if<!std::is_reference<T>::value, void>::type>
    inline void set(const std::string& name, T&& val) {
        _tbl[name].set(std::forward<typename std::remove_reference<T>::type>(val));
    }

    template <typename T>
    inline const T& get(const std::string& name) const {
        auto it = _tbl.find(name);
        IE_ASSERT(it != _tbl.end());
        return it->second.get<T>();
    }
    template <typename T>
    inline T& get(const std::string& name) {
        auto it = _tbl.find(name);
        IE_ASSERT(it != _tbl.end());
        return it->second.get<T>();
    }

    template <typename T>
    inline const T& getOrDefault(const std::string& name, const T& def) const {
        auto it = _tbl.find(name);
        if (it != _tbl.end()) {
            return it->second.get<T>();
        }
        return def;
    }

    template <typename T>
    inline T& getOrSet(const std::string& name, const T& def) {
        auto it = _tbl.find(name);
        if (it != _tbl.end()) {
            return it->second.get<T>();
        }
        auto res = _tbl.insert({name, Any(def)});
        assert(res.second);
        return res.first->second.get<T>();
    }
    template <
        typename T,
        typename _Check = typename std::enable_if<!std::is_reference<T>::value, void>::type>
    inline T& getOrSet(const std::string& name, T&& def) {
        auto it = _tbl.find(name);
        if (it != _tbl.end()) {
            return it->second.get<typename std::decay<T>::type>();
        }
        auto res = _tbl.insert({name, Any(std::forward<typename std::remove_reference<T>::type>(def))});
        assert(res.second);
        return res.first->second.get<typename std::decay<T>::type>();
    }

    inline iterator begin() { return _tbl.begin(); }
    inline iterator end() { return _tbl.end(); }

    inline const_iterator begin() const { return _tbl.begin(); }
    inline const_iterator end() const { return _tbl.end(); }

    inline const_iterator cbegin() const { return _tbl.cbegin(); }
    inline const_iterator cend() const { return _tbl.cend(); }

    inline void copyFrom(const AttributesMap& other) {
        for (const auto& p : other._tbl) {
            _tbl[p.first] = p.second;
        }
    }

    inline void printImpl(std::ostream& os) const {
        printTo(os, _tbl);
    }

    inline void printImpl(DotLabel& lbl) const {
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
    inline const AttributesMap& attrs() const { return _attrs; }
    inline AttributesMap& attrs() { return _attrs; }

protected:
    inline EnableCustomAttributes() = default;
    inline EnableCustomAttributes(const EnableCustomAttributes&) = default;
    inline EnableCustomAttributes(EnableCustomAttributes&&) = default;
    inline ~EnableCustomAttributes() = default;
    inline EnableCustomAttributes& operator=(const EnableCustomAttributes&) = default;
    inline EnableCustomAttributes& operator=(EnableCustomAttributes&&) = default;

private:
    AttributesMap _attrs;
};

}  // namespace vpu
