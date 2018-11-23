// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <sstream>

static const char kPathSeparator =
#if defined _WIN32 || defined __CYGWIN__
    '\\';
#else
    '/';
#endif

class ModelsPath {
    std::stringstream _rel_path;
    mutable std::string _abs_path;

 public:

    ModelsPath() = default;

    ModelsPath(const ModelsPath & that) {
        _rel_path << that._rel_path.str();
    }

    template <class T>
    ModelsPath operator + (const T & relative_path) const {
        ModelsPath newPath(*this);
        newPath += relative_path;
        return newPath;
    }

    template <class T>
    ModelsPath & operator += (const T & relative_path) {
        _rel_path << relative_path;
        return *this;
    }

    template <class T>
    ModelsPath & operator << (const T & serializable) {
        _rel_path << serializable;
        return *this;
    }

    std::string str() const {
        return this->operator std::string();
    }

    const char * c_str() const {
        _abs_path = this->operator std::string ();
        return _abs_path.c_str();
    }

    operator std::string() const;
};

inline std::ostream & operator << (std::ostream &os, const ModelsPath & path) {
    os << path.str();
    return os;
}