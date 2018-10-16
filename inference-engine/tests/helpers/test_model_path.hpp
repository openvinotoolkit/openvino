// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fstream>
#ifndef _WIN32 
# include <libgen.h>
# include <dirent.h>
#else
# include <os/windows/w_dirent.h>
#endif

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

    operator std::string() const {
        std::string absPathPublic = get_models_path() + kPathSeparator + "src" + kPathSeparator + "models" + _rel_path.str();
        if (exist(absPathPublic)) {
            return absPathPublic;
        }

        // checking dirname
        std::string publicDir = getDirname(absPathPublic);

        struct stat sb;
        if (stat(publicDir.c_str(), &sb) != 0) {
            publicDir = "";
        } else if (!S_ISDIR(sb.st_mode)) {
            publicDir = "";
        }

        std::string absPathPrivate = get_models_path() + kPathSeparator + "src" + kPathSeparator + "models_private" + _rel_path.str();
        if (exist(absPathPrivate)) {
            return absPathPrivate;
        }

        if (!publicDir.empty()) {
            return absPathPublic;
        }

        std::string privateDir = getDirname(absPathPrivate);

        if (stat(privateDir.c_str(), &sb) != 0 || !S_ISDIR(sb.st_mode)) {
            ::testing::AssertionFailure() << "path to model invalid, no private or public model found for: " + _rel_path.str();
        }

        return absPathPrivate;
    }

 private :
#ifndef _WIN32
    static std::string getDirname (std::string filePath) {
        std::vector<char> input(filePath.begin(), filePath.end());
        return dirname(&*input.begin());
    }
#else
    static std::string getDirname (std::string filePath) {
        char dirname[_MAX_DIR];
        _splitpath(filePath.c_str(), nullptr, dirname, nullptr, nullptr);
        return dirname;
    }
#endif
#ifdef MODELS_PATH
    static const char* getModelPathNonFatal() noexcept {
        const char* models_path = std::getenv("MODELS_PATH");

        if (models_path == nullptr && MODELS_PATH == nullptr) {
            return nullptr;
        }

        if (models_path == nullptr) {
            return MODELS_PATH;
        }

        return models_path;
    }
#else
    static const char *getModelPathNonFatal() {
        return nullptr;
    }
#endif

    static std::string get_models_path() {
        const char* models_path = getModelPathNonFatal();

        if (nullptr == models_path) {
            ::testing::AssertionFailure() << "MODELS_PATH not defined";
        }

        return std::string(models_path);
    }


 protected:
    static bool exist(const std::string& name) {
        std::ifstream file(name);
        if(!file)            // If the file was not found, then file is 0, i.e. !file=1 or true.
            return false;    // The file was not found.
        else                 // If the file was found, then file is non-0.
            return true;     // The file was found.
    }
};

inline std::ostream & operator << (std::ostream &os, const ModelsPath & path) {
    os << path.str();
    return os;
}
