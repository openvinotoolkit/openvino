// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if defined(_WIN32)

#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN_UNDEF
#    endif

#    ifndef NOMINMAX
#        define NOMINMAX
#        define NOMINMAX_UNDEF
#    endif

#    if defined(_M_IX86) && !defined(_X86_) && !defined(_AMD64_)
#        define _X86_
#    endif

#    if defined(_M_X64) && !defined(_X86_) && !defined(_AMD64_)
#        define _AMD64_
#    endif

#    if defined(_M_ARM) && !defined(_ARM_) && !defined(_ARM64_)
#        define _ARM_
#    endif

#    if defined(_M_ARM64) && !defined(_ARM_) && !defined(_ARM64_)
#        define _ARM64_
#    endif

// clang-format off
    #include <string.h>
    #include <windef.h>
    #include <fileapi.h>
    #include <winbase.h>
    #include <sys/stat.h>
// clang-format on

// Copied from linux libc sys/stat.h:
#    define S_ISREG(m) (((m)&S_IFMT) == S_IFREG)
#    define S_ISDIR(m) (((m)&S_IFMT) == S_IFDIR)

/// @brief structure to store directory names
struct dirent {
    char* d_name;

    explicit dirent(const wchar_t* wsFilePath) {
        size_t i;
        auto slen = wcslen(wsFilePath);
        d_name = static_cast<char*>(malloc(slen + 1));
        wcstombs_s(&i, d_name, slen + 1, wsFilePath, slen);
    }
    ~dirent() {
        free(d_name);
    }
};

/// @brief class to store directory data (files meta)
class DIR {
    WIN32_FIND_DATAA FindFileData;
    HANDLE hFind;
    dirent* next;

    static inline bool endsWith(const std::string& src, const char* with) {
        int wl = static_cast<int>(strlen(with));
        int so = static_cast<int>(src.length()) - wl;
        if (so < 0)
            return false;
        return 0 == strncmp(with, &src[so], wl);
    }

public:
    DIR(const DIR& other) = delete;
    DIR(DIR&& other) = delete;
    DIR& operator=(const DIR& other) = delete;
    DIR& operator=(DIR&& other) = delete;

    explicit DIR(const char* dirPath) : next(nullptr) {
        std::string ws = dirPath;
        if (endsWith(ws, "\\"))
            ws += "*";
        else
            ws += "\\*";
        hFind = FindFirstFileA(ws.c_str(), &FindFileData);
        FindFileData.dwReserved0 = hFind != INVALID_HANDLE_VALUE;
    }

    ~DIR() {
        if (!next)
            delete next;
        next = nullptr;
        FindClose(hFind);
    }

    /**
     * @brief Check file handler is valid
     * @return status True(success) or False(fail)
     */
    bool isValid() const {
        return (hFind != INVALID_HANDLE_VALUE && FindFileData.dwReserved0);
    }

    /**
     * @brief Add directory to directory names struct
     * @return pointer to directory names struct
     */
    dirent* nextEnt() {
        if (next != nullptr)
            delete next;
        next = nullptr;

        if (!FindFileData.dwReserved0)
            return nullptr;

        wchar_t wbuf[4096];

        size_t outSize;
        mbstowcs_s(&outSize, wbuf, 4094, FindFileData.cFileName, 4094);
        next = new dirent(wbuf);
        FindFileData.dwReserved0 = FindNextFileA(hFind, &FindFileData);
        return next;
    }
};

/**
 * @brief Create directory data struct element
 * @param string directory path
 * @return pointer to directory data struct element
 */
inline DIR* opendir(const char* dirPath) {
    auto dp = new DIR(dirPath);
    if (!dp->isValid()) {
        delete dp;
        return nullptr;
    }
    return dp;
}

/**
 * @brief Walk throw directory data struct
 * @param pointer to directory data struct
 * @return pointer to directory data struct next element
 */
inline struct dirent* readdir(DIR* dp) {
    return dp->nextEnt();
}

/**
 * @brief Remove directory data struct
 * @param pointer to struct directory data
 * @return void
 */
inline void closedir(DIR* dp) {
    delete dp;
}

#    ifdef WIN32_LEAN_AND_MEAN_UNDEF
#        undef WIN32_LEAN_AND_MEAN
#        undef WIN32_LEAN_AND_MEAN_UNDEF
#    endif

#    ifdef NOMINMAX_UNDEF
#        undef NOMINMAX_UNDEF
#        undef NOMINMAX
#    endif

#else

#    include <dirent.h>
#    include <sys/types.h>

#endif
