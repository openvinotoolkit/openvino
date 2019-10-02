// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if defined(_WIN32)

#ifndef WIN32_LEAN_AND_MEAN
# define WIN32_LEAN_AND_MEAN
# define WIN32_LEAN_AND_MEAN_UNDEF
#endif

#ifndef NOMINMAX
# define NOMINMAX
# define NOMINMAX_UNDEF
#endif

#if defined(_M_IX86) && !defined(_X86_) && !defined(_AMD64_)
# define _X86_
#endif

#if defined(_M_X64) && !defined(_X86_) && !defined(_AMD64_)
# define _AMD64_
#endif

#include <string>
#include <windef.h>
#include <fileapi.h>
#include <Winbase.h>
#include <sys/stat.h>

// Copied from linux libc sys/stat.h:
#define S_ISREG(m) (((m) & S_IFMT) == S_IFREG)
#define S_ISDIR(m) (((m) & S_IFMT) == S_IFDIR)

struct dirent {
    char *d_name;

    explicit dirent(const wchar_t *wsFilePath) {
        size_t i;
        auto slen = wcslen(wsFilePath);
        d_name = static_cast<char *>(malloc(slen + 1));
        wcstombs_s(&i, d_name, slen + 1, wsFilePath, slen);
    }
    ~dirent() {
        free(d_name);
    }
};

class DIR {
    WIN32_FIND_DATAA FindFileData;
    HANDLE hFind;
    dirent *next;

    static inline bool endsWith(const std::string &src, const char *with) {
        int wl = static_cast<int>(strlen(with));
        int so = static_cast<int>(src.length()) - wl;
        if (so < 0) return false;
        return 0 == strncmp(with, &src[so], wl);
    }

public:
    DIR(const DIR &other) = delete;
    DIR(DIR &&other) = delete;
    DIR& operator=(const DIR &other) = delete;
    DIR& operator=(DIR &&other) = delete;

    explicit DIR(const char *dirPath) : next(nullptr) {
        std::string ws = dirPath;
        if (endsWith(ws, "\\"))
            ws += "*";
        else
            ws += "\\*";
        hFind = FindFirstFileA(ws.c_str(), &FindFileData);
        FindFileData.dwReserved0 = hFind != INVALID_HANDLE_VALUE;
    }

    ~DIR() {
        if (!next) delete next;
        next = nullptr;
        FindClose(hFind);
    }

    bool isValid() const {
        return (hFind != INVALID_HANDLE_VALUE && FindFileData.dwReserved0);
    }

    dirent* nextEnt() {
        if (next != nullptr) delete next;
        next = nullptr;

        if (!FindFileData.dwReserved0) return nullptr;

        wchar_t wbuf[4096];

        size_t outSize;
        mbstowcs_s(&outSize, wbuf, 4094, FindFileData.cFileName, 4094);
        next = new dirent(wbuf);
        FindFileData.dwReserved0 = FindNextFileA(hFind, &FindFileData);
        return next;
    }
};


static DIR* opendir(const char *dirPath) {
    auto dp = new DIR(dirPath);
    if (!dp->isValid()) {
        delete dp;
        return nullptr;
    }
    return dp;
}

static struct dirent* readdir(DIR *dp) {
    return dp->nextEnt();
}

static void closedir(DIR *dp) {
    delete dp;
}

#ifdef WIN32_LEAN_AND_MEAN_UNDEF
# undef WIN32_LEAN_AND_MEAN
# undef WIN32_LEAN_AND_MEAN_UNDEF
#endif

#ifdef NOMINMAX_UNDEF
# undef NOMINMAX_UNDEF
# undef NOMINMAX
#endif

#else

#include <sys/types.h>
#include <dirent.h>

#endif
