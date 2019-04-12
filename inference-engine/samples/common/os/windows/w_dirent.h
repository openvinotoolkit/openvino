// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if defined(_WIN32)

#ifndef NOMINMAX
# define NOMINMAX
#endif

#include <winsock2.h>
#include <windows.h>
#include <stdlib.h>

#else

#include <unistd.h>
#include <cstdlib>
#include <string.h>

#endif

#include <string>

#include <sys/stat.h>

#if defined(WIN32)
    // Copied from linux libc sys/stat.h:
    #define S_ISREG(m) (((m) & S_IFMT) == S_IFREG)
    #define S_ISDIR(m) (((m) & S_IFMT) == S_IFDIR)
#endif

struct dirent {
    char *d_name;

    explicit dirent(const wchar_t *wsFilePath) {
        size_t i;
        auto slen = wcslen(wsFilePath);
        d_name = static_cast<char*>(malloc(slen + 1));
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


static DIR *opendir(const char* dirPath) {
    auto dp = new DIR(dirPath);
    if (!dp->isValid()) {
        delete dp;
        return nullptr;
    }
    return dp;
}

static struct dirent *readdir(DIR *dp) {
    return dp->nextEnt();
}

static void closedir(DIR *dp) {
    delete dp;
}
