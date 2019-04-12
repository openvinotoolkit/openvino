// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <string>
#if defined(WIN32)
#include "w_unistd.h"
#include "debug.h"
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
    WIN32_FIND_DATA FindFileData;
    HANDLE hFind;
    dirent *next;

public:
    DIR(const DIR &other) = delete;
    DIR(DIR &&other) = delete;
    DIR& operator=(const DIR &other) = delete;
    DIR& operator=(DIR &&other) = delete;

    explicit DIR(const char *dirPath) : next(nullptr) {
        // wchar_t  ws[1024];
        // swprintf(ws, 1024, L"%hs\\*", dirPath);
        std::string ws = dirPath;
        if (InferenceEngine::details::endsWith(ws, "\\"))
            ws += "*";
        else
            ws += "\\*";
        hFind = FindFirstFile(ws.c_str(), &FindFileData);
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
        FindFileData.dwReserved0 = FindNextFile(hFind, &FindFileData);
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
#else

#include <sys/types.h>
#include <dirent.h>

#endif

