// Copyright (C) 2018-2020 Intel Corporation
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

#if defined(_M_ARM) && !defined(_ARM_) && !defined(_ARM64_)
# define _ARM_
#endif

#if defined(_M_ARM64) && !defined(_ARM_) && !defined(_ARM64_)
# define _ARM64_
#endif

#include <string.h>
#include <windef.h>
#include <fileapi.h>
#include <Winbase.h>
#include <sys/stat.h>

// Copied from linux libc sys/stat.h:
#define S_ISREG(m) (((m) & S_IFMT) == S_IFREG)
#define S_ISDIR(m) (((m) & S_IFMT) == S_IFDIR)

typedef struct dirent {
    char *d_name;
}dirent;

static dirent *createDirent(const wchar_t *wsFilePath) {
    dirent *d = (dirent *)malloc(sizeof(dirent));
    size_t i;
    size_t slen = wcslen(wsFilePath);
    d->d_name = (char *)(malloc(slen + 1));
    wcstombs_s(&i, d->d_name, slen + 1, wsFilePath, slen);
    return d;
}

static void freeDirent(dirent **d) {
    free((*d)->d_name);
    (*d)->d_name = NULL;
    free(*d);
    *d = NULL;
}

typedef struct DIR {
    WIN32_FIND_DATAA FindFileData;
    HANDLE hFind;
    dirent *next;
}DIR;

static int endsWith(const char *src, const char *with) {
    int wl = (int)(strlen(with));
    int so = (int)(strlen(with)) - wl;
    if (so < 0) return 0;
    if (strncmp(with, &(src[so]), wl) == 0)
        return 1;
    else
        return 0;
}
static int isValid(DIR* dp) {
    if (dp->hFind != INVALID_HANDLE_VALUE && dp->FindFileData.dwReserved0) {
        return 1;
    } else {
        return 0;
    }
}
static DIR *opendir(const char *dirPath) {
    DIR *dp = (DIR *)malloc(sizeof(DIR));
    dp->next = NULL;
    char *ws = (char *)(malloc(strlen(dirPath) + 1));
    strcpy(ws, dirPath);
    if (endsWith(ws, "\\"))
        strcat(ws, "*");
    else
        strcat(ws, "\\*");
    dp->hFind = FindFirstFileA(ws, &dp->FindFileData);
    dp->FindFileData.dwReserved0 = dp->hFind != INVALID_HANDLE_VALUE;
    free(ws);
    if (isValid(dp)) {
        free(dp);
        return NULL;
    }
    return dp;
}

static struct dirent *readdir(DIR *dp) {
    if (dp->next != NULL) freeDirent(&(dp->next));

    if (!dp->FindFileData.dwReserved0) return NULL;

    wchar_t wbuf[4096];

    size_t outSize;
    mbstowcs_s(&outSize, wbuf, 4094, dp->FindFileData.cFileName, 4094);
    dp->next = createDirent(wbuf);
    dp->FindFileData.dwReserved0 = FindNextFileA(dp->hFind, &(dp->FindFileData));
    return dp->next;
}

static void closedir(DIR *dp){
    if (dp->next) {
        freeDirent(&(dp->next));
    }
    free(dp);
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
