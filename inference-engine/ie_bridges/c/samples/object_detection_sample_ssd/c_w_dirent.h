// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if defined(_WIN32)

    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN_UNDEF
    #endif

    #ifndef NOMINMAX
        #define NOMINMAX
        #define NOMINMAX_UNDEF
    #endif

    #if defined(_M_IX86) && !defined(_X86_) && !defined(_AMD64_)
        #define _X86_
    #endif

    #if defined(_M_X64) && !defined(_X86_) && !defined(_AMD64_)
        #define _AMD64_
    #endif

    #if defined(_M_ARM) && !defined(_ARM_) && !defined(_ARM64_)
        #define _ARM_
    #endif

    #if defined(_M_ARM64) && !defined(_ARM_) && !defined(_ARM64_)
        #define _ARM64_
    #endif

    // clang-format off
    #include <string.h>
    #include <windef.h>
    #include <fileapi.h>
    #include <Winbase.h>
    #include <sys/stat.h>
    // clang-format on

    // Copied from linux libc sys/stat.h:
    #define S_ISREG(m) (((m)&S_IFMT) == S_IFREG)
    #define S_ISDIR(m) (((m)&S_IFMT) == S_IFDIR)

/// @brief structure to store directory names
typedef struct dirent {
    char* d_name;
} dirent;

/**
 * @brief Add directory to directory names struct
 * @param int argc - count of args
 * @param char *argv[] - array values of args
 * @param char *opts - array of options
 * @return pointer to directory names struct
 */
static dirent* createDirent(const wchar_t* wsFilePath) {
    dirent* d = (dirent*)malloc(sizeof(dirent));
    size_t i;
    size_t slen = wcslen(wsFilePath);
    d->d_name = (char*)(malloc(slen + 1));
    wcstombs_s(&i, d->d_name, slen + 1, wsFilePath, slen);
    return d;
}

/**
 * @brief Free directory names struct
 * @param point to directory names structure
 * @return none
 */
static void freeDirent(dirent** d) {
    free((*d)->d_name);
    (*d)->d_name = NULL;
    free(*d);
    *d = NULL;
}

/// @brief structure to store directory data (files meta)
typedef struct DIR {
    WIN32_FIND_DATAA FindFileData;
    HANDLE hFind;
    dirent* next;
} DIR;

/**
 * @brief Compare two string, second string is the end of the first
 * @param string to compare
 * @param end string to find
 * @return status 1(success) or 0(fail)
 */
static int endsWith(const char* src, const char* with) {
    int wl = (int)(strlen(with));
    int so = (int)(strlen(with)) - wl;
    if (so < 0)
        return 0;
    if (strncmp(with, &(src[so]), wl) == 0)
        return 1;
    else
        return 0;
}

/**
 * @brief Check file handler is valid
 * @param struct of directory data
 * @return status 1(success) or 0(fail)
 */
static int isValid(DIR* dp) {
    if (dp->hFind != INVALID_HANDLE_VALUE && dp->FindFileData.dwReserved0) {
        return 1;
    } else {
        return 0;
    }
}

/**
 * @brief Create directory data struct element
 * @param string directory path
 * @return pointer to directory data struct element
 */
static DIR* opendir(const char* dirPath) {
    DIR* dp = (DIR*)malloc(sizeof(DIR));
    dp->next = NULL;
    char* ws = (char*)(malloc(strlen(dirPath) + 1));
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

/**
 * @brief Walk throw directory data struct
 * @param pointer to directory data struct
 * @return pointer to directory data struct next element
 */
static struct dirent* readdir(DIR* dp) {
    if (dp->next != NULL)
        freeDirent(&(dp->next));

    if (!dp->FindFileData.dwReserved0)
        return NULL;

    wchar_t wbuf[4096];

    size_t outSize;
    mbstowcs_s(&outSize, wbuf, 4094, dp->FindFileData.cFileName, 4094);
    dp->next = createDirent(wbuf);
    dp->FindFileData.dwReserved0 = FindNextFileA(dp->hFind, &(dp->FindFileData));
    return dp->next;
}

/**
 * @brief Remove directory data struct
 * @param pointer to struct directory data
 * @return none
 */
static void closedir(DIR* dp) {
    if (dp->next) {
        freeDirent(&(dp->next));
    }
    free(dp);
}

    #ifdef WIN32_LEAN_AND_MEAN_UNDEF
        #undef WIN32_LEAN_AND_MEAN
        #undef WIN32_LEAN_AND_MEAN_UNDEF
    #endif

    #ifdef NOMINMAX_UNDEF
        #undef NOMINMAX_UNDEF
        #undef NOMINMAX
    #endif

#else

    #include <dirent.h>
    #include <sys/types.h>

#endif