// Copyright (C) 020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "XLinkFileUtils.h"
#include "XLinkStringUtils.h"

#include <string.h>

#ifdef _WIN32
#include <Windows.h>
#else
#define __USE_GNU
#include <dlfcn.h>      // For dladdr
#include <stdlib.h>
#endif

#ifdef _WIN32
static int convertCharToWide(const char* const str, wchar_t** dest)
{
    const int srcLen = MultiByteToWideChar( CP_UTF8 , 0 , str , -1, NULL , 0 ) + 1;
    *dest = NULL;
    wchar_t* tempDest = (wchar_t *)malloc(srcLen * sizeof(wchar_t));
    if (!tempDest)
        return 0;

    if (!MultiByteToWideChar(CP_UTF8, 0, str, -1, tempDest, srcLen)) {
        free(tempDest);
        return 0;
    }
    *dest = tempDest;
    return 1;
}

static int convertWideToChar(const wchar_t* const str, char* dest, int destSize)
{
    wchar_t buffer[256];
    int length = GetShortPathNameW(str, NULL, 0);
    length = GetShortPathNameW(str, buffer, length);
    auto res = WideCharToMultiByte(CP_UTF8, 0, buffer, length, dest, destSize, NULL, NULL);
    if (!res)
        return 0;
    return 1;
}
#endif

#ifdef __cplusplus
extern "C"
{
#endif

FILE* utf8_fopen(const char* filename, const char* mode)
{
#ifdef _WIN32
    wchar_t *wfilename = NULL, *wmode = NULL;
    if (!convertCharToWide(filename, &wfilename))
        return NULL;
    if (!convertCharToWide(mode, &wmode)) {
        free(wfilename);
        return NULL;
    }
    FILE* res = _wfopen(wfilename, wmode);
    free(wfilename);
    free(wmode);
    return res;
#else
    return fopen(filename, mode);
#endif
}

int utf8_open(const char* filename, int flag, int mode)
{
#ifdef _WIN32
    wchar_t* wfilename = NULL;
    if (!convertCharToWide(filename, &wfilename))
        return -1;

    int res = _wopen(wfilename, flag, mode);
    free(wfilename);
    return res;
#else
    return open(filename, flag, mode);
#endif
}

int utf8_access(const char* filename, int mode)
{
#ifdef _WIN32
    wchar_t* wfilename = NULL;
    if (!convertCharToWide(filename, &wfilename))
        return -1;

    int res = _waccess(wfilename, mode);
    free(wfilename);
    return res;
#else
    return access(filename, mode);
#endif
}

int utf8_getenv_s(size_t bufferSize, char* buffer, const char* varName)
{
#ifdef _WIN32
    wchar_t* wvarName = NULL;
    if (!convertCharToWide(varName, &wvarName))
        return 0;

    wchar_t* wbuffer = (wchar_t *)malloc(bufferSize * sizeof(wchar_t));
    if (!wbuffer) {
        free(wvarName);
        return 0;
    }

    auto ret = GetEnvironmentVariableW(wvarName, wbuffer, bufferSize);
    free(wvarName);

    int res = 0;
    if (ret) {
        // ConvertWideToChar return false due to GetShortPathNameW, which works only for already existant files...
        if (!convertWideToChar(wbuffer, buffer, (int)bufferSize)) {
            // Print warning that ENV variable is FOUND but it is not possible to convert it to char*
            printf("[WARNING]EVN Variable set, but file doesnt exist..."); // testing
            res = 1;
        }
    }
    free(wbuffer);
    return res == 0;
#else
    char* value = getenv(varName);
    if (!value)
        return 0;

    const size_t len = strlen(value);
    if (mv_strncpy(buffer, bufferSize, value, len + 1) != 0)
        return 0;

    return 1;
#endif
}

int utf8_shared_lib_path(size_t bufferSize, char* buffer)
{
#ifdef _WIN32
    HMODULE hm = NULL;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            L"utf8_getenv_s", &hm)) {
        return 0;
    }
    wchar_t* wbuffer = (wchar_t *)malloc(bufferSize * sizeof(wchar_t));
    if (!wbuffer) {
        return 0;
    }
    if (!GetModuleFileNameW(hm, wbuffer, (DWORD)bufferSize - 1)) {
        free(wbuffer);
        return 0;
    }
    int res = convertWideToChar(wbuffer, buffer, (int)bufferSize);
    free(wbuffer);

    return res;
#else
    Dl_info info;
    dladdr((void*)&utf8_getenv_s, &info);
    int rc = mv_strncpy(buffer, bufferSize, info.dli_fname, bufferSize - 1);
    if (rc != 0) {
        return 0;
    }
    return 1;
#endif
}

#ifdef __cplusplus
} // extern "C"
#endif
