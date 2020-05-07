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
    const int srcLen = (int)wcslen(str) + 1;
    if (!WideCharToMultiByte(CP_UTF8, 0, str, srcLen, dest, destSize, NULL, NULL))
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

    size_t returnValue = 0;
    int res = _wgetenv_s(&returnValue, wbuffer, bufferSize, wvarName);
    free(wvarName);
    if (returnValue == 0) // returnValue == 0 means "variable not found"
        res = 1;

    if (res == 0) {
        if (!convertWideToChar(wbuffer, buffer, (int)bufferSize))
            res = 1;
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
