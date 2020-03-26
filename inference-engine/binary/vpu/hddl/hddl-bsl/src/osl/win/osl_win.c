// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include "osl.h"
#include <Shlwapi.h>
#pragma comment(lib, "Shlwapi.lib")

errno_t bsl_strncpy(char* _Destination, size_t _SizeInBytes, char const* _Source, size_t _MaxCount) {
  return strncpy_s(_Destination, _SizeInBytes, _Source, _MaxCount);
}

errno_t bsl_fopen(FILE** _Stream, char const* _FileName, char const* _Mode) {
  return fopen_s(_Stream, _FileName, _Mode);
}

bool check_path_is_dir(const char* path) {
  return PathIsDirectoryA(path);
}