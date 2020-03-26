// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include "osl.h"
#include <dirent.h>

errno_t bsl_strncpy(char* _Destination, size_t _SizeInBytes, char const* _Source, size_t _MaxCount) {
  if (_Destination == NULL) {
    return 1;
  }

  if (_Source == NULL) {
    return 1;
  }

  size_t source_len = strnlen(_Source, _MaxCount);

  bool source_not_null_terminated = source_len >= _MaxCount;
  if (source_not_null_terminated) {
    return 1;
  }

  bool destination_too_small = source_len >= _SizeInBytes;
  if (destination_too_small) {
    return 1;
  }

  // At this point, source is ensured to be null terminated and can be fit into destination
  strncpy(_Destination, _Source, _SizeInBytes);
  _Destination[_SizeInBytes - 1] = '\0';  // Useless. Just to prevent the kw warning
  return 0;
}

errno_t bsl_fopen(FILE** _Stream, char const* _FileName, char const* _Mode) {
  *_Stream = fopen(_FileName, _Mode);
  if (*_Stream == NULL) {
    return 1;
  }
  return 0;
}

bool check_path_is_dir(const char *path) {
  DIR* p_dir = opendir(path);
  if (p_dir != NULL) {
    closedir(p_dir);
    return true;
  }
  return false;
}
