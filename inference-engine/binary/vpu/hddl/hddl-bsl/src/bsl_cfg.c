// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include "hddl-bsl.h"
#include "hddl_bsl_priv.h"
#include "bsl_cfg.h"

BSL_STATUS cfg_get_path(char* path, size_t buffer_size) {
  BSL_STATUS status = BSL_SUCCESS;
#ifdef WIN32
  char* envValue = NULL;
  size_t sz = 0;
  errno_t dup_status = _dupenv_s(&envValue, &sz, "HDDL_INSTALL_DIR");
  if (dup_status != 0) {
    status = BSL_ERROR_HDDL_INSTALL_DIR_NOT_PROVIDED;
    goto cfg_get_path_exit;
  }
#else
  char* envValue = getenv("HDDL_INSTALL_DIR");
#endif

  if (!envValue) {
    *path = '\0';
    status = BSL_ERROR_HDDL_INSTALL_DIR_NOT_PROVIDED;
    goto cfg_get_path_exit;
  }

  if (!check_path_is_dir(envValue)) {
    status = BSL_ERROR_HDDL_INSTALL_DIR_NOT_DIR;
    snprintf(path, buffer_size, "%s", envValue);
    goto cfg_get_path_exit;
  }

  if (strlen(envValue) + strlen("/config/bsl.json") > buffer_size - 1) {
    status = BSL_ERROR_HDDL_INSTALL_DIR_TOO_LONG;
    goto cfg_get_path_exit;
  }

  snprintf(path, buffer_size, "%s/config/bsl.json", envValue);
  FILE* fp = NULL;
  status = bsl_fopen(&fp, path, "r");
  if (status) {
    status = BSL_ERROR_CFG_OPEN_FAILED;
    goto cfg_get_path_exit;
  }
  fclose(fp);

cfg_get_path_exit:
#ifdef WIN32
  free(envValue);
#endif
  return status;
}

BSL_STATUS cfg_load_file(const char* file_path, char** buffer) {
  FILE* fp;
  errno_t bsl_open_status = bsl_fopen(&fp, file_path, "rb");
  if (bsl_open_status != 0) {
    return BSL_ERROR_CFG_OPEN_FAILED;
  }

  fseek(fp, 0, SEEK_END);
  long raw_length = ftell(fp);
  if (raw_length < 0) {
    fclose(fp);
    return BSL_ERROR_CFG_OPEN_FAILED;
  }

  rewind(fp);

  size_t length = (size_t)raw_length;
  *buffer = malloc(length + 1);
  if (*buffer == NULL) {
    fclose(fp);
    return BSL_ERROR_MEMORY_ALLOC_FAILED;
  }

  size_t bytes_read = fread(*buffer, 1, length, fp);
  if (bytes_read != length) {
    fclose(fp);
    free(*buffer);
    return BSL_ERROR_INVALID_CFG_FILE;
  }

  (*buffer)[length] = '\0';
  fclose(fp);
  return BSL_SUCCESS;
}

BSL_STATUS cfg_open(const char* file_path, CFG_HANDLER* handler) {
  BSL_STATUS status;
  char* buffer;
  status = cfg_load_file(file_path, &buffer);
  if (status != BSL_SUCCESS) {
    return status;
  }

  *handler = json_tokener_parse(buffer);
  free(buffer);

  if (*handler == NULL) {
    return BSL_ERROR_CFG_PARSING_GET_NULL_OBJECT;
  }
  return status;
}

BSL_STATUS cfg_close(CFG_HANDLER handler) {
  if (handler) {
    json_object_put(handler);
  }
  return BSL_SUCCESS;
}

CFG_HANDLER cfg_get_field(CFG_HANDLER config, const char* field_name) {
  return json_object_object_get(config, field_name);
}

void cfg_print_all(CFG_HANDLER config) {
  const char* string = json_object_to_json_string_ext(config, JSON_C_TO_STRING_PRETTY || JSON_C_TO_STRING_SPACED);
  if (string) {
    printf("%s\n", string);
  }
}

bool cfg_get_autoscan_switch(CFG_HANDLER handler) {
  struct json_object* jdev = json_object_object_get(handler, "autoscan");
  return (bool)json_object_get_boolean(jdev);
}

bool cfg_type_is_enabled(CFG_HANDLER config) {
  struct json_object* enabled_obj = json_object_object_get(config, "enabled");
  if (!enabled_obj) {
    return false;
  }
  if (json_object_get_boolean(enabled_obj)) {
    return true;
  }
  return false;
}
