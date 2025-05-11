// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <cstring>

class Tokenizer {
public:
  /// Initialize tokenizer with an input and token separator buffers.
  Tokenizer(const void *str, size_t str_size, const void *separator,
            size_t separator_size)
      : str((const uint8_t *)str), str_size(str_size), separator(separator),
        separator_size(separator_size) {}

  /// Get next token.
  const void *next(size_t *token_size) {
    const void *token = this->str;
    if (this->str_size >= this->separator_size) {
      for (size_t i = 0; i < this->str_size - this->separator_size; i++)
        if (0 == memcmp((const uint8_t *)this->str + i, this->separator,
                        this->separator_size)) {
          *token_size = this->str_size - this->separator_size;
          this->str += i + this->separator_size;
          this->str_size -= i + this->separator_size;
          return token;
        }
    }
    *token_size = this->str_size;
    this->str = nullptr;
    this->str_size = 0;
    return token;
  }

private:
  const uint8_t *str;
  size_t str_size;
  const void *separator;
  size_t separator_size;
};
