// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/frontend/manager.hpp"
#include "tokenizer.h"
#include <string>

#define COUNT_OF(A) (sizeof(A) / sizeof(A[0]))
const char split_sequence[] = {'F', 'U', 'Z', 'Z', '_', 'N', 'E', 'X',
                               'T', '_', 'F', 'I', 'E', 'L', 'D'};
const char *PDPD = "paddle";

using namespace ov;
using namespace ov::frontend;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  /// split input buffer to model and params
  Tokenizer tokenizer(data, size, split_sequence, sizeof(split_sequence));
  size_t model_size = 0;
  const void *model_buf = tokenizer.next(&model_size);
  size_t params_size = 0;
  const void *params_buf = tokenizer.next(&params_size);

  try {
    ov::frontend::FrontEndManager frontend_manager = FrontEndManager();
    ov::frontend::FrontEnd::Ptr frontend =
        frontend_manager.load_by_framework(PDPD);
    ov::frontend::InputModel::Ptr input_model;
    std::stringstream model;
    std::stringstream params;
    model << std::string((const char *)model_buf, model_size);
    std::istream* in_model(&model);
    if (params_buf) {
      params << std::string((const char *)params_buf, params_size);
      std::istream* in_params(&params);
      input_model = frontend->load(in_model, in_params);
    } else
      input_model = frontend->load(in_model);
    std::shared_ptr<ov::Model> function = frontend->convert(input_model);
  } catch (const std::exception&) {
    return 0;  // fail gracefully on expected exceptions
  }
  return 0;
}
