// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "frontend_manager/frontend_manager.hpp"
#include "ngraph/ngraph.hpp"
#include "tokenizer.h"
#include <string>

#define COUNT_OF(A) (sizeof(A) / sizeof(A[0]))
const char split_sequence[] = {'F', 'U', 'Z', 'Z', '_', 'N', 'E', 'X',
                               'T', '_', 'F', 'I', 'E', 'L', 'D'};
const char *PDPD = "pdpd";

using namespace ngraph;
using namespace ngraph::frontend;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  /// split input buffer to model and params
  Tokenizer tokenizer(data, size, split_sequence, sizeof(split_sequence));
  size_t model_size = 0;
  const void *model_buf = tokenizer.next(&model_size);
  size_t params_size = 0;
  const void *params_buf = tokenizer.next(&params_size);

  try {
    ngraph::frontend::FrontEndManager frontend_manager = FrontEndManager();
    ngraph::frontend::FrontEnd::Ptr frontend =
        frontend_manager.load_by_framework(PDPD);
    ngraph::frontend::InputModel::Ptr input_model;
    std::stringstream model;
    model << std::string((const char *)model_buf, model_size);
    if (params_buf) {
      std::stringstream params;
      params << std::string((const char *)params_buf, params_size);
      input_model = frontend->load_from_streams({&model, &params});
    } else
      input_model = frontend->load_from_stream(model);
    std::shared_ptr<ngraph::Function> function = frontend->convert(input_model);
  } catch (const std::exception&) {
    return 0;  // fail gracefully on expected exceptions
  }
  return 0;
}