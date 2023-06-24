// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <torch/extension.h>

void regclass_mha_gpt(pybind11::module m);
void regclass_emb_gpt(pybind11::module m);
void regclass_attn_gpt(pybind11::module m);