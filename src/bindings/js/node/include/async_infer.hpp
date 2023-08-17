// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

/** @brief Infers specified inputs and returns the result using callback.
 * @param infer_request an InferRequestWrap object that is used to infer input.
 * @param inputs  An object with a collection of pairs key (input_name) and a value (tensor, tensor's data)
 *  or an Array with values (tensors, tensors' data)
 * @param callback A Javascript Function to call
 */
void asyncInfer(const Napi::CallbackInfo& info);
