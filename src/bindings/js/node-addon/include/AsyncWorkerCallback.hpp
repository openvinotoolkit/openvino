
#pragma once

#include <napi.h>

#include <openvino/runtime/core.hpp>

#include "ModelWrap.hpp"
#include "errors.hpp"

Napi::Value read_model_callback(const Napi::CallbackInfo&);
