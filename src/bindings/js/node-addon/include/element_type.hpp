#pragma once
#include <napi.h>

#include "helper.hpp"

/// \brief Creates JS object to represent C++ enum class Type_t with possible element types
Napi::Value enumElementType(const Napi::CallbackInfo& info);