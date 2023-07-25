#pragma once
#include <napi.h>

#include "helper.hpp"

/// \brief Creates JS object to represent C++ enum class ResizeAlgorithm
Napi::Value enumResizeAlgorithm(const Napi::CallbackInfo& info);
