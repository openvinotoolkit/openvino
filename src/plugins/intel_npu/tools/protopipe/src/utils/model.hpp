#pragma once

#include <string>
#include <openvino/openvino.hpp>

#include "scenario/inference.hpp"

namespace utils {

// Loads a model, ensures all IO tensors are named, and saves to a temp file if needed.
OpenVINOParams::ModelPath ensureNamedModel(const OpenVINOParams::ModelPath& modelPath);

std::string make_default_tensor_name(const ov::Output<const ov::Node>& output);

} // namespace utils
