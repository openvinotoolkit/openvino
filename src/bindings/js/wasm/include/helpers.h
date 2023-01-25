#pragma once

#include "openvino/openvino.hpp"

std::shared_ptr<ov::Model> loadModel(std::string xml_path, std::string bin_path);

ov::CompiledModel compileModel(std::shared_ptr<ov::Model> model, std::string shape, std::string layout);

ov::Tensor performInference(ov::CompiledModel cm, ov::Tensor t);

ov::Tensor getRandomTensor();
