#pragma once

#include <iostream>

#include "openvino/openvino.hpp"
#include "./helpers.h"

class Session {
  private:
    ov::CompiledModel model;
  public:
    std::string shape;
    int output_tensor_size;
    Session(std::string xml_path, std::string bin_path, std::string shape, std::string layout);
    uintptr_t run(uintptr_t arrayBuffer, int size);
};
