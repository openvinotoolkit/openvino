#pragma once

#include "openvino/openvino.hpp"
#include "./helpers.h"

class Session {
  private:
    std::shared_ptr<ov::Model> model;
  public:
    Session(std::string xml_path, std::string bin_path);
    uintptr_t run(std::string shape, std::string layout, uintptr_t arrayBuffer, int size);
};
