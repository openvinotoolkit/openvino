#pragma once

#include <map>
#include <string>
#include <vector>

namespace ov {
namespace frontend {
namespace paddle {

// Structure to hold PaddlePaddle operator information from JSON
struct Operator {
    std::string type;                                  // Operator type (e.g., "conv2d", "pool2d")
    std::vector<std::string> inputs;                   // Input tensor names
    std::vector<std::string> outputs;                  // Output tensor names
    std::map<std::string, nlohmann::json> attributes;  // Operator attributes
};

}  // namespace paddle
}  // namespace frontend
}  // namespace ov