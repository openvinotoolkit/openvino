#include <fstream>
#include <vector>

#include "openvino/runtime/core.hpp"

void decrypt_file(std::ifstream & stream,
                  const std::string & pass,
                  std::vector<uint8_t> & result) {
}

int main() {
//! [part0]
std::vector<uint8_t> model_data, weights_data;

std::string password; // taken from an user
std::ifstream model_file("model.xml"), weights_file("model.bin");

// Read model files and decrypt them into temporary memory block
decrypt_file(model_file, password, model_data);
decrypt_file(weights_file, password, weights_data);
//! [part0]

//! [part1]
ov::Core core;
// Load model from temporary memory block
std::string str_model(model_data.begin(), model_data.end());
auto model = core.read_model(str_model,
    ov::Tensor(ov::element::u8, {weights_data.size()}, weights_data.data()));
//! [part1]

return 0;
}
