#include <inference_engine.hpp>

int main() {
using namespace InferenceEngine;
//! [part0]
std::vector<uint8_t> model;
std::vector<uint8_t> weights;

// Read model files and decrypt them into temporary memory block
decrypt_file(model_file, password, model);
decrypt_file(weights_file, password, weights);
//! [part0]

//! [part1]
Core core;
// Load model from temporary memory block
std::string strModel(model.begin(), model.end());
CNNNetwork network = core.ReadNetwork(strModel, make_shared_blob<uint8_t>({Precision::U8, {weights.size()}, C}, weights.data()));
//! [part1]

return 0;
}
