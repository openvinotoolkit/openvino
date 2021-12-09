#include <ie_core.hpp>
#include <fstream>
#include <vector>

void decrypt_file(std::ifstream & stream,
                  const std::string & pass,
                  std::vector<uint8_t> & result) {
}

int main() {
//! [part0]
std::vector<uint8_t> model;
std::vector<uint8_t> weights;

std::string password; // taken from an user
std::ifstream model_file("model.xml"), weights_file("model.bin");

// Read model files and decrypt them into temporary memory block
decrypt_file(model_file, password, model);
decrypt_file(weights_file, password, weights);
//! [part0]

//! [part1]
InferenceEngine::Core core;
// Load model from temporary memory block
std::string strModel(model.begin(), model.end());
InferenceEngine::CNNNetwork network = core.ReadNetwork(strModel, 
    InferenceEngine::make_shared_blob<uint8_t>({InferenceEngine::Precision::U8,
        {weights.size()}, InferenceEngine::C}, weights.data()));
//! [part1]

return 0;
}
