#include <ie_core.hpp>

int main() {
int FLAGS_bl = 1;
auto imagesData = std::vector<std::string>(2);
auto imagesData2 = std::vector<std::string>(4);
//! [part0]
int dynBatchLimit = FLAGS_bl; //take dynamic batch limit from command line option

// Read network model
InferenceEngine::Core core;
InferenceEngine::CNNNetwork network = core.ReadNetwork("sample.xml");


// enable dynamic batching and prepare for setting max batch limit
const std::map<std::string, std::string> dyn_config = 
{ { InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED, InferenceEngine::PluginConfigParams::YES } };
network.setBatchSize(dynBatchLimit);

// create executable network and infer request
auto executable_network = core.LoadNetwork(network, "CPU", dyn_config);
auto infer_request = executable_network.CreateInferRequest();

// ...

// process a set of images
// dynamically set batch size for subsequent Infer() calls of this request
size_t batchSize = imagesData.size();
infer_request.SetBatch(batchSize);
infer_request.Infer();

// ...

// process another set of images
batchSize = imagesData2.size();
infer_request.SetBatch(batchSize);
infer_request.Infer();
//! [part0]

return 0;
}
