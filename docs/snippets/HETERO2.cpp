#include <ie_core.hpp>

int main() {
using namespace InferenceEngine;
//! [part2]
InferenceEngine::Core core;
auto network = core.ReadNetwork("sample.xml");
auto executable_network = core.LoadNetwork(network, "HETERO:FPGA,CPU");
//! [part2]
return 0;
}
