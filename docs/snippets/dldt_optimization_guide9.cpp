#include <inference_engine.hpp>
#include "ie_plugin_config.hpp"
#include "hetero/hetero_plugin_config.hpp"


int main() {
using namespace InferenceEngine;
//! [part9]
while(…) {
	capture frame
	populate NEXT InferRequest
	start NEXT InferRequest //this call is async and returns immediately
	wait for the CURRENT InferRequest //processed in a dedicated thread
	display CURRENT result
	swap CURRENT and NEXT InferRequests
}
//! [part9]
return 0;
}
