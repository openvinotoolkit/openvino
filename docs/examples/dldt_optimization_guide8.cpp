#include <inference_engine.hpp>
#include "ie_plugin_config.hpp"

#include "hetero/hetero_plugin_config.hpp"


int main() {
using namespace InferenceEngine;
while(…) {

	capture frame

	populate CURRENT InferRequest

	Infer CURRENT InferRequest //this call is synchronous

	display CURRENT result

}

return 0;
}
