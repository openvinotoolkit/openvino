#include <inference_engine.hpp>
#include "ie_plugin_config.hpp"
#include "hetero/hetero_plugin_config.hpp"


int main() {
//! [part8]
while(true) {
    // capture frame
    // populate CURRENT InferRequest
    // Infer CURRENT InferRequest //this call is synchronous
    // display CURRENT result
}
//! [part8]
return 0;
}
