#include <ie_core.hpp>

int main() {
using namespace InferenceEngine;
//! [part9]
while(true) {
    // capture frame
    // populate NEXT InferRequest
    // start NEXT InferRequest //this call is async and returns immediately
    // wait for the CURRENT InferRequest //processed in a dedicated thread
    // display CURRENT result
    // swap CURRENT and NEXT InferRequests
}
//! [part9]
return 0;
}
