#include <inference_engine.hpp>

int main() {
using namespace InferenceEngine;
//! [part4]
subgraph1: 1. input preprocessing (mean data/FPGA):EXECUTED       layerType:                    realTime: 129        cpu: 129            execType:
subgraph1: 2. input transfer to DDR:EXECUTED       layerType:                    realTime: 201        cpu: 0              execType:
subgraph1: 3. FPGA execute time:EXECUTED       layerType:                    realTime: 3808       cpu: 0              execType:
subgraph1: 4. output transfer from DDR:EXECUTED       layerType:                    realTime: 55         cpu: 0              execType:
subgraph1: 5. FPGA output postprocessing:EXECUTED       layerType:                    realTime: 7          cpu: 7              execType:
subgraph1: 6. copy to IE blob:EXECUTED       layerType:                    realTime: 2          cpu: 2              execType:
subgraph2: out_prob:          NOT_RUN        layerType: Output             realTime: 0          cpu: 0              execType: unknown
subgraph2: prob:              EXECUTED       layerType: SoftMax            realTime: 10         cpu: 10             execType: ref
Total time: 4212     microseconds
//! [part4]
return 0;
}
