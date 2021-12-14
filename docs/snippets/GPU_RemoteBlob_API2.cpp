#include <ie_core.hpp>
#include <gpu/gpu_context_api_va.hpp>
#include <gpu/gpu_config.hpp>


int main() {
using namespace InferenceEngine;
//! [part2]

// ...


// initialize the objects
CNNNetwork network = ie.ReadNetwork(xmlFileName, binFileName);


// ...


auto inputInfoItem = *inputInfo.begin();
inputInfoItem.second->setPrecision(Precision::U8);
inputInfoItem.second->setLayout(Layout::NCHW);
inputInfoItem.second->getPreProcess().setColorFormat(ColorFormat::NV12);

VADisplay disp = get_VA_Device();
// create the shared context object
auto shared_va_context = gpu::make_shared_context(ie, "GPU", disp);
// compile network within a shared context
ExecutableNetwork executable_network = ie.LoadNetwork(network,
                                                      shared_va_context,
                                                      { { GPUConfigParams::KEY_GPU_NV12_TWO_INPUTS,
                                                          PluginConfigParams::YES } });


// decode/inference loop
for (int i = 0; i < nframes; i++) {
//     ...
    // execute decoding and obtain decoded surface handle
    decoder.DecodeFrame();
    VASurfaceID va_surface = decoder.get_VA_output_surface();
//     ...
    //wrap decoder output into RemoteBlobs and set it as inference input
    auto nv12_blob = gpu::make_shared_blob_nv12(ieInHeight,
                                                ieInWidth,
                                                shared_va_context,
                                                va_surface
                                                );
    inferRequests[currentFrame].SetBlob(input_name, nv12_blob);
    inferRequests[currentFrame].StartAsync();
    inferRequests[prevFrame].Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY);
}
//! [part2]
return 0;
}
