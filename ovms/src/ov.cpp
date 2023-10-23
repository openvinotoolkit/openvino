#include <chrono>
#include <thread>
#include <string>
#include <iostream>

#include <openvino/openvino.hpp>

int main()
{
    ov::Core ieCore;
    ieCore.set_property(ov::cache_dir("/home/siddhant/Documents/open-source/openvino/tmp_cache"));

    auto directory_path = ieCore.get_property("GPU", ov::cache_dir);

    std::cout << directory_path << std::endl;

    auto ssdlite_model = ieCore.read_model("/home/siddhant/Documents/open-source/openvino/ovms/src/ssdlite_mobilenet_v2_ov/1/ssdlite_mobilenet_v2.xml");
    ov::CompiledModel compiledModel_ssdlite = ieCore.compile_model(ssdlite_model, "GPU");

    auto brain_model = ieCore.read_model("/home/siddhant/Documents/open-source/openvino/ovms/src/brain-tumor-segmentation-0002-2/1/brain-tumor-segmentation-0002.onnx");
    ov::CompiledModel compiledModel_brain = ieCore.compile_model(brain_model, "GPU");

    auto inception_model = ieCore.read_model("/home/siddhant/Documents/open-source/openvino/ovms/src/inception-resnet-v2-tf/1/inception-resnet-v2-tf.xml");
    ov::CompiledModel compiledModel_inception = ieCore.compile_model(inception_model, "GPU");

    std::this_thread::sleep_for(std::chrono::seconds(1));
    compiledModel_ssdlite = {};
    compiledModel_brain = {};
    compiledModel_inception = {};
    std::cout << "Hello\n"
              << std::flush;
}
