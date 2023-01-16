#include "openvino/runtime/core.hpp"
#include "ie_plugin_config.hpp"

int gpu_custom_kernels_api() {
    //! [part0]
    ov::Core core;
    // Load GPU Extensions
    core.set_property("GPU", {{ CONFIG_KEY(CONFIG_FILE), "<path_to_the_xml_file>" }});
    //! [part0]

    return 0;
}
