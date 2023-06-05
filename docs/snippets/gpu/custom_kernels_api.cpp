#include "openvino/runtime/core.hpp"

#ifndef IN_OV_COMPONENT
#    define IN_OV_COMPONENT
#    define WAS_OV_LIBRARY_DEFINED
#endif

#include "ie_plugin_config.hpp"

#ifdef WAS_OV_LIBRARY_DEFINED
#    undef IN_OV_COMPONENT
#    undef WAS_OV_LIBRARY_DEFINED
#endif

int main() {
    IE_SUPPRESS_DEPRECATED_START
    //! [part0]
    ov::Core core;
    // Load GPU Extensions
    core.set_property("GPU", {{ CONFIG_KEY(CONFIG_FILE), "<path_to_the_xml_file>" }});
    //! [part0]
    IE_SUPPRESS_DEPRECATED_END

    return 0;
}
