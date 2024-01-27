#include "openvino/runtime/core.hpp"

#ifndef IN_OV_COMPONENT
#    define IN_OV_COMPONENT
#    define WAS_OV_LIBRARY_DEFINED
#endif

#ifdef WAS_OV_LIBRARY_DEFINED
#    undef IN_OV_COMPONENT
#    undef WAS_OV_LIBRARY_DEFINED
#endif

int main() {
    OPENVINO_SUPPRESS_DEPRECATED_START
    //! [part0]
    ov::Core core;
    // Load GPU Extensions
    core.set_property("GPU", {{ "CONFIG_FILE", "<path_to_the_xml_file>" }});
    //! [part0]
    OPENVINO_SUPPRESS_DEPRECATED_END

    return 0;
}
