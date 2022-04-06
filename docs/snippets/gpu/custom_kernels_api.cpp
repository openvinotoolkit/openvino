#include <openvino/runtime/core.hpp>

int main() {
    //! [part0]
    ov::Core core;
    // Load GPU Extensions
    core.set_property("GPU", {{ CONFIG_KEY(CONFIG_FILE), "<path_to_the_xml_file>" }});
    //! [part0]

    return 0;
}
