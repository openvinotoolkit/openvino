#include <openvino/runtime/core.hpp>

int main() {
//! [part0]
ov::Core core;
auto available_devices = core.get_available_devices();
//! [part0]
return 0;
}
