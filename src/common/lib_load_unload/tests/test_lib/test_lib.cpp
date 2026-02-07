#include <iostream>


#include "shutdown.hpp"

extern "C" void test_lib_shutdown() {
    std::cout << "Test library shutdown: Releasing test library resources..." << std::endl;
}
