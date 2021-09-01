#include "common.hpp"

namespace py = pybind11;

namespace Common {
const char* string_to_char_arr(const std::string& str) {
    char* arr = new char[str.size() + 1];
    std::copy(str.begin(), str.end(), arr);
    arr[str.size()] = '\0';
    return arr;
}
};  // namespace Common
