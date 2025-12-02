#include <iostream>

int main () {
    const char *patchGenerator = R"V0G0N(
[
    {
        "str": "std::cout << \"prefix\\n[100%] Built target\\n\";",
        "comment": "error_1"
    },
    {
        "str": "std::cout << \"prefix\\n[100%] Built target\\n\";",
        "comment": "error_2"
    },
    {
        "str": "wrong code #1",
        "comment": "wrong code 1",
        "state": "BREAK"
    },
    {
        "str": "wrong code #2",
        "comment": "wrong code 2"
    }
]
)V0G0N";
    return 0;
}