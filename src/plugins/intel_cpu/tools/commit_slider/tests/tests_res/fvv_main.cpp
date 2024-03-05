#include <iostream>

int main () {
    const char *patchGenerator = R"V0G0N(
[
    {
        "str": "std::cout << \"prefix\\nfailed_1\\n\";",
        "comment": "error_1"
    },
    {
        "str": "std::cout << \"prefix\\nfailed_2\\n\";",
        "comment": "error_2"
    },
    {
        "str": "std::cout << \"prefixsuccess_1\\n\\n\";",
        "comment": "success_1",
        "state": "BREAK"
    },
    {
        "str": "std::cout << \"prefix\\nsuccess_2\\n\";",
        "comment": "success_2"
    }
]
)V0G0N";
    return 0;
}