#include <iostream>

int main (int argc, char *argv[]) {
    if (argv[2][0] == 'b') {
        std::cout << "prefix\nfailed\n";
    } else {
        std::cout << "prefix\nsuccess\n";
    };
    const char *patchGenerator = R"V0G0N(
[
    {
        "str": "// depends on model only",
        "comment": "commit #1"
    },
    {
        "str": "// depends on model only",
        "comment": "commit #2"
    },
    {
        "str": "// depends on model only",
        "comment": "commit #3"
    },
    {
        "str": "// depends on model only",
        "comment": "commit #4"
    }
]
)V0G0N";
    return 0;
}