#include <iostream>

int main (int argc, char *argv[]) {
    if (argv[2][0] == 'b') {
        std::cout << "Throughput: 500.0 FPS\n";
    } else {
        std::cout << "Throughput: 1000.0 FPS\n";
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