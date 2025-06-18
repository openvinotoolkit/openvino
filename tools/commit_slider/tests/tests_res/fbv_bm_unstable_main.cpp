#include <iostream>

int main () {
    const char *patchGenerator = R"V0G0N(
[
    {
        "str": "std::cout << \"prefix\\nThroughput: 1000.0 FPS\\n\";",
        "comment": "success_1"
    },
    {
        "str": "std::cout << \"prefix\\nThroughput: 1000.0 FPS\\n\";",
        "comment": "success_2"
    },
    {
        "str": "std::cout << \"prefix\\nThroughput: 500.0 FPS\\n\";",
        "comment": "error_1",
        "state": "BREAK"
    },
    {
        "str": "std::cout << \"prefix\\nThroughput: 500.0 FPS\\n\";",
        "comment": "error_2"
    },
    {
        "str": "std::cout << \"prefix\\nThroughput: 500.0 FPS\\n\";",
        "comment": "error_1"
    },
    {
        "str": "std::cout << \"prefix\\nThroughput: 500.0 FPS\\n\";",
        "comment": "error_2"
    },
    {
        "str": "std::cout << \"prefix\\nThroughput: 500.0 FPS\\n\";",
        "comment": "error_1"
    },
    {
        "str": "std::cout << \"prefix\\nThroughput: 0.0 FPS\\n\";",
        "comment": "error_2"
    }
]
)V0G0N";
    return 0;
}