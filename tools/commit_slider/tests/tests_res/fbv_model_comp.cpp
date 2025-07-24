#include <iostream>

int main () {
    const char *patchGenerator = R"V0G0N(
[
    {
        "str": "std::cout << \"prefix\\nCompile model took 1000.0 ms\\n\";",
        "comment": "success_1"
    },
    {
        "str": "std::cout << \"prefix\\nCompile model took 1000.0 ms\\n\";",
        "comment": "success_2"
    },
    {
        "str": "std::cout << \"prefix\\nCompile model took 500.0 ms\\n\";",
        "comment": "error_1",
        "state": "BREAK"
    },
    {
        "str": "std::cout << \"prefix\\nCompile model took 500.0 ms\\n\";",
        "comment": "error_2"
    },
    {
        "str": "std::cout << \"prefix\\nCompile model took 500.0 ms\\n\";",
        "comment": "error_1"
    },
    {
        "str": "std::cout << \"prefix\\nCompile model took 500.0 ms\\n\";",
        "comment": "error_2"
    },
    {
        "str": "std::cout << \"prefix\\nCompile model took 500.0 ms\\n\";",
        "comment": "error_1"
    },
    {
        "str": "std::cout << \"prefix\\nCompile model took 500.0 ms\\n\";",
        "comment": "error_2"
    }
]
)V0G0N";
    return 0;
}