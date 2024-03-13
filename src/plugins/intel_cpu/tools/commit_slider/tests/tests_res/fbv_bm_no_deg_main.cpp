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
    }
]
)V0G0N";
    return 0;
}