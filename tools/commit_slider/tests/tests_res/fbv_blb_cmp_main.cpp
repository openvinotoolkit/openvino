#include <iostream>
#include <fstream>
using namespace std;

int main() {
  ofstream blobFile("blob.txt");
  const char *patchGenerator = R"V0G0N(
    [
        {
            "str": "blobFile << \"address line\\n0.000\\n0.000\";",
            "comment": "success_1"
        },
        {
            "str": "blobFile << \"address line\\n0.000\\n0.000\";",
            "comment": "success_2"
        },
        {
            "str": "blobFile << \"address line\\n1.000\\n0.000\";",
            "comment": "error_1",
            "state": "BREAK"
        },
        {
            "str": "blobFile << \"address line\\n1.000\\n0.000\";",
            "comment": "error_2"
        },
        {
            "str": "blobFile << \"address line\\n1.000\\n0.000\";",
            "comment": "error_3"
        },
        {
            "str": "blobFile << \"address line\\n1.000\\n0.000\";",
            "comment": "error_4"
        },
        {
            "str": "blobFile << \"address line\\n1.000\\n0.000\";",
            "comment": "error_5"
        },
        {
            "str": "blobFile << \"address line\\n1.000\\n0.000\";",
            "comment": "error_6"
        }
    ]
    )V0G0N";
    
  blobFile.close();
  return 0;
}
