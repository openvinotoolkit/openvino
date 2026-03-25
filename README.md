## Installation

### Prerequisites

* Python 3.6 or later
* pip or conda package manager

### Installation Methods

#### pip

```bash
pip install openvino
``` 

#### conda

```bash
conda install -c intel openvino
``` 

#### Docker

```bash
docker pull openvino/openvino-toolkit
``` 

#### Source Build

```bash
git clone https://github.com/openvinotoolkit/openvino.git
cd openvino
mkdir build
cd build
cmake ..
cmake --build .
``` 

### Quick Start

```python
import openvino as ov

# Load the model
model = ov.read_model('model.xml')

# Compile the model
compiled_model = ov.compile_model(model, 'CPU')

# Create an execution engine
exec_engine = ov.ExecutionEngine(compiled_model)

# Run inference
output = exec_engine.run()

print(output)
``` 

### API Reference

https://docs.openvino.ai/latest/index.html

## Contributing

* Fork the repository
* Create a new branch for your feature or bug fix
* Commit your changes
* Push to the branch
* Open a pull request

## License

Apache-2.0
