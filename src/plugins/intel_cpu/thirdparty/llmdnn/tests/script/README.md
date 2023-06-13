# Torch extension to help test

## usage
prepare python enviroment
```
python3 -m venv .env
source .env/bin/activate
pip3 install -r requirements.txt
```

compile extension
```
cmake . -DBUILD_PYTHON_TESTS=ON
```

run test
```
python test_mha_gpt.py
```