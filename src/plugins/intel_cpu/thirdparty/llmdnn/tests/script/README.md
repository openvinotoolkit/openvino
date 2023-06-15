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
./build.sh
```
debug version extension(llmdnn needs to config debug version also)
```
DEBUG_EXT=1 ./build.sh
```

run test
```
pytest
```