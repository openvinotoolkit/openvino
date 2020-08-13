## Software Requirements 
- openjdk 11

### Linux
To install openjdk use: 
```bash
sudo apt-get install -y openjdk-11-jdk
```

## Building on Linux

Build Inference Engine Java API alongside with the Inference Engine build. 
You need to run Inference Engine build with the following flags:

```bash
  cd <IE_ROOT>
  mkdir -p build
  cd build
  cmake -DENABLE_JAVA=ON ..
  make --jobs=$(nproc --all)
```
