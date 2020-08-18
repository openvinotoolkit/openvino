## Software Requirements 
- openjdk 11

### Linux
To install openjdk: 

* Ubuntu 18.04
```bash
sudo apt-get install -y openjdk-11-jdk
```

* Ubuntu 16.04
```bash
sudo apt-get install -y openjdk-9-jdk
```

## Building on Linux

To create Inference Engine Java API add ```-DENABLE_JAVA=ON``` flag in cmake command while building the Inference Engine.
