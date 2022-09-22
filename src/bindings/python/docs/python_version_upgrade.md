# Python version upgrade

#### Notes
Upgrade described in this documentation file can be useful when using a system such as Ubuntu18, which default Python is no longer supported (in this case Python 3.6). The recommended action is to use a newer system instead of upgrading Python. 

*Warning: You make all changes at your own risk.*

## Building and installing Python for Linux

Download Python from Python releases page and extract it:
https://www.python.org/downloads/release 
```bash
curl -O https://www.python.org/ftp/python/3.8.13/Python-3.8.13.tgz
tar -xf Python-3.8.13.tgz
```

Prepare the build with the `.configure` tool, ensuring that `pip` will be installed:
```bash
cd Python-3.8.13
./configure --with-ensurepip=install
```

Build Python with number of jobs suitable for your machine:
```bash
make -j 8
```

Install your new Python version, making sure not to overwrite system Python by using `altinstall` target:
```bash
sudo make altinstall
```

Verify your installation:
```bash
python3.8 --version
> Python 3.8.13
```
