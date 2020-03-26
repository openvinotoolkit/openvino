# Share Memmory

Based on Android ION

## Driver Part

### Build & Install

```sh
make clean
make debug
sudo rmmod myd_ion
sudo insmod myd_ion.ko
```

or simply

```sh
make install
```

## User Space Part

all interface are inside ion_myx_lib/ion.h, there is also a test/sample named test.c

### Build

```sh
make
```

### Test

```sh
sudo test
```

### Usage

directly use simple code to describe it.

#### only one process

___

```c
ion_user_handle_t handle;
int fd,map_fd;
fd =  ion_open();
//alloc it
ret = ion_alloc(fd, len, align, heap_mask, alloc_flags, &handle);

//map the kernel buffer
ret = ion_map(fd, handle, len, prot, map_flags, 0, &ptr, &map_fd);

//free it
ret = ion_free(fd, handle);

//close it
ion_close(fd);

//unmap
munmap(ptr, len);
```

#### memory share in usespace

___

main process

```c
ion_user_handle_t handle;
int fd,map_fd, share_fd;
fd =  ion_open();
//alloc it
ret = ion_alloc(fd, len, align, heap_mask, alloc_flags, &handle);


//share it
ret = ion_share(fd, handle, &share_fd);

//here you can pass the share_fd to other process by inter-porcess commnunction
sendmessage(child_process,shared_fd)

//map the kernel buffer, then got buffer here
ptr = mmap(NULL, len, prot, map_flags, share_fd, 0);


//free it
ret = ion_free(fd, handle);

//close it
ion_close(fd);

//unmap
munmap(ptr, len);
```

other process

```c
ion_user_handle_t handle;
int fd,map_fd, share_fd;

//by inter-porcess commnunction, get a shared_fd herere
recvmessage(shared_fd)

fd = ion_open();

//got buffer here
ptr = mmap(NULL, len, prot, map_flags, shared_fd, 0);
```
