FROM centos:7

RUN yum -y update && \
    yum -y --enablerepo=extras install epel-release && \
    yum -y install \
    gcc gcc-c++ \
    cmake3 make \
    git \
    curl unzip \
    autoconf automake autogen libtool \
    wget patch diffutils zlib-devel ncurses-devel \
    python python-devel python-setuptools \
    doxygen graphviz \
    which \
    'perl(Data::Dumper)'

RUN ln -s /usr/bin/cmake3 /usr/bin/cmake

RUN cmake --version
RUN make --version
RUN gcc --version
RUN c++ --version

# Install nGraph in /root/ngraph
COPY . /root/ngraph
RUN mkdir /root/ngraph/build
WORKDIR /root/ngraph/build

RUN cmake .. -L -DNGRAPH_ONNX_IMPORT_ENABLE=TRUE

RUN make -j"$(nproc)"
RUN make install

CMD cd /root/ngraph/build && make unit-test-check
