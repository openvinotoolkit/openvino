FROM ubuntu:24.04

WORKDIR /opt

RUN apt-get update && apt-get install -y python3 python3-pip pipx python3-pytest dos2unix

COPY standalone_mode_requirements.txt /requirements.txt
COPY tests/requirements.txt /tests_requirements.txt
RUN python3 -m venv penv && . penv/bin/activate && pip install -r /requirements.txt -r /tests_requirements.txt

COPY tests/archtests /opt/tests/archtests
COPY tests/unittests /opt/tests/unittests
COPY tests/assets /opt/tests/assets
COPY tests/*.py /opt/tests/
COPY tests/init.sh /init.sh

COPY *.py /opt/
COPY providers /opt/providers
COPY tools /opt/tools

RUN <<EOF
chmod +x /*.sh
chmod +x /opt/*.py
chmod +x /opt/tests/*.py
chmod +x /opt/tests/archtests/*.py
chmod +x /opt/tests/unittests/*.py

dos2unix /init.sh
dos2unix /opt/*
dos2unix /opt/tests/*.py
dos2unix /opt/tests/archtests/*.py
dos2unix /opt/tests/unittests/*.py
EOF

ENV http_proxy=${http_proxy} https_proxy=${https_proxy} HTTP_PROXY=${http_proxy} HTTPS_PROXY=${https_proxy} no_proxy=${no_proxy} NO_PROXY=${no_proxy}

ENTRYPOINT ["/init.sh"]
