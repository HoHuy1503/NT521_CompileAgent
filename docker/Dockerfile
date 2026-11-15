FROM gcc:13.2.0

ARG SSH_PASSWORD="root"

ENV DEBIAN_FRONTEND=noninteractive \
    GIT_TERMINAL_PROMPT=0

# Install build, SSH, and Stage 3 security analysis tools in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server \
    tree \
    cmake \
    ninja-build \
    meson \
    bison \
    flex \
    gettext \
    zlib1g-dev \
    libssl-dev \
    vim \
    proxychains4 \
    git \
    file \
    binutils \
    build-essential \
    autoconf \
    automake \
    libtool \
    pkg-config \
    bear \
    cppcheck \
    gdb \
    python3 \
    python3-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/sshd /work /opt/rules \
    && sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && echo "root:$SSH_PASSWORD" | chpasswd \
    && echo "cd /work" >> /root/.bashrc

# Install Semgrep inside an isolated virtualenv (PEP 668 safe on Debian 12)
RUN python3 -m venv /opt/semgrep \
    && /opt/semgrep/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/semgrep/bin/pip install --no-cache-dir semgrep \
    && ln -sf /opt/semgrep/bin/semgrep /usr/local/bin/semgrep

# Copy custom C/C++ Semgrep rules
COPY ./rules /opt/rules
RUN sed -i 's/\r$//' /opt/rules/*

# Copy compiler and strip wrappers, strip CRLF if built on Windows, and symlink real binaries
COPY ./docker/wrappers/wrapper_compiler.sh /wrapper_compiler.sh
COPY ./docker/wrappers/wrapper_strip.sh /wrapper_strip.sh
RUN sed -i 's/\r$//' /wrapper_compiler.sh /wrapper_strip.sh \
    && chmod +x /wrapper_compiler.sh /wrapper_strip.sh \
    && mv /usr/local/bin/gcc /usr/local/bin/gcc-real && ln -s /wrapper_compiler.sh /usr/local/bin/gcc \
    && mv /usr/local/bin/g++ /usr/local/bin/g++-real && ln -s /wrapper_compiler.sh /usr/local/bin/g++ \
    && mv /usr/bin/cc /usr/bin/cc-real && ln -s /wrapper_compiler.sh /usr/bin/cc \
    && mv /usr/local/bin/c++ /usr/local/bin/c++-real && ln -s /wrapper_compiler.sh /usr/local/bin/c++ \
    && mv /usr/bin/strip /usr/bin/strip-real && ln -s /wrapper_strip.sh /usr/bin/strip

WORKDIR /work
