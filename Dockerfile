FROM gcc:13.2.0

ARG SSH_PASSWORD="root"

# Install required build, inspection, and SSH tools cleanly in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server \
    tree \
    cmake \
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
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/sshd /work \
    && sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && echo "root:$SSH_PASSWORD" | chpasswd \
    && echo "cd /work" >> /root/.bashrc

# Copy compiler and strip wrappers, strip CRLF if built on Windows, and symlink real binaries
COPY wrapper_compiler.sh /wrapper_compiler.sh
COPY wrapper_strip.sh /wrapper_strip.sh
RUN sed -i 's/\r$//' /wrapper_compiler.sh /wrapper_strip.sh \
    && chmod +x /wrapper_compiler.sh /wrapper_strip.sh \
    && mv /usr/local/bin/gcc /usr/local/bin/gcc-real && ln -s /wrapper_compiler.sh /usr/local/bin/gcc \
    && mv /usr/local/bin/g++ /usr/local/bin/g++-real && ln -s /wrapper_compiler.sh /usr/local/bin/g++ \
    && mv /usr/bin/cc /usr/bin/cc-real && ln -s /wrapper_compiler.sh /usr/bin/cc \
    && mv /usr/local/bin/c++ /usr/local/bin/c++-real && ln -s /wrapper_compiler.sh /usr/local/bin/c++ \
    && mv /usr/bin/strip /usr/bin/strip-real && ln -s /wrapper_strip.sh /usr/bin/strip

WORKDIR /work
