FROM nvidia/cuda:11.6.1-devel-ubuntu20.04

ARG UID
ARG GID

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y xrdp
RUN apt update && apt install -y --no-install-recommends --no-install-suggests \
    openssh-server \
    sudo supervisor \
    lxde \
    build-essential cmake ninja-build libopencv-dev \
    vim git curl wget zsh tmux

RUN groupadd ubuntu -g "${GID}"
RUN useradd ubuntu \
    -m \
    -u "${UID}" -g "${GID}" \
    -G sudo \
    -s /usr/bin/zsh \
    -p '$1$HsFW8xpG$AQVUMz3uUEDaIbR/uwpam.'

ADD --chown ubuntu:ubuntu .zshrc /home/ubuntu/.zshrc

# ssh server
RUN mkdir /var/run/sshd

RUN echo 'Port 22' >> /etc/ssh/sshd_config
RUN echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config

EXPOSE 22

# xrdp
# Disable forking, new cursors and enable high tls ciphers for xrdp
RUN sed -i ' \
    s/fork=true/fork=false/g; \
    s/#tls_ciphers=HIGH/tls_ciphers=HIGH/g; \
    s/^new_cursors=true/new_cursors=false/g \
    ' /etc/xrdp/xrdp.ini

# Disable root login and syslog logging for xrdp-sesman
RUN sed -i ' \
    s/AllowRootLogin=true/AllowRootLogin=false/g; \
    s/EnableSyslog=1/EnableSyslog=0/g \
    ' /etc/xrdp/sesman.ini

# Allow all users to start xserver
RUN echo 'allowed_users=anybody' > /etc/X11/Xwrapper.config

EXPOSE 3389

ADD supervisord.conf /etc/supervisord.conf
CMD ["/usr/bin/supervisord"]
