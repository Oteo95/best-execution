FROM tensorflow/tensorflow:latest

COPY dockerfiles/requirements.txt /tmp/

RUN set -ex; \
    pip3 --no-cache-dir install --requirement /tmp/requirements.txt

ARG USERNAME=best_exec_agent
ARG USER_UID=8989
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME
ENV PYTHONPATH "/workspaces/best-execution/src"
