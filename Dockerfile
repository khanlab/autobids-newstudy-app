FROM debian:bullseye-20230109 as requirements
RUN echo "deb http://deb.debian.org/debian bullseye-backports main" > /etc/apt/sources.list.d/backports.list \
    && apt-get update -qq \
    && apt-get install -y -q --no-install-recommends \
    build-essential=12.9 \
    ca-certificates=20210119 \
    cryptsetup=2:2.3.7-1+deb11u1 \
    curl=7.74.0-1.3+deb11u10 \
    default-jre=2:1.11-72 \
    fakeroot=1.25.3-1.1 \
    fuse2fs=1.46.2-2 \
    fuse-overlayfs=1.4.0-1 \
    git=1:2.30.2-1+deb11u2 \
    git-annex=8.20210223-2 \
    libcurl4=7.74.0-1.3+deb11u10 \
    libseccomp-dev=2.5.1-1+deb11u1 \
    pkg-config=0.29.2-1 \
    python3=3.9.2-3 \
    python3-pip=20.3.4-4+deb11u1 \
    python3-setuptools=52.0.0-4 \
    python-is-python3=3.9.2-1 \
    squashfs-tools=1:4.4-2+deb11u2 \
    squashfuse=0.1.103-3 \
    ssh=1:8.4p1-5+deb11u2 \
    uidmap=1:4.8.1-1 \
    unzip=6.0-26+deb11u1 \
    wget=1.21-1+deb11u1 \
    && apt-get install -y -q --no-install-recommends -t bullseye-backports golang=2:1.19~1~bpo11+1 \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

FROM requirements as apptainer
ENV APPTAINER_VERSION "1.1.5"
RUN mkdir /opt/download \
    && wget --progress=dot:giga -O /opt/download/apptainer.tar.gz https://github.com/apptainer/apptainer/releases/download/v${APPTAINER_VERSION}/apptainer-${APPTAINER_VERSION}.tar.gz \
    && mkdir /opt/apptainer-src \
    && tar -xzf /opt/download/apptainer.tar.gz -C /opt/apptainer-src
WORKDIR /opt/apptainer-src/apptainer-${APPTAINER_VERSION}
RUN ls \
    && mkdir /opt/apptainer \
    && ./mconfig --prefix=/opt/apptainer --without-suid \
    && make -C ./builddir \
    && make -C ./builddir install
ENV PATH /opt/apptainer/bin:$PATH

FROM apptainer AS apptainer-builds
COPY ./compose/cfmm2tar-custom /opt/cfmm2tar-custom
RUN mkdir /opt/apptainer-images \
    && apptainer build /opt/apptainer-images/cfmm2tar_v1.1.1.sif docker://tristankk/cfmm2tar-custom:v1.1.1 \
    && apptainer build /opt/apptainer-images/tar2bids_v0.2.3.sif docker://khanlab/tar2bids:v0.2.3 \
    && apptainer build /opt/apptainer-images/gradcorrect_v0.0.3a.sif docker://khanlab/gradcorrect:v0.0.3a

FROM requirements as wheel
WORKDIR /opt/autobidsportal
COPY . .
RUN pip install --no-cache-dir pip==22.2.2 poetry==1.3.0 \
    && poetry build -f wheel

FROM requirements as autobidsportal
COPY --from=wheel /opt/autobidsportal/dist/*.whl /opt/wheels/
COPY --from=apptainer /opt/apptainer /opt/apptainer/
COPY --from=apptainer-builds /opt/apptainer-images /opt/apptainer-images/
ENV OTHER_OPTIONS='--tls-aes'
WORKDIR /opt/autobidsportal
COPY ./autobidsportal.ini.example autobidsportal.ini
COPY ./bids_form.py .
RUN git config --system user.name "Autobids Portal" \
    && git config --system user.email "autobids@dummy.com" \
    && WHEEL=$(ls /opt/wheels | grep whl) \
    && pip install --no-cache-dir "/opt/wheels/${WHEEL}[deploy]" \
    && rm -r /opt/wheels 
ENV DCM4CHE_VERSION=5.24.1
ENV PATH=/apps/dcm4che/dcm4che-${DCM4CHE_VERSION}/bin:/apps/DicomRaw:/apps/cfmm2tar:/opt/apptainer/bin:$PATH
ENV _JAVA_OPTIONS="-Xmx2048m"
CMD ["uwsgi", "--ini=autobidsportal.ini"]