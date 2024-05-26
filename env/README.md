# Environment
We developed our work in a Docker container.
To activate the environment, you have to first create a file `.env` with the following content:
```
UID={Current User's UID}
GID={Current User's GID}
```

Then, make sure you have `docker` and `docker-compose` installed, and run:
```sh
docker compose up -d
```

It should start an OpenSSH server and a xrdp server in the container,
and the ports are exposed to your host's `4422` and `13389` respectively.

Finally, you can connect to the environment either with a SSH client or a RDP client.
The default user is `ubuntu` and the password is `ubuntu`.

To stop the environment, you can execute:
```sh
docker compose stop
```
so that you can start it again with:
```sh
docker compose start
```

To destroy it, you can execute:
```sh
docker compose down
```
