# MMpose 3.11 Dev Environment

This repo provides a stable Docker-based development environment for MMpose using Python 3.11.

## Getting Started

Build and start the dev container:

```bash
docker compose up -d --build
```

Stop the environment:

```bash
docker compose down
```

Open a shell in the running container:

```bash
docker compose exec mmpose-dev bash
```

## Notes

- Source code lives on the host and is mounted into `/workspace`.
- The container runs continuously for iterative development.
- MMpose is not installed yet; this is only the base environment.
