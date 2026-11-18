# Operation ID Plan: frontend-test-runner-service

Canonical registry of `operation_id` values for every route in
`frontend_test_runner_service`. These IDs feed the org-wide n8n custom-node
generator and the `ultrareview` cross-service catalogue, so they must be:

- **stable** — never rename without coordinating downstream,
- **`verb_noun` snake_case** — matches the org standard (nlp_service,
  worker_management, kag, analysis_service),
- **globally unique** — disambiguated against the rest of the fleet's verbs.

This service is a Playwright test-execution microservice (single async-job
business surface plus three standard health probes). Job verbs are namespaced
with `_test_job_` so they do not collide with sibling jobs (BDD, BDD-gen,
analyze, redesign, review, critique) elsewhere in the fleet.

## Files to edit

- `src/asgi.py` (service banner — `service_root`)
- `src/api/health.py` (3 health probes)
- `src/api/routes.py` (5 test-runner endpoints)

## Mapping Tables

### `src/asgi.py` (tag: `health`)

| Method | Function | operation_id    | Rationale                                   |
|--------|----------|-----------------|---------------------------------------------|
| GET    | `root`   | `service_root`  | Service banner + endpoint index at `/`.     |

### `src/api/health.py` (tag: `health`)

| Method | Function    | operation_id   | Rationale                                                          |
|--------|-------------|----------------|--------------------------------------------------------------------|
| GET    | `health`    | `health_check` | Basic liveness + service identity at `/health`.                    |
| GET    | `readiness` | `health_ready` | Readiness probe — checks DB + Redis at `/health/ready`.            |
| GET    | `liveness`  | `health_live`  | Process-up liveness probe at `/health/live`.                       |

### `src/api/routes.py` (tag: `test-runner`)

| Method | Function            | operation_id            | Rationale                                                                              |
|--------|---------------------|-------------------------|----------------------------------------------------------------------------------------|
| POST   | `start_execution`   | `execute_test_run`      | Submit a Playwright test-run job. `execute` verb namespaces against generic `start_*`. |
| GET    | `get_job_status`    | `get_test_job_status`   | Poll job status. `_test_job_` infix disambiguates against fleet's other job pollers.   |
| GET    | `get_job_result`    | `get_test_job_result`   | Fetch full execution result (summary + per-test outcomes).                             |
| GET    | `get_report`        | `get_test_job_report`   | Fetch HTML report URL + artifact paths for a completed job.                            |
| GET    | `get_artifact`      | `get_test_job_artifact` | Serve a persisted report artifact file (safe-path-bounded under job dir).              |
