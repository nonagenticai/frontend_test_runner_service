"""The identity a detached job acts as: captured in request scope, re-bound at task start (P7).

A detached job's in-cell calls authenticate with an ``X-W2V-Cell-Token`` that
w2v_common's ``identity_event_hook`` mints FRESH on every send (60 s TTL). It
mints only from what the job's execution context holds: a tenant whose
provenance is VOUCHED, plus the run (w2v job), w2v project and human actor.
Nothing is held, so a job running for hours keeps minting valid tokens.

The job therefore needs exactly one thing from its request: that identity,
captured while ``AuthMiddleware``'s binding is still in scope and bound again,
explicitly, when the task starts. It must never come from a header (a header is
settable by any peer) and never from the forwarded licence (the cell lane
carries none; the licence forwarding is the dual-send fallback scheduled for
removal).

The tenant and its provenance are captured TOGETHER from the authenticated
binding and re-bound together, so the provenance can never be carried onto a
different tenant.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from w2v_common.auth.identity_context import (
    get_actor_id_from_context,
    get_license_id_from_context,
    get_subject_id_from_context,
    get_tenant_id_from_context,
    get_tenant_source_from_context,
    get_w2v_job_id_from_context,
    get_w2v_project_id_from_context,
    parse_attribution_uuid,
    set_identity_context,
    set_w2v_job_context,
    set_w2v_project_context,
)
from w2v_common.tenant import MissingTenantError, TenantSource, parse_tenant_id


@dataclass(frozen=True, slots=True)
class TaskIdentity:
    """Who a detached job acts for. ``tenant_id`` is mandatory: a job never runs tenant-less."""

    tenant_id: uuid.UUID
    tenant_source: TenantSource | None
    run: str | None = None
    project: str | None = None
    actor: str | None = None
    #: The AUTHENTICATED subject / licence the request was attributed to (who pays).
    #: The hook stamps them on every outbound call and, from w2v_common 0.93.0,
    #: signs them into the cell token — a job that drops them bills NULL.
    subject_id: str | None = None
    license_id: str | None = None

    @classmethod
    def unvouched(cls, tenant_id: str | uuid.UUID) -> TaskIdentity:
        """Tenant only, NO provenance: RLS binds, but no cell token is minted and nothing delegates."""
        return cls(parse_tenant_id(tenant_id, source="TaskIdentity.unvouched"), None)


def capture_task_identity(tenant_id: str | uuid.UUID) -> TaskIdentity:
    """Snapshot the authenticated request's identity for a job it is about to detach.

    Call in REQUEST scope. ``tenant_id`` is the tenant the route resolved; it
    must be the one ``AuthMiddleware`` bound, or this raises — a job is never
    dispatched for a tenant the request did not authenticate as, and never
    without one (``MissingTenantError``, fail loud at dispatch, not 401s later
    from inside a job that already returned 202).
    """
    bound = get_tenant_id_from_context()
    if bound is None:
        raise MissingTenantError(
            "cannot dispatch a background job: no authenticated tenant is bound in "
            "request scope, so its in-cell calls could not be attributed or signed"
        )
    requested = parse_tenant_id(tenant_id, source="capture_task_identity")
    if requested != bound:
        raise MissingTenantError(
            f"cannot dispatch a background job for tenant {requested}: the request "
            f"authenticated as tenant {bound}"
        )
    return TaskIdentity(
        tenant_id=bound,
        tenant_source=get_tenant_source_from_context(),
        run=get_w2v_job_id_from_context() or None,
        project=get_w2v_project_id_from_context() or None,
        actor=parse_attribution_uuid(get_actor_id_from_context()),
        # The middleware's authenticated binding only — never a header. (From
        # 0.93.0 this getter excludes an unsigned header subject, so a job can
        # never promote one into a signed claim.)
        subject_id=get_subject_id_from_context() or None,
        license_id=get_license_id_from_context() or None,
    )


def bind_task_identity(identity: TaskIdentity, *, caller_license_token: str | None = None) -> None:
    """Bind ``identity`` in the CURRENT task's context, replacing whatever it inherited.

    Every field is set explicitly — nothing relies on what the task copied from
    the request — so each outbound in-cell call mints its own token for exactly
    this tenant / run / project / actor / subject / licence. ``caller_license_token`` is the
    dual-send licence for ``caller_token_event_hook`` clients only; nothing in
    the cell binding reads it.
    """
    set_identity_context(
        identity.tenant_id,
        identity.license_id,
        identity.subject_id,
        caller_license_token=caller_license_token,
        tenant_source=identity.tenant_source,
        actor_id=identity.actor,
    )
    set_w2v_project_context(identity.project)
    set_w2v_job_context(identity.run)
