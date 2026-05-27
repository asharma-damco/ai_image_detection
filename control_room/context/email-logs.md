# Email & Communication Log — Dagriya FinTech Assessment

---

## 2026-05-11 | TYPE: exec | CHG-019

**Prepared by:** Damco Solutions | abhisheks12@damcogroup.com
**Engagement:** SOW 15042026 — Tradecure Platform Assessment
**Status:** Sprint 1 complete — all 8 deliverables delivered

---

```
═══════════════════════════════════════════════════════════
EXECUTIVE SUMMARY — DAGRIYA FINTECH ASSESSMENT
Damco Solutions | SOW 15042026 | Delivered: 2026-05-11
═══════════════════════════════════════════════════════════

ENGAGEMENT OUTCOME
──────────────────
Delivered on time. All 8 SOW deliverables complete.
5-day assessment. 14/14 sprint stories done. 20 source
files produced. 8 client deliverables assembled.

HEADLINE FINDINGS
─────────────────
Findings          40 total  (18 Critical · 15 High · 7 Medium)
Stop-ship items   12        (platform must not process live orders)
SEBI compliance   0 / 9     controls fully met (3 partial, 6 absent)
Architecture      1 / 10    prototype-grade; all state in-memory
Verdict           NOT PRODUCTION READY · NOT SEBI ELIGIBLE

WHAT THIS MEANS FOR THE CLIENT
───────────────────────────────
The Tradecure platform functions correctly in single-user,
single-instance, low-volume conditions. It cannot safely:

  • Run live orders for multiple concurrent users
    → User A's trades silently execute on User B's broker account
      (VIL-011, confirmed by controlled test)

  • Scale horizontally
    → Any second server instance doubles all order placements
      immediately (VIL-017)

  • Receive SEBI algo approval
    → 6 of 9 mandatory controls are completely absent,
      including price protection, notional cap, and daily
      exposure limit (D2 SEBI Gap Matrix)

  • Survive an external VAPT
    → Secrets are committed to the git repository;
      VAPT fails at first scan (VIL-001)

TOP 3 RISKS (UNRESOLVED — CLIENT ACTION REQUIRED)
───────────────────────────────────────────────────
1. LIVE CREDENTIALS IN GIT HISTORY [CRITICAL]
   AngelOne and Zerodha API keys, TOTP secrets, and JWT
   signing keys are committed to version control in Copy.env.
   Any party with repository read access can place real orders
   today. Client must rotate all credentials immediately —
   before any other work begins.

2. CROSS-USER ORDER EXECUTION [CRITICAL · STOP-SHIP]
   SmartApiService holds one JWT token for all users on a
   shared singleton. Under concurrent load, User A's orders
   are placed on User B's broker account. Confirmed by
   controlled test (VALID-01). Financial harm is live.

3. NO SEBI PRE-TRADE RISK CONTROLS [CRITICAL · REGULATORY]
   No price protection, no per-order notional cap, no daily
   exposure limit. A single malfunctioning strategy can
   exhaust a user's entire capital in one order with no
   circuit breaker. SEBI algo approval is blocked until
   all three are implemented.

WINS
────
• First independent baseline established — 40 net-new
  findings; no prior audit had flagged any of them
• Full VIL cross-referenced across D2, D3, D4, D7
  — implementation team can begin Phase 1 immediately
• Phase 1 code samples ready — 9 remediations with
  before/after TypeScript included in D7
• D8 ready for commercial scoping — implementation SOW
  can be issued within 7 business days

REMEDIATION PATH
────────────────
Phase 1  Wk 1–2   Stop-ship fixes        3–4 dev-weeks
Phase 2  Wk 3–6   Production hardening   8–10 dev-weeks
Phase 3  Wk 7–12  SEBI compliance        10–14 dev-weeks
─────────────────────────────────────────────────────
Total    12 weeks                         24–28 dev-weeks
Team     2 Sr NestJS + 1 DevOps + 1 QA

NEXT MILESTONE
──────────────
Client: Rotate broker credentials this week (2 hours, no
        dev required). Cannot wait for SOW signature.
Damco:  Commercial team scopes implementation SOW from D8
        within 7 business days of today.
Gate:   Implementation SOW signed → Phase 1 begins.

DELIVERABLES HANDED OVER
─────────────────────────
D1  Technical Assessment Report     (40 findings, 13 sections)
D2  SEBI Compliance Gap Matrix      (9 controls, full remediation)
D3  Security Finding Register       (38 SF-IDs, OWASP-mapped)
D4  Architecture Readiness          (verdict 1/10, SPOF map)
D5  Validated Issue List            (VIL-001–VIL-040, cross-ref)
D6  Prior Report Validation         (first assessment baseline)
D7  Remediation Roadmap             (3 phases, code samples)
D8  Implementation SOW Inputs       (scope, BOE, team, risk)
═══════════════════════════════════════════════════════════
Prepared by: Damco Solutions | abhisheks12@damcogroup.com
```

---
