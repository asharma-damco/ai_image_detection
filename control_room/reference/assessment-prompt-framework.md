# Tradecure Platform — Lead Technical Specialist Assessment Prompt Framework
## SOW 15042026 | Dagriya FinTech | Damco Engagement

---

> **How to use this framework:**
> Each prompt is self-contained. Load the relevant files into context before running.
> Every prompt is written to extract findings that map to a specific SOW deliverable (D1–D8).
> Prompts are sequenced by day per the SOW assessment timeline.
> Where a prompt says [LOAD: file/module], open those files in context first.

---

## PRE-ASSESSMENT PRIMER (Run this first — Day 1 Morning)

### PRIMER-01 | Codebase Orientation & Risk Surface Mapping

```
You are a lead technical specialist conducting a formal technical assessment of a live
algorithmic trading platform called Tradecure. The platform is built on NestJS (backend)
and Next.js (frontend), integrated with Indian retail brokers (AngelOne, Zerodha),
and is live on real user accounts trading F&O instruments.

Before I ask specific questions, I need you to do the following:

1. MAP THE RISK SURFACE: Scan the repository structure and identify:
   - Every module, service, and controller in the backend
   - Every external integration point (broker APIs, WebSocket feeds, databases)
   - Every scheduled/cron-based process
   - Every place real money or real orders can be affected

2. IDENTIFY THE BLAST RADIUS: For this platform, a bug does not just cause a UI error.
   It can cause duplicate orders, financial loss, data corruption, or regulatory breach.
   Categorise every module you find into one of:
   - CRITICAL (directly touches orders, money, user accounts, or trading state)
   - HIGH (session management, authentication, real-time data feeds)
   - MEDIUM (reporting, display, configuration)
   - LOW (utility, logging helpers, UI components)

3. FLAG IMMEDIATE RED FLAGS: Before deep analysis, flag anything that is visibly wrong
   at first scan — obvious credential exposure, commented-out auth guards,
   TODO comments in order execution code, console.log in financial logic,
   any try/catch block that swallows errors silently.

Output format:
- Module inventory table with risk category
- Integration point list with protocol and authentication method
- Immediate red flag list with file reference and one-line description
- Your confidence level in the codebase based on first scan (Low/Medium/High)
  with a 2-sentence justification
```

---

## DAY 1 — Security Foundation & Dependency Posture

### SEC-01 | Secrets & Credential Exposure — Full Surface Scan

```
[LOAD: All .env files, docker-compose.yml, Dockerfile, any config/ directories,
 .github/workflows/, any CI/CD configuration files]

You are auditing this codebase for credential and secret exposure.
Think like an attacker who has just obtained read access to this repository.

Conduct the following checks exhaustively:

1. COMMITTED SECRETS SCAN
   Search for: API keys, JWT secrets, database connection strings,
   broker credentials (AngelOne/Zerodha tokens), AWS credentials,
   webhook URLs containing tokens, private keys.
   Locations to check: all .env files, config files, test files,
   seed files, migration files, git history references.

2. HARDCODED VALUES IN SOURCE
   Search TypeScript/JavaScript source for: strings matching patterns of
   API keys (length > 20, alphanumeric), connection strings (mongodb://, postgres://),
   base64-encoded blobs in source, any string containing 'secret', 'password',
   'token', 'key', 'credential' assigned to a variable with a literal value.

3. ENVIRONMENT VARIABLE DISCIPLINE
   - Are all secrets loaded from environment variables?
   - Is there a .env.example file? Does it contain real values or placeholders?
   - Are there any places where process.env values are logged to console?
   - Is there any place where secrets are returned in API responses?

4. BROKER CREDENTIAL HANDLING
   - Where are AngelOne/Zerodha API tokens stored? (Database? Memory? Env?)
   - Are they encrypted at rest? With what mechanism?
   - Are they transmitted over encrypted channels only?
   - What happens when a broker token expires? Is there a refresh mechanism
     that could expose the token in logs?

5. DATABASE CREDENTIAL STORAGE
   - Are user broker credentials stored in the database?
   - If yes: are they encrypted? What algorithm and key management?
   - Is the encryption key also stored in the database (critical failure pattern)?

For each finding, output:
- SEVERITY: Critical / High / Medium
- FILE: exact file path and line
- FINDING: one-sentence description
- EXPLOITATION PATH: how an attacker would use this
- IMMEDIATE ACTION REQUIRED: Yes/No
```

---

### SEC-02 | Authentication Architecture — Complete Guard Coverage Audit

```
[LOAD: All controllers, guards, middleware, auth module, JWT configuration]

You are auditing the authentication and authorisation architecture of this NestJS application.
This is a financial platform — every unprotected endpoint is a direct business risk.

1. ENDPOINT INVENTORY WITH AUTH STATUS
   List every HTTP endpoint (GET, POST, PUT, PATCH, DELETE) and WebSocket gateway.
   For each, determine:
   - Is an @UseGuards() decorator present? Which guard?
   - Is the guard applied at controller level or route level?
   - If no guard: is this endpoint intentionally public? Is that justified?
   - Can this endpoint affect financial data, orders, or user accounts?

2. GUARD IMPLEMENTATION QUALITY
   - Does the JWT guard validate the token signature, expiry, and issuer?
   - Is the JWT secret strong (>256 bits)? Is it the same across environments?
   - Is the 'none' algorithm explicitly rejected?
   - Does the guard verify the token has not been revoked?
   - Is there a token blacklist for logout? Or are tokens valid until expiry regardless?

3. AUTHORISATION GAPS (beyond authentication)
   Authentication proves who you are. Authorisation proves what you can do.
   - For each endpoint that returns or modifies user-specific data: does the code
     verify the authenticated user owns the resource being accessed?
   - Example failure pattern: GET /orders/:id returns any order if authenticated,
     without checking the order belongs to the requesting user.
   - List every endpoint where this ownership check is missing.

4. WEBSOCKET AUTHENTICATION
   - Is the WebSocket connection handshake authenticated?
   - If a client connects with a valid token and the token expires mid-session,
     what happens? Are they disconnected? Can they continue sending messages?
   - Can a client subscribe to another user's data stream?

5. PRIVILEGE ESCALATION PATHS
   - Is there an admin role? How is it granted?
   - Can a regular user perform admin actions by manipulating request payloads?
   - Are there any endpoints that trust user-supplied role claims?

Output a complete endpoint table:
Route | Method | Guard Applied | Ownership Check | Risk if Bypassed | Severity
```

---

### SEC-03 | HTTP Security Hardening — Production Readiness Check

```
[LOAD: main.ts, app.module.ts, any middleware configuration, cors configuration]

Assess the HTTP security hardening of this NestJS application against production
standards for a financial platform.

1. SECURITY HEADERS (Helmet or equivalent)
   Check for presence and correct configuration of:
   - Content-Security-Policy: Is it set? Does it block inline scripts?
   - X-Frame-Options: DENY or SAMEORIGIN?
   - X-Content-Type-Options: nosniff present?
   - Strict-Transport-Security: Is HSTS configured with appropriate max-age?
   - Referrer-Policy: Is it set?
   - Permissions-Policy: Is it configured?
   If Helmet is used: is it using default config (which has known gaps) or custom?

2. CORS CONFIGURATION
   - What is the allowed origin configuration?
   - Is it set to wildcard (*) in any environment?
   - Are credentials allowed (allowCredentials: true)?
   - If credentials are allowed AND origin is wildcard: this is a critical misconfiguration.
   - Does the CORS config differ between environments?

3. RATE LIMITING
   - Is there rate limiting on authentication endpoints (login, token refresh)?
   - Is there rate limiting on order placement endpoints?
   - What is the limit? Is it per-IP, per-user, or global?
   - Can the rate limit be bypassed by rotating IPs or using proxies?
   - What happens when the limit is hit — 429 with Retry-After? Silent drop?

4. REQUEST VALIDATION & INPUT SANITISATION
   - Is class-validator used for all DTOs?
   - Is transform enabled on the ValidationPipe?
   - Is whitelist enabled (strips unknown properties)?
   - Is forbidNonWhitelisted enabled?
   - Are there any endpoints accepting raw body without validation?
   - Is there SQL/NoSQL injection protection?
   - For MongoDB: are there any places where user input is used in a query
     object directly without sanitisation (NoSQL injection risk)?

5. HTTPS ENFORCEMENT
   - Is HTTP-to-HTTPS redirect configured?
   - Are cookies set with Secure, HttpOnly, and SameSite flags?

Output: Pass/Fail/Partial for each control with specific file reference and fix required.
```

---

### SEC-04 | Dependency Vulnerability Assessment

```
[LOAD: backend/package.json, backend/package-lock.json,
 frontend/package.json, frontend/package-lock.json]

Analyse the dependency manifests for this NestJS/Next.js application and identify:

1. KNOWN CVE EXPOSURE
   List all dependencies with known CVEs, grouped by severity.
   For each Critical/High CVE:
   - Package name and current version
   - CVE identifier and description
   - Is the vulnerable code path exercised in this application?
   - Fixed version available?

2. OUTDATED DEPENDENCY RISK
   - Identify packages more than 2 major versions behind current stable
   - Flag any packages that are abandoned or deprecated
   - Identify any packages with known supply chain compromise history

3. DEPENDENCY CHAIN RISKS
   - Are any packages pulling in a significantly larger dependency than expected?
   - Are there duplicate packages (same package, different versions)?
   - Are any dev dependencies accidentally included in the production bundle?

4. BROKER SDK / FINANCIAL LIBRARY AUDIT
   - Identify all packages used for broker integrations
   - Are these official SDKs from the broker? Or third-party wrappers?
   - What is the maintenance status of these packages?
   - Are there any known issues with order handling in these SDKs?

5. LICENSE COMPLIANCE
   - Are there any GPL-licensed packages in a commercial product?
   - Flag any license incompatibilities.

Output: Severity-ranked dependency risk register.
Format: Package | Version | CVE/Risk | Severity | Fix Version | Exercised in Prod Path
```

---

## DAY 2 — Concurrency, Order Integrity & State Management

### CONC-01 | Race Condition & Shared State Deep Dive

```
[LOAD: All NestJS services marked CRITICAL from PRIMER-01,
 specifically any service handling user sessions, order state, or strategy state]

You are analysing this NestJS backend for race conditions and shared mutable state.
NestJS services are singletons by default. Any instance variable on a service
is shared across ALL concurrent requests. This is the most common source of
data corruption in NestJS applications under concurrent load.

1. SINGLETON SERVICE STATE AUDIT
   For every service class, identify any instance variables (this.xxx properties).
   For each instance variable found:
   - What type is it? (Map, Array, Object, primitive, Set)
   - Is it written to during request handling?
   - Is it read during request handling?
   - If written AND read: this is a shared mutable state bug.
   - What is the worst-case corruption scenario?

2. REQUEST-SCOPED STATE ANALYSIS
   - Is REQUEST scope used where it should be?
   - Are there services that should be request-scoped but are registered as singleton?
   - What is the NestJS module registration for each critical service?

3. CONCURRENT ORDER EXECUTION HAZARDS
   - Trace the order placement flow for simultaneous strategy triggers.
   - Is there a TOCTOU vulnerability on daily exposure limit checks?
   - Is there any locking mechanism (mutex, distributed lock, database transaction)?

4. WEBSOCKET STATE CONTAMINATION
   - Where is the mapping of WebSocket clients to users stored?
   - What happens if the same user connects from two browser tabs simultaneously?
   - What happens during server restart?

5. ASYNC/AWAIT PITFALLS
   - Unhandled promise rejections leaving shared state partially modified?
   - Promise.all() failures leaving other promises in-flight?

For each finding:
- DEFECT TYPE: Shared Mutable State / TOCTOU / Unhandled Rejection / Other
- AFFECTED USERS, WORST CASE, FILE + LINE
```

---

### CONC-02 | Order Execution Pipeline — Atomicity & Integrity Audit

```
[LOAD: Order service, order controller, broker integration service,
 any transaction or database write related to order placement]

Trace the complete order lifecycle end-to-end:
Strategy triggers → Pre-checks → Order construction → Broker API call
→ Response handling → Database write → State update → User notification

1. ATOMICITY GUARANTEES
   - Broker API succeeds but database write fails: what is system state?
   - Are retries idempotent? (Non-idempotent retry = duplicate order risk)

2. DUPLICATE ORDER PREVENTION
   - Idempotency key mechanism on order placement?
   - Request timeout handling — can original order have succeeded?
   - Database-level constraint preventing duplicate orders?

3. PRE-TRADE VALIDATION COMPLETENESS
   Before submission, check for: market open, margin/balance, per-order notional cap,
   daily exposure limit, instrument eligibility.

4. ERROR HANDLING IN THE EXECUTION PATH
   - Broker API error: logged? User notified? Strategy paused?
   - Non-standard responses (partial fill, queued): handled explicitly?
   - Catch blocks that log and continue without propagating?

5. ORDER STATE MACHINE CORRECTNESS
   - All valid states defined? Any stuck states with no recovery?
   - Current order state always consistent with broker record?

Output: Full execution pipeline diagram with defect annotations.
```

---

### CONC-03 | WebSocket Connection Lifecycle — Memory & State Leak Analysis

```
[LOAD: WebSocket gateway, WebSocket module, any event emitter or PubSub service,
 Redis adapter if present, client connection handlers]

1. CONNECTION LIFECYCLE COMPLETENESS
   On disconnect: event listener removal, subscription cancellation,
   user-to-socket mapping removal, interval/timeout clearance.

2. MEMORY LEAK IDENTIFICATION
   - Event listeners added on connect without removal on disconnect?
   - setInterval()/setTimeout() inside connection handlers not cleared on disconnect?
   - Socket client list bounded?

3. RECONNECTION HANDLING
   - Reconnect: clean state or stale state inherited?
   - Duplicate subscriptions on reconnect?

4. BROKER FEED MANAGEMENT
   - Per-user or shared broker feed subscription?
   - Fan-out logic thread-safe?
   - In-flight orders if broker feed disconnects during market hours?

5. SCALING CONSTRAINTS
   - Redis adapter for multi-instance WebSocket?
   - Without it: cross-instance event delivery fails.

For each finding: FILE + LINE, failure mode, user impact, fix complexity.
```

---

### CONC-04 | Strategy Scheduling & Cron Execution Integrity

```
[LOAD: All @Cron decorated methods, TasksService, strategy execution service,
 any Bull/BullMQ queue configuration]

1. CRON JOB OVERLAP PREVENTION
   - Job scheduled every 1 min but takes 90s: does next instance start?
   - Mutex/lock preventing overlap?

2. DISTRIBUTED EXECUTION SAFETY
   - Multiple instances: does each run all cron jobs (N× duplicate orders)?
   - Distributed lock ensuring exactly-once execution per trigger?

3. MARKET HOURS ENFORCEMENT
   - Centralised, timezone-aware market hours check (9:15 AM–3:30 PM IST)?
   - Server UTC vs IST conversion bug?
   - Handling at exactly 3:30 PM?

4. STRATEGY STATE CONSISTENCY
   - User disables strategy mid-execution: halted mid-way or completes?
   - Strategy enable/disable checked at start only or throughout?
   - Repeated-failure strategy: auto-disabled or continues triggering?

5. ERROR ISOLATION
   - Strategy A failure prevents Strategy B execution?
   - Dead-letter queue for failed strategy executions?

Output: Risk-rated finding per cron job with overlap scenario diagram.
```

---

## DAY 3 — SEBI Compliance & Security Hardening

### SEBI-01 | Nine-Control Compliance Gap Matrix — Exhaustive Audit

```
[LOAD: Order service, risk management module, audit logging service,
 admin/kill-switch code, database schemas/models]

For EACH of the nine controls, output:
CONTROL | STATUS (COMPLIANT/PARTIAL/NON-COMPLIANT) | EVIDENCE (file:line or "Not found")
GAP DESCRIPTION | REGULATORY CONSEQUENCE | REMEDIATION SCOPE | PRIORITY (P1/P2/P3)

THE NINE CONTROLS:
1. MARKET PRICE PROTECTION — price band vs LTP before submission
2. PER-ORDER NOTIONAL CAP — quantity × price max enforced before placement
3. PER-DAY EXPOSURE LIMIT — running daily total per user, reset at market open
4. ALGO IDENTIFIER TAGGING — algo tag in every broker API order payload
5. IMMUTABLE ORDER AUDIT TRAIL — append-only log at every order state transition
6. KILL SWITCH — halt all strategies + cancel all open orders within seconds
7. FIVE-YEAR DATA RETENTION — no deletion, archival strategy, backup
8. COMPLIANCE REPORTING CAPABILITY — trade reports exportable on demand
9. VAPT CERTIFICATION READINESS — credential hygiene, network segmentation, test isolation

Produce: Compliance scorecard + regulatory risk summary + priority-ordered remediation list.
```

---

### SEBI-02 | Kill Switch — Implementation & Reliability Deep Dive

```
[LOAD: Admin controller, kill switch service, strategy management service,
 order cancellation service]

1. WHAT DOES IT ACTUALLY DO?
   - Stops new orders? Cancels existing open orders? Both atomically?
   - Expected time from activation to all orders cancelled?

2. RELIABILITY UNDER FAILURE
   - Broker API timeout during cancellation: reports success?
   - 10 orders open, 3 cancellations fail: operator notified?
   - Accessible under high server load?

3. ACCESSIBILITY & AUTHORISATION
   - API call? UI button? Both? Admin-only?
   - Remotely triggerable? Confirmation step that delays emergency activation?

4. STATE RECOVERY AFTER KILL SWITCH
   - Platform state after activation?
   - Strategies re-enable on server restart?

5. TESTING EVIDENCE
   - Unit/integration tests for kill switch path?

Rate: PRODUCTION READY / NEEDS IMPROVEMENT / NOT PRODUCTION SAFE
```

---

### SEC-05 | JWT & Session Security — Token Lifecycle Audit

```
[LOAD: Auth module, JWT strategy, token generation/validation code,
 refresh token implementation, logout handler]

1. TOKEN CONFIGURATION — algorithm, secret strength, expiry, refresh token reuse
2. TOKEN STORAGE ON CLIENT — localStorage (XSS vulnerable) vs httpOnly cookie
3. LOGOUT & TOKEN REVOCATION — server-side revocation? blacklist? Redis session?
4. BROKER TOKEN LIFECYCLE — AngelOne/Zerodha daily expiry, renewal during active strategies
5. CONCURRENT SESSION HANDLING — multiple sessions, compromise isolation

Output: JWT security scorecard with severity-rated findings.
```

---

## DAY 4 — Architecture, Database & Infrastructure

### ARCH-01 | System Architecture — Production Readiness Assessment

```
[LOAD: main.ts, app.module.ts, docker-compose.yml, Kubernetes/ECS configs,
 AWS configuration, nginx/reverse proxy config]

1. SINGLE POINT OF FAILURE MAPPING
   Full system topology: app servers, database, in-memory store, WebSocket layer,
   broker API connectivity. Map every component whose failure takes down the platform.

2. IN-MEMORY STATE RISK ASSESSMENT
   All state living only in server memory: strategy states, WebSocket maps,
   cron locks, order caches, session data.
   What happens on restart? On deployment? On auto-scaling?

3. AWS INFRASTRUCTURE ASSESSMENT
   - Single EC2? Load balancer? WebSocket sticky sessions?
   - Auto-scaling config (CPU-based is wrong for I/O-bound apps)?
   - Zero-downtime deployment? CloudWatch alarms?

4. FAILURE RECOVERY POSTURE — RTO, RPO, automated restart, alert path

5. MINIMUM VIABLE ARCHITECTURE FOR PRODUCTION
   What must be added (Redis, queue, read replica)?
   What must be changed (stateless services, distributed locks)?
   Effort per change: Low/Medium/High.
```

---

### DB-01 | Database Design & Query Performance — Production Load Analysis

```
[LOAD: All Mongoose schemas/models, database service files, aggregation pipelines,
 index definitions, migration files]

1. INDEX AUDIT — every query in order execution path: fields queried, index present?
   Compound indexes for userId+date+instrument? Full collection scan flags.

2. UNBOUNDED QUERY IDENTIFICATION — queries without .limit(), aggregations without
   early $match, find() without predicates.

3. MONGODB→POSTGRESQL MIGRATION READINESS
   Rate each schema: EASY / MEDIUM / COMPLEX / REDESIGN REQUIRED
   Embedded docs, dynamic schemas, arrays, aggregation pipelines, ODM features.

4. DATA INTEGRITY CONSTRAINTS
   - Unique constraints where needed? Orphaned record prevention?
   - Financial amounts as float? (CRITICAL — must use integer paise or Decimal128)
   - Timestamps in UTC consistently?

5. CONNECTION POOL CONFIGURATION — pool size, exhaustion handling, connection leaks.

Output: Query performance risk register + MongoDB migration complexity matrix.
```

---

### ARCH-02 | Error Handling Philosophy — Failure Mode Analysis

```
[LOAD: All service files, exception filters, interceptors, global error handler]

1. GLOBAL ERROR HANDLING ARCHITECTURE — NestJS exception filter, consistent format?

2. SILENT FAILURE PATTERN SCAN — empty catch blocks, swallowed promise rejections,
   failed order writes logged but not escalated. Rate every instance CRITICAL.

3. LOGGING QUALITY — structured JSON logging? Correlation IDs? Order events with
   full context? Centralised log store (CloudWatch/ELK) or stdout only?

4. OBSERVABILITY GAPS — orders/min metric, broker error rate, WS connection count,
   strategy execution time. Can ops team verify platform health at 9:30 AM?

5. GRACEFUL DEGRADATION — broker API down: strategies fail gracefully?
   Circuit breaker on external service calls?

Output: Error handling risk register. Every silent failure path marked CRITICAL.
```

---

## DAY 5 — Validation, Compilation & Deliverable Prep

### VALID-01 | Controlled Execution — Concurrent Session Corruption Test Design

```
[LOAD: Auth service, session-related services, user data service]

Design a minimal reproducible test for concurrent user data corruption:
- No live broker credentials required
- Consistently reproducible
- Objectively proves User A's data appears in User B's response

Specify: exact setup, endpoints, sequence, timing, assertion.
Then trace through code and predict: pass or fail? Cite exact line causing failure.
```

---

### VALID-02 | Order Execution Failure Reproduction — Root Cause Trace

```
[LOAD: Complete order execution path — strategy trigger to broker API call]

1. For each failure mode from CONC-02: exact code path, trigger condition,
   observable symptom.

2. For top 3 failure modes: write minimal Jest test reproducing the failure.
   Mock broker API, set up exact conditions, assert incorrect behaviour,
   comment showing correct behaviour.

3. Risk-rank all failure modes: Frequency × Severity = Risk Score.
   Present top 5 highest-risk failure modes.
```

---

### VALID-03 | Prior Assessment Report Validation

```
[LOAD: Prior assessment reports + relevant code sections they reference]

For each prior report finding:
FINDING | PRIOR STATUS | INDEPENDENT VERIFICATION (YES/NO/PARTIAL with file ref)
CURRENT STATUS: CONFIRMED / RESOLVED / INACCURATE / SUPERSEDED / NOT VERIFIABLE

NEW FINDINGS: Critical/High findings NOT in any prior report.

Summary: Prior report accuracy rate | Findings resolved | New findings not previously identified.
```

---

### MASTER-COMPILE | Validated Issue List (VIL) — D5 Deliverable

```
For each issue, complete every field:

VIL-[NNN]
WORKSTREAM | CATEGORY | SEVERITY (Critical/High/Medium/Low)
TITLE | FILE REFERENCE | ROOT CAUSE | FAILURE MODE
USER IMPACT | FINANCIAL IMPACT | REGULATORY IMPACT
REPRODUCED VIA | PRIOR REPORT STATUS
REMEDIATION APPROACH | ESTIMATED EFFORT | IMPLEMENTATION PHASE

SUMMARY STATISTICS: Critical/High/Medium/Low counts,
SEBI non-compliant controls, security immediate-action items,
total remediation effort estimate.

STOP-SHIP CRITERIA: Findings that mean the platform should not process
live orders until resolved. Be direct. Be specific.
```

---

## SPECIALIST DEEP-DIVES (Run when specific issues are confirmed)

### DEEP-01 | Financial Calculation Integrity Audit

```
[LOAD: Any service computing P&L, margin, quantity, price, or monetary values]

1. Every monetary computation: what numeric type? JavaScript number = float64 = WRONG.
2. Decimal/BigDecimal library in use?
3. Monetary values stored as integers (paise) in DB? Or float?
4. Worst-case rounding error on typical F&O order (lot size × price)?
```

---

### DEEP-02 | NestJS Dependency Injection & Module Architecture Quality

```
[LOAD: All *.module.ts files, provider registrations]

1. Circular dependencies — forwardRef() usage?
2. Module boundary discipline — god modules, unnecessary exports?
3. Provider scope correctness — per-request state in singleton scope?
4. Testability — constructor injection? External deps as interfaces?
   Test coverage on order execution path (target >80% on CRITICAL modules)?
```

---

### DEEP-03 | API Design & Contract Quality — Consumer Risk Assessment

```
[LOAD: All controllers, DTOs, API response structures, OpenAPI/Swagger config]

1. Response consistency — HTTP status codes used correctly?
   200 with error in body (breaks frontend error handling)?
2. DTO validation completeness — class-validator on all DTOs?
3. API versioning strategy — can frontend break during remediation?
4. Pagination on all list endpoints — orders/trades without limit = DoS risk.
5. Swagger/OpenAPI documentation present?

Output: API quality scorecard + contract risks.
```

---

## ASSESSMENT CLOSE-OUT

### ROADMAP-01 | Phased Remediation Roadmap — D7 Deliverable

```
PHASE 1 — CRITICAL STABILISATION (Week 1-2)
Criteria: Critical findings with immediate financial/regulatory/operational risk.
Per item: VIL reference, effort estimate, dependency.

PHASE 2 — PRODUCTION HARDENING (Week 3-6)
Criteria: High severity, architecture changes, P1/P2 SEBI compliance gaps.
Per item: VIL reference, effort, team skills required.

PHASE 3 — QUALITY & COMPLIANCE COMPLETION (Week 7-12)
Criteria: Medium severity, remaining SEBI controls, PostgreSQL migration,
test coverage, observability.

For each phase: total effort (person-days), team composition, prerequisites,
definition of done, risk if skipped.

IMPLEMENTATION SOW INPUTS (D8):
1. Validated scope summary
2. Phased workstream breakdown
3. Basis of estimate
4. Risk register for implementation engagement
5. Required client involvement during implementation
```

---

*End of Assessment Prompt Framework*
*SOW 15042026 | Dagriya FinTech | Damco Engagement*
*Total Prompts: 20 primary + 3 deep-dive + 1 compile + 1 roadmap = 25 prompts*
