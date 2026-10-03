# Security Review, Hardening, and UI Remediation

**Date:** 2026-10-02  
**Scope:** Existing repository plus the secure local web UI and API integration.

## Review performed

- Inspected all source, scripts, tests, data-flow entry points, configuration files, and dependency files.
- Searched for API keys, tokens, private keys, passwords, hardcoded credentials, `.env` files, unsafe HTML sinks, shell execution, arbitrary file access, local storage, and unsafe URL handling.
- Checked whether the repository is a Git worktree. It is not currently a Git worktree, so there was no repository history/configuration to audit.
- Reviewed the only project dependency (`pytest`) with `pip-audit` in a clean virtual environment.
- Added the UI only after the review and re-ran tests and validation after every security-relevant fix.

## Findings and remediation

### Secrets and credentials

**Finding:** No actual secret or credential was found.

**Verified:**

- No `.env` file is present.
- `.env.example` contains only an email placeholder for the optional OpenAlex polite pool.
- No frontend/API credentials exist.
- No credentials are embedded in JavaScript or data output.
- No API key is required for the runtime UI.

### Dependency vulnerability

The initial clean audit reported:

- `pytest 8.4.2` with a known vulnerability fixed in `9.0.3`.
- Temporary audit environment `pip 26.1.2` with a vulnerability fixed in `26.2`.

Remediation:

- Updated `requirements.txt` from `pytest>=8,<9` to `pytest>=9.0.3,<10.0`.
- Rebuilt the clean audit environment with `pip 26.2`.
- Re-ran `pip-audit`.

Final result:

```text
No known vulnerabilities found
```

### User-input handling

The primary research idea is treated as untrusted input.

Implemented protections:

- JSON request body required.
- Only the `idea` field is accepted.
- Maximum request body: 12,000 bytes.
- Maximum idea length: 5,000 characters.
- Minimum idea length: 12 characters.
- Invalid UTF-8, malformed JSON, unexpected fields, non-string values, and control characters are rejected.
- User input is never used as a path, shell command, SQL expression, or executable code.
- User input is not logged.

### Frontend rendering

The UI renders research ideas, paper titles, evidence, metadata, and graph labels as data.

Implemented protections:

- No `innerHTML`, `dangerouslySetInnerHTML`, `eval`, `Function`, or `document.write`.
- Dynamic content is rendered with `textContent` and DOM node creation.
- External links are accepted only for `http:` and `https:` URLs.
- External links use `target="_blank"` with `rel="noopener noreferrer"`.
- No sensitive data is stored in `localStorage`, cookies, or query parameters.
- No external scripts, stylesheets, fonts, or images are loaded.

### Backend/API surface

The UI server exposes only:

- `GET /`
- `GET /api/health`
- `GET /api/graph` (a filtered projection of the trusted serialized graph)
- `POST /api/reason`

Implemented protections:

- Exact route matching; no arbitrary file serving.
- Fixed knowledge-state path; no user-controlled file access.
- JSON content type required for reasoning requests.
- Request size limit enforced before parsing.
- Safe, concise client errors.
- Internal exceptions are not sent to the browser.
- No shell execution, subprocess invocation, SQL, uploads, or dynamic imports.
- No CORS headers are enabled; the UI uses same-origin requests only.
- Research data is never executed.

### HTTP/browser headers

The server sends:

- `Content-Security-Policy`
- `X-Content-Type-Options: nosniff`
- `Referrer-Policy: no-referrer`
- `Cross-Origin-Opener-Policy: same-origin`
- `Cross-Origin-Resource-Policy: same-origin`
- restrictive `Permissions-Policy`
- `Cache-Control: no-store`

The CSP intentionally permits inline script/style because the preview-safe UI is a single self-contained HTML document. A production deployment should replace this with nonces or external hashed assets and add deployment-appropriate frame protections.

## Validation after remediation

```text
pytest: 7 passed
pip-audit: No known vulnerabilities found
validate_project.py: PASS
secret scan: no actual credentials found
```

## Remaining deployment risks

These are documented limitations rather than hidden assumptions:

1. The demo server binds to `0.0.0.0` so the sandbox preview can reach it. It should be placed behind HTTPS, an authenticated reverse proxy, and a rate-limited deployment boundary in production.
2. The demo has no user authentication because it is a local research-onboarding prototype.
3. The standard-library server is not intended to be an internet-facing production server.
4. The CSP uses `unsafe-inline` for preview portability. A production build should use CSP nonces or hashed static assets.
5. The knowledge state is trusted as a local build artifact, but fresh external metadata should be regenerated and validated through the existing corpus pipeline rather than uploaded through the UI.

## Security conclusion

No blocking security issue remains for the local demo. The UI was implemented after the review, dependency remediation was applied, and the hardened integration passed the project test and validation suite.
