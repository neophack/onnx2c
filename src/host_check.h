/* This file is part of onnx2c.
 *
 * Deployment host (domain) authorization for the WebAssembly build.
 *
 * The WASM converter only runs on a whitelist of deployment domains (plus any
 * raw IP address, for self-hosting). Centralizing the whitelist and the check
 * logic here makes it easy to adjust the allowed sites in one place.
 *
 * The check is performed inside the compiled WASM, so simply copying the web
 * frontend to another domain is not enough to use the tool -- the binary
 * itself refuses to produce code. Even if the JS-side redirect is stripped
 * out, this gate still blocks conversion.
 *
 * To change the allowed deployments, edit ALLOWED_DOMAINS below.
 */
#pragma once

#ifdef __EMSCRIPTEN__

#include <string>

#include <emscripten.h>

namespace onnx2c {

/* Whitelisted deployment domains. A host is permitted when it equals one of
 * these, or is a subdomain of one (e.g. "www.freeworld.site"), or is a raw
 * IP address (IPv4/IPv6). Add or remove entries here to change policy. */
inline constexpr const char* const ALLOWED_DOMAINS[] = {
	"neophack.github.io",
	"freeworld.site",
};

/* Canonical site visitors are redirected to when the host is not authorized. */
inline constexpr const char CANONICAL_SITE[] =
	"https://neophack.github.io/onnx2c-web/";

/* Error string returned by the converter when the host is not authorized.
 * JS matches the leading marker prefix to trigger a redirect. */
inline constexpr const char HOST_BLOCKED_ERROR[] =
	"__ONNX2C_HOST_BLOCKED__: this deployment is not authorized. "
	"Redirecting to https://neophack.github.io/onnx2c-web/ ...";

/* The marker prefix JS looks for. Kept in sync with HOST_BLOCKED_ERROR. */
inline constexpr const char HOST_BLOCKED_MARKER[] = "__ONNX2C_HOST_BLOCKED__";

/* Read window.location.hostname from inside the WASM module.
 * Returns an empty string when the host cannot be determined (e.g. file://,
 * sandboxed context). */
inline std::string current_hostname()
{
	char buf[256];
	buf[0] = '\0';
	// Runs in the browser context where `location` is guaranteed to exist
	// (the build uses -sENVIRONMENT=web).
	EM_ASM({
		try {
			var h = (typeof location !== "undefined" && location && location.hostname) || "";
			h = String(h).toLowerCase();
			if (h.length > 255) h = h.slice(0, 255);
			stringToUTF8(h, $0, 256);
		} catch (e) {
			// leave the buffer empty
		}
	}, buf);
	return std::string(buf);
}

/* True if `host` is a literal IP address (IPv4 or IPv6), i.e. not a DNS
 * hostname. Used to permit direct-IP deployments. */
inline bool is_ip_address(const std::string& host)
{
	if (host.empty())
		return false;
	if (host.find(':') != std::string::npos)
		return true; // IPv6 (contains colons)
	bool saw_digit = false;
	for (char c : host)
	{
		if (c >= '0' && c <= '9') { saw_digit = true; continue; }
		if (c == '.') continue;
		return false; // any other char => not an IPv4 literal
	}
	return saw_digit; // all dots/digits with at least one digit => IPv4
}

/* True if `host` equals `suffix` or is a subdomain of it. Respects domain
 * boundaries so "evilgithub.io" does not match "github.io". */
inline bool domain_suffix_match(const std::string& host, const std::string& suffix)
{
	if (host == suffix)
		return true;
	if (host.length() <= suffix.length())
		return false;
	// The char right before the suffix must be a domain separator.
	if (host[host.length() - suffix.length() - 1] != '.')
		return false;
	return host.compare(host.length() - suffix.length(),
	    suffix.length(), suffix) == 0;
}

/* The main policy check. Returns true if `host` (typically
 * current_hostname()) is an authorized deployment. */
inline bool host_is_allowed(const std::string& host)
{
	if (host.empty())
		return false; // file:// or unknown origin
	if (is_ip_address(host))
		return true;
	for (const char* d : ALLOWED_DOMAINS)
	{
		if (domain_suffix_match(host, d))
			return true;
	}
	return false;
}

/* Convenience: check the current runtime host. */
inline bool current_host_allowed()
{
	return host_is_allowed(current_hostname());
}

} // namespace onnx2c

#endif // __EMSCRIPTEN__
