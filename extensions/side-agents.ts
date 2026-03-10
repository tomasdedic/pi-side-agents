// ─── Imports ────────────────────────────────────────────────────────────────
// `complete` calls the LLM API; `Message` is the shape of one chat message.
import { complete, type Message } from "@mariozechner/pi-ai";
// Helpers to convert the internal session history into LLM-compatible messages
// and to turn those messages into a single readable string.
import { convertToLlm, serializeConversation } from "@mariozechner/pi-coding-agent";
// Type-only imports: the public surface the extension receives from Pi.
import type { ExtensionAPI, ExtensionContext, SessionEntry } from "@mariozechner/pi-coding-agent";
// TypeBox is used to declare the JSON-Schema shapes of tool parameters.
import { Type } from "@sinclair/typebox";
// spawnSync runs a child process and blocks until it finishes.
import { spawnSync } from "node:child_process";
// `fs.promises` for async file I/O; `readFileSync` for the one synchronous
// settings read at extension load time.
import { promises as fs, readFileSync } from "node:fs";
// Path utilities: basename = filename only, dirname = parent folder,
// join = combine path segments, resolve = absolute path.
import { basename, dirname, join, resolve } from "node:path";

// ═══════════════════════════════════════════════════════════════════════════════
// TYPESCRIPT / JAVASCRIPT CONCEPTS PRIMER
// ═══════════════════════════════════════════════════════════════════════════════
//
// This file uses several TypeScript/JavaScript concepts that may be unfamiliar.
// Here is a plain-English reference for everything you will encounter below.
//
// ── async / await / Promise ───────────────────────────────────────────────────
// JavaScript runs in a single thread. Slow operations (disk reads, network
// calls, shell commands) must not freeze that thread. Instead they return a
// Promise — an object that says "I will give you a value *later*, once the
// operation finishes."
//
//   async function foo(): Promise<string> { return "hello"; }
//   ▸ `async` marks a function that is allowed to pause and resume.
//   ▸ It always returns a Promise, even if you just write `return "hello"`.
//   ▸ Promise<string> = "will eventually produce a string".
//   ▸ Promise<void>   = "will eventually finish, producing nothing useful".
//   ▸ Promise<T | undefined> = "will produce a T, or nothing (undefined)".
//
//   const result = await foo();
//   ▸ `await` pauses THIS function until the Promise resolves (finishes).
//   ▸ You can only write `await` inside an `async` function.
//   ▸ While paused, JavaScript is free to run other code elsewhere.
//   ▸ If the Promise rejects (throws an error), `await` re-throws it here.
//
//   void someAsyncFn();
//   ▸ Calls the async function but deliberately does NOT await it.
//   ▸ "Fire and forget" — we don't care about its result or errors.
//   ▸ Suppresses the TypeScript warning about an un-awaited Promise.
//
// ── try / catch / finally ─────────────────────────────────────────────────────
//   try {
//     riskyOperation();      // if this throws …
//   } catch (err) {
//     handleError(err);      // … execution jumps here with the error
//   } finally {
//     cleanup();             // … this ALWAYS runs, whether or not there was an error
//   }
//   ▸ Works identically with async/await: if an `await` rejects, execution
//     jumps to the nearest `catch` block.
//   ▸ `err` is typed as `unknown` in TypeScript, so you must check it before
//     using it (hence the stringifyError() helper in this file).
//
// ── Types ─────────────────────────────────────────────────────────────────────
//   type Foo = { name: string; age?: number };
//   ▸ Declares a named shape. TypeScript checks this at compile time only;
//     the `type` keyword completely disappears at runtime.
//   ▸ `age?: number` — the `?` means the field is optional: it may be a
//     number, or it may simply not exist (undefined).
//
//   type Status = "running" | "done" | "failed";
//   ▸ A "union type": the variable must hold one of those exact string values.
//   ▸ TypeScript will error if you try to assign anything else.
//
//   Record<string, AgentRecord>
//   ▸ A plain JavaScript object used as a dictionary.
//   ▸ Keys are strings; every value must be an AgentRecord.
//   ▸ Equivalent to writing: { [key: string]: AgentRecord }
//   ▸ Used for the registry's `agents` field: agents["fix-auth"] = { … }
//
// ── Generics  <T> ─────────────────────────────────────────────────────────────
//   async function readJsonFile<T>(path: string): Promise<T | undefined>
//   ▸ The `<T>` is a *type placeholder* filled in at the call site.
//   ▸ readJsonFile<RegistryFile>(path) → T becomes RegistryFile.
//   ▸ readJsonFile<ExitMarker>(path)   → T becomes ExitMarker.
//   ▸ One function, many safe types — TypeScript checks each separately.
//
// ── Collections: Set and Map ──────────────────────────────────────────────────
//   new Set<string>()
//   ▸ A collection of *unique* values (duplicates are silently ignored).
//   ▸ .has(x)  → boolean    .add(x)  → adds x    .size → count
//
//   new Map<string, AgentStatusSnapshot>()
//   ▸ A key→value store where any type can be a key (unlike plain objects).
//   ▸ .get(key) → value or undefined    .set(key, val)    .has(key)
//   ▸ Unlike a plain object, iteration order is guaranteed (insertion order).
//
// ── Arrow functions ───────────────────────────────────────────────────────────
//   const double = (n: number) => n * 2;
//   ▸ A compact way to write a function. No `function` keyword needed.
//   ▸ `(x) => expression`        → returns expression directly (no braces).
//   ▸ `(x) => { statement; }`    → needs an explicit `return` for a value.
//   ▸ Commonly passed as callbacks: array.filter(item => item.ok)
//
// ── Destructuring ─────────────────────────────────────────────────────────────
//   const { repoRoot, agentId } = options;
//   ▸ Pulls named fields out of an object into local variables in one step.
//   ▸ Equivalent to: const repoRoot = options.repoRoot; const agentId = options.agentId;
//
//   const [first, second] = array;
//   ▸ Pulls elements out of an array by position.
//
// ── Spread operator … ─────────────────────────────────────────────────────────
//   [...existingArray, newItem]           // copy array, append newItem
//   { ...existingObject, newKey: val }    // copy object, add/override one field
//   ▸ Does NOT modify the original; always creates a new array/object.
//
// ── Optional chaining  ?. ─────────────────────────────────────────────────────
//   result.error?.message
//   ▸ If result.error is null or undefined, the whole expression is undefined
//     instead of crashing with "Cannot read property 'message' of undefined".
//   ▸ Chains can be longer: a?.b?.c?.d
//
// ── Nullish coalescing  ?? ────────────────────────────────────────────────────
//   result.stdout ?? ""
//   ▸ If result.stdout is null or undefined, use "" as the fallback.
//   ▸ Unlike `||`, it does NOT treat 0 or false as missing — only null/undefined.
//
// ── Template literals ─────────────────────────────────────────────────────────
//   `Hello ${name}, you have ${count} messages.`
//   ▸ Backtick strings that can embed any expression inside ${…}.
//   ▸ Can span multiple lines naturally, no escape sequences needed.
//
// ── Type assertions ───────────────────────────────────────────────────────────
//   value as AgentRecord
//   ▸ Tells TypeScript "treat this as an AgentRecord, I know what I'm doing."
//   ▸ Does NOT change the runtime value at all; purely a compile-time hint.
//   ▸ Use sparingly — if you're wrong, runtime errors follow.
//
// ── Type guard functions ──────────────────────────────────────────────────────
//   (entry): entry is Foo => entry.type === "foo"
//   ▸ A function that returns a boolean AND tells TypeScript: "if true,
//     narrow the type of `entry` to Foo inside the if-block."
//   ▸ Example: .filter((b): b is { type:"text"; text:string } => b.type === "text")
//     Only text blocks pass through, and TypeScript knows b.text exists.
//
// ── NodeJS.Timeout | undefined ────────────────────────────────────────────────
//   let timer: NodeJS.Timeout | undefined
//   ▸ NodeJS.Timeout is the handle returned by setInterval() / setTimeout().
//   ▸ Stored so we can call clearInterval(timer) to cancel it later.
//   ▸ `| undefined` means "not yet started" (before the first call).
//
// ── for … of  and  Object.entries / Object.values ────────────────────────────
//   for (const [key, value] of Object.entries(someObject)) { … }
//   ▸ Object.entries() converts { a:1, b:2 } into [["a",1],["b",2]].
//   ▸ for…of iterates the array; destructuring pulls key and value apart.
//
//   for (const value of Object.values(someObject)) { … }
//   ▸ Like Object.entries() but gives only the values, not the keys.
//
// ═══════════════════════════════════════════════════════════════════════════════

// ─── Environment-variable keys ───────────────────────────────────────────────
// The folder that holds the shared .pi/side-agents registry.
// In a normal repo this is auto-detected; the env var lets tests override it.
const ENV_STATE_ROOT = "PI_SIDE_AGENTS_ROOT";
// Set on the child Pi process so it knows its own agent ID.
const ENV_AGENT_ID = "PI_SIDE_AGENT_ID";
// The session file path of the parent Pi instance, forwarded to the child.
const ENV_PARENT_SESSION = "PI_SIDE_PARENT_SESSION";
// The absolute path of the parent git repository root, forwarded to the child.
const ENV_PARENT_REPO = "PI_SIDE_PARENT_REPO";
// The per-agent runtime directory (logs, kickoff prompt, exit marker).
const ENV_RUNTIME_DIR = "PI_SIDE_RUNTIME_DIR";

// ─── Internal string constants ───────────────────────────────────────────────
// Key under which the status bar line is registered in the Pi TUI.
const STATUS_KEY = "side-agents";
// Schema version stored in registry.json – lets us reject stale files.
const REGISTRY_VERSION = 1;
// Custom session-entry type written by the child to record its own session link.
const CHILD_LINK_ENTRY_TYPE = "side-agent-link";
// Custom message type emitted when an agent changes status (shown in chat).
const STATUS_UPDATE_MESSAGE_TYPE = "side-agent-status";
// Custom message type emitted when an agent is kicked off (carries the prompt).
const PROMPT_UPDATE_MESSAGE_TYPE = "side-agent-prompt";

// ─── System prompt for context summarisation ─────────────────────────────────
// Sent to the LLM when building the kickoff prompt so it only distils
// parent-conversation details that are actually relevant to the child task.
const SUMMARY_SYSTEM_PROMPT = `You are writing a minimal handoff summary for a background coding agent.

Use the parent conversation only as context. Include only details that are directly relevant to the child task.

If the parent conversation is unrelated to the child task, output exactly:
NONE

Preferred content (but only when relevant):
- objective/constraints already established
- decisions already made
- key files/components to inspect
- risks/caveats`;

// ─── Agent status type ────────────────────────────────────────────────────────
// All possible lifecycle states an agent can be in.
// "done" is the success terminal state; "failed"/"crashed" are failure ones.
type AgentStatus =
	| "allocating_worktree"  // setting up the git worktree
	| "spawning_tmux"        // creating the tmux window
	| "running"              // Pi is actively working inside the tmux window
	| "waiting_user"         // child Pi finished a turn and is waiting for input
	| "done"                 // exited with code 0 (auto-removed from registry)
	| "failed"               // exited with a non-zero code
	| "crashed";             // tmux window disappeared before an exit marker was written

// All non-terminal statuses – used when validating user-supplied wait-state lists.
// `AgentStatus[]` = an array where every element must be an AgentStatus value.
// Writing the type explicitly here lets TypeScript catch a typo like "runing".
const ALL_AGENT_STATUSES: AgentStatus[] = [
	"allocating_worktree",
	"spawning_tmux",
	"running",
	"waiting_user",
	"failed",
	"crashed",
];

// The statuses that agent-wait-any stops polling for by default.
const DEFAULT_WAIT_STATES: AgentStatus[] = ["waiting_user", "failed", "crashed"];

// ─── Data shapes (types) ─────────────────────────────────────────────────────

// One row in registry.json – everything the parent needs to track / reconnect to
// a running or finished agent.
type AgentRecord = {
	id: string;                   // unique slug, e.g. "fix-auth-leak"
	parentSessionId?: string;     // .pi session file of the spawning Pi instance
	childSessionId?: string;      // .pi session file of the child Pi instance
	tmuxSession?: string;         // tmux session name the window lives in
	tmuxWindowId?: string;        // tmux @window_id (stable across renames)
	tmuxWindowIndex?: number;     // human-visible tmux window number (#N)
	worktreePath?: string;        // absolute path to the git worktree
	branch?: string;              // git branch, e.g. "side-agent/fix-auth-leak"
	model?: string;               // provider/model string given to child Pi
	task: string;                 // raw task description text
	status: AgentStatus;
	startedAt: string;            // ISO timestamp
	updatedAt: string;            // ISO timestamp of last status change
	finishedAt?: string;          // ISO timestamp when exit marker was read
	runtimeDir?: string;          // folder with kickoff.md, backlog.log, exit.json
	logPath?: string;             // full path to backlog.log
	promptPath?: string;          // full path to kickoff.md
	exitFile?: string;            // full path to exit.json
	exitCode?: number;            // process exit code from exit.json
	error?: string;               // human-readable error string on failure
	warnings?: string[];          // non-fatal warnings accumulated during startup
};

// The on-disk structure of .pi/side-agents/registry.json.
// `type` = compile-time shape only, erased at runtime.
type RegistryFile = {
	version: 1;  // literal type `1` — only the exact number 1 is valid, not 2 or 0
	// Record<string, AgentRecord> is a dictionary:
	//   key   = agent id string, e.g. "fix-auth-leak"
	//   value = an AgentRecord object with all the agent's details
	// Accessing: registry.agents["fix-auth-leak"]  → AgentRecord | undefined
	agents: Record<string, AgentRecord>;
};

// Returned by allocateWorktree() on success.
type AllocateWorktreeResult = {
	worktreePath: string;
	slotIndex: number;    // numeric index of the slot directory (0001, 0002 …)
	branch: string;
	warnings: string[];
};

// Parameters accepted by startAgent().
type StartAgentParams = {
	task: string;
	branchHint?: string;      // caller-supplied slug fragment for the branch name
	model?: string;           // optional model override
	includeSummary: boolean;  // whether to call the LLM to distil parent context
};

// What startAgent() returns to the caller on success.
type StartAgentResult = {
	id: string;
	tmuxWindowId: string;
	tmuxWindowIndex: number;
	worktreePath: string;
	branch: string;
	warnings: string[];
	prompt: string;           // the full kickoff prompt sent to the child
};

// What prepareFreshRuntimeDir() returns.
type PrepareRuntimeDirResult = {
	runtimeDir: string;
	archivedRuntimeDir?: string; // previous runtime dir was moved here
	warning?: string;            // set if archiving failed and we deleted instead
};

// Shape of exit.json written by the launch script when the child process ends.
type ExitMarker = {
	exitCode?: number;
	finishedAt?: string;
};

// Return type from the run() / runOrThrow() helpers.
type CommandResult = {
	ok: boolean;
	status: number | null;
	stdout: string;
	stderr: string;
	error?: string;
};

// Describes a single agent changing from one status to another, used for
// emitting status-change messages to the parent chat.
type StatusTransitionNotice = {
	id: string;
	fromStatus: AgentStatus;
	toStatus: AgentStatus;
	tmuxWindowIndex?: number;
};

// Compact snapshot of an agent's status – kept in memory between status polls
// so we can detect which agents changed and emit transition messages.
type AgentStatusSnapshot = {
	status: AgentStatus;
	tmuxWindowIndex?: number;
};

// ─── Module-level polling state ───────────────────────────────────────────────
// `NodeJS.Timeout | undefined` — either a live timer handle (returned by
// setInterval) or undefined, meaning the timer hasn't been started yet.
let statusPollTimer: NodeJS.Timeout | undefined;
// The most-recent ExtensionContext and ExtensionAPI are stored here so the
// timer callback can call renderStatusLine without a closure over stale values.
// Both start as `undefined` (the `| undefined` union type) until the first
// session_start event fires and assigns real values.
let statusPollContext: ExtensionContext | undefined;
let statusPollApi: ExtensionAPI | undefined;
// A plain boolean. No special type annotation needed — TypeScript infers
// `boolean` from the initial value `false`.
// Guard flag: prevents concurrent status refreshes from stacking up.
let statusPollInFlight = false;
// Map<K, V> is a key→value store.
// Outer key: stateRoot path string → Inner Map: agentId → snapshot.
// Nested Maps track what each agent's status was at the previous poll so the
// next poll can compute what changed and emit transition messages.
const statusSnapshotsByStateRoot = new Map<string, Map<string, AgentStatusSnapshot>>();
// `string | undefined` union: either the text of the last status-bar line we
// rendered, or undefined if nothing has been rendered yet (forces a write on
// the very first render even if the line happens to be "").
let lastRenderedStatusLine: string | undefined;

// ─── Tiny utility functions ───────────────────────────────────────────────────

// Returns the current time as an ISO-8601 string, e.g. "2024-01-15T12:34:56.789Z".
function nowIso() {
	return new Date().toISOString();
}

// Returns a Promise that resolves after `ms` milliseconds (non-blocking pause).
// Promise<void> — the Promise produces no useful value, we just wait for it.
// `new Promise((resolve) => …)` is the low-level way to create a Promise:
//   - the function you pass receives a `resolve` callback
//   - calling resolve() is what makes the Promise "done"
//   - setTimeout schedules resolve() to be called after `ms` ms
// Callers write: `await sleep(300);` to pause for 300 ms without blocking.
function sleep(ms: number): Promise<void> {
	return new Promise((resolveNow) => setTimeout(resolveNow, ms));
}

// Converts any thrown value to a plain string for safe display/logging.
// `err: unknown` — TypeScript types caught errors as `unknown` because you can
// throw anything (a string, a number, an object …), not just Error instances.
// We must check the type with `instanceof` before accessing `.message`.
// `instanceof Error` — checks whether `err` is an Error object at runtime.
function stringifyError(err: unknown): string {
	if (err instanceof Error) return err.message;
	return String(err); // fallback: coerce anything else to a string
}

// Wraps a string in single-quotes and escapes any embedded single-quotes so
// it is safe to paste into a shell command string.
function shellQuote(value: string): string {
	return `'${value.replace(/'/g, `'"'"'`)}'`;
}

// Creates a fresh, empty registry object with the correct version.
// `{}` here is an empty object literal — it satisfies `Record<string, AgentRecord>`
// because it has no keys yet (which is valid: zero entries is a valid dictionary).
function emptyRegistry(): RegistryFile {
	return {
		version: REGISTRY_VERSION,
		agents: {}, // Record<string, AgentRecord> — empty dictionary, no agents yet
	};
}

// Returns true if the status means the agent will never change state again.
function isTerminalStatus(status: AgentStatus): boolean {
	return status === "done" || status === "failed" || status === "crashed";
}

// ─── Backlog / output capture constants ──────────────────────────────────────
// Prefix injected into backlog.log lines that came from the kickoff prompt.
const PROMPT_LOG_PREFIX = "[side-agent][prompt]";
// Characters kept when summarising a task for display or tool responses.
const TASK_PREVIEW_MAX_CHARS = 220;
// Per-line and total character budgets applied when sanitising backlog output
// before returning it to the LLM (prevents prompt-injection / token explosion).
const BACKLOG_LINE_MAX_CHARS = 240;
const BACKLOG_TOTAL_MAX_CHARS = 2400;
// How many lines of tmux scrollback to capture when reading the backlog.
const TMUX_BACKLOG_CAPTURE_LINES = 300;
// Regex that matches "separator" lines (rows of dashes/equals) we skip in output.
const BACKLOG_SEPARATOR_RE = /^[-─—_=]{5,}$/u;
// Regexes for stripping ANSI escape sequences and stray control characters
// so raw terminal output is safe to embed in JSON / show to the LLM.
const ANSI_CSI_RE = /\x1b\[[0-?]*[ -/]*[@-~]/g;  // CSI sequences (colours, cursor moves …)
const ANSI_OSC_RE = /\x1b\][^\x07]*(?:\x07|\x1b\\)/g; // OSC sequences (window titles …)
const CONTROL_RE = /[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/g; // remaining non-printable bytes
// Limits applied to the LLM-generated context summary.
const SUMMARY_MAX_LINES = 10;
const SUMMARY_MAX_CHARS = 700;
// Matches the "no relevant context" family of LLM responses so we can treat them
// as "no summary" and fall back to sending just the raw task.
const SUMMARY_NONE_RE = /^(?:none|n\/a|no relevant context(?: from parent session)?\.?|unrelated)\s*$/i;

// ─── Backlog path helpers ─────────────────────────────────────────────────────

// Resolves the backlog log file path for a record, using stored paths when
// available and falling back to the canonical runtime-dir location.
function resolveBacklogPathForRecord(stateRoot: string, record: AgentRecord): string {
	if (record.logPath) return record.logPath;
	if (record.runtimeDir) return join(record.runtimeDir, "backlog.log");
	return join(getRuntimeDir(stateRoot, record.id), "backlog.log");
}

// Appends a structured log of the kickoff prompt to the agent's backlog.log.
// Each line is prefixed with a timestamp and the agent ID so it is traceable.
// This is best-effort; if the write fails, agent startup continues anyway.
async function appendKickoffPromptToBacklog(
	stateRoot: string,
	record: AgentRecord,
	prompt: string,
	loggedAt = nowIso(),
): Promise<void> {
	const backlogPath = resolveBacklogPathForRecord(stateRoot, record);
	const promptLines = prompt.replace(/\r\n?/g, "\n").split("\n");
	const body = promptLines
		.map((line) => `${PROMPT_LOG_PREFIX} ${loggedAt} ${record.id}: ${line}`)
		.join("\n");
	const payload =
		`${PROMPT_LOG_PREFIX} ${loggedAt} ${record.id}: kickoff prompt begin\n` +
		`${body}\n` +
		`${PROMPT_LOG_PREFIX} ${loggedAt} ${record.id}: kickoff prompt end\n`;

	try {
		await ensureDir(dirname(backlogPath));
		await fs.appendFile(backlogPath, payload, "utf8");
		// Cache the resolved paths back onto the record so future calls are cheaper.
		record.logPath = record.logPath ?? backlogPath;
		record.runtimeDir = record.runtimeDir ?? dirname(backlogPath);
	} catch {
		// Best effort only; prompt logging must not block agent startup.
	}
}

// Updates the in-memory record's status and bumps updatedAt.
// Returns true if the status actually changed, false if it was already the same.
// The `_stateRoot` parameter is unused but kept for signature consistency.
async function setRecordStatus(_stateRoot: string, record: AgentRecord, nextStatus: AgentStatus): Promise<boolean> {
	const previousStatus = record.status;
	if (previousStatus === nextStatus) return false;

	record.status = nextStatus;
	record.updatedAt = nowIso();
	return true;
}

// ─── Status display helpers ───────────────────────────────────────────────────

// Returns a short 4-5 character abbreviation of a status for the status bar.
function statusShort(status: AgentStatus): string {
	switch (status) {
		case "allocating_worktree":
			return "alloc";
		case "spawning_tmux":
			return "tmux";
		case "running":
			return "run";
		case "waiting_user":
			return "wait";
		case "done":
			return "done";
		case "failed":
			return "fail";
		case "crashed":
			return "crash";
	}
}

// Maps each status to a TUI colour role so the status bar uses consistent colours:
// warning = yellow (transient startup states), muted = grey (normal/done),
// accent = blue (needs attention), error = red (failures).
function statusColorRole(status: AgentStatus): "warning" | "muted" | "accent" | "error" {
	switch (status) {
		// Rare/transient states: highlight so they stand out.
		case "allocating_worktree":
		case "spawning_tmux":
			return "warning";
		// Normal working states: keep low visual weight.
		case "running":
		case "done":
			return "muted";
		// Needs user attention.
		case "waiting_user":
			return "accent";
		// Terminal failure.
		case "failed":
		case "crashed":
			return "error";
	}
}

// ─── Text sanitisation helpers ────────────────────────────────────────────────

// Strips all ANSI escape sequences and stray control characters from `text`,
// returning a plain printable string safe for JSON / LLM consumption.
function stripTerminalNoise(text: string): string {
	return text.replace(ANSI_CSI_RE, "").replace(ANSI_OSC_RE, "").replace(/\r/g, "").replace(CONTROL_RE, "");
}

// Truncates `text` to at most `maxChars` characters, appending "…" if cut.
function truncateWithEllipsis(text: string, maxChars: number): string {
	if (maxChars <= 0) return "";
	if (text.length <= maxChars) return text;
	if (maxChars === 1) return "…";
	return `${text.slice(0, maxChars - 1)}…`;
}

// Cleans up a raw LLM summary response:
// - strips ANSI noise and surrounding whitespace
// - unwraps markdown code fences (```…```)
// - collapses consecutive blank lines
// - enforces SUMMARY_MAX_LINES / SUMMARY_MAX_CHARS limits
// - returns "" if the LLM effectively said "no relevant context"
function normalizeGeneratedSummary(raw: string): string {
	const cleaned = stripTerminalNoise(raw).trim();
	if (!cleaned) return "";

	// If the LLM wrapped its answer in a code fence, extract the inner text.
	const fenced = cleaned.match(/^```(?:markdown|md)?\s*\n?([\s\S]*?)\n?```$/i);
	const unfenced = (fenced ? fenced[1] : cleaned).trim();
	if (!unfenced) return "";
	if (SUMMARY_NONE_RE.test(unfenced)) return "";

	// Collapse consecutive blank lines and cap at SUMMARY_MAX_LINES.
	const compactLines: string[] = [];
	let previousBlank = false;
	for (const rawLine of unfenced.replace(/\r\n?/g, "\n").split("\n")) {
		const line = rawLine.trimEnd();
		const blank = line.trim().length === 0;
		if (blank) {
			if (previousBlank) continue; // skip second+ consecutive blank line
			previousBlank = true;
		} else {
			previousBlank = false;
		}
		compactLines.push(line);
		if (compactLines.length >= SUMMARY_MAX_LINES) break;
	}

	const summary = compactLines.join("\n").trim();
	if (!summary || SUMMARY_NONE_RE.test(summary)) return "";
	return truncateWithEllipsis(summary, SUMMARY_MAX_CHARS);
}

// Collapses whitespace in `task` and truncates it for display / tool responses.
function summarizeTask(task: string): string {
	const collapsed = stripTerminalNoise(task).replace(/\s+/g, " ").trim();
	return truncateWithEllipsis(collapsed, TASK_PREVIEW_MAX_CHARS);
}

// Returns true if a line is purely decorative (a row of dashes or equals signs).
function isBacklogSeparatorLine(line: string): boolean {
	return BACKLOG_SEPARATOR_RE.test(line.trim());
}

// Splits text on line endings, dropping only a trailing empty string that
// results from a final newline (preserves intentional blank lines in the middle).
function splitLines(text: string): string[] {
	return text
		.split(/\r?\n/)
		.filter((line, i, arr) => !(i === arr.length - 1 && line.length === 0));
}

// Walks `lines` backwards and collects up to `minimumLines` non-blank,
// non-separator lines, then reverses the result back to chronological order.
function collectRecentBacklogLines(lines: string[], minimumLines: number): string[] {
	if (minimumLines <= 0) return [];

	const selected: string[] = [];
	for (let i = lines.length - 1; i >= 0; i -= 1) {
		const cleaned = stripTerminalNoise(lines[i]).trimEnd();
		if (cleaned.length === 0) continue;
		if (isBacklogSeparatorLine(cleaned)) continue;
		selected.push(lines[i]);
		if (selected.length >= minimumLines) break;
	}

	return selected.reverse();
}

// Splits `text` into lines and returns the tail of at most `minimumLines`
// meaningful (non-blank, non-separator) lines.
function selectBacklogTailLines(text: string, minimumLines: number): string[] {
	return collectRecentBacklogLines(splitLines(text), minimumLines);
}

// Strips ANSI noise, drops blank/separator lines, and enforces the per-line
// and total character budgets before handing backlog content to the LLM.
//
// `string[]` — an array of strings (equivalent to Array<string>).
// `const out: string[] = []` — declares an empty array that will hold strings.
// `let remaining = BACKLOG_TOTAL_MAX_CHARS` — a mutable counter (`let` allows
// reassignment; `const` does not).
//
// `for (const raw of lines)` — for…of loop:
// ▸ Iterates every element of the `lines` array in order.
// ▸ Each iteration `raw` holds the current element.
// ▸ `const` inside a for…of is fine — it's a new binding each iteration.
function sanitizeBacklogLines(lines: string[]): string[] {
	const out: string[] = [];
	let remaining = BACKLOG_TOTAL_MAX_CHARS;

	for (const raw of lines) {
		if (remaining <= 0) break;
		const cleaned = stripTerminalNoise(raw).trimEnd();
		if (cleaned.length === 0) continue;
		if (isBacklogSeparatorLine(cleaned)) continue;

		// Apply per-line limit first, then check against the total budget.
		const line = truncateWithEllipsis(cleaned, BACKLOG_LINE_MAX_CHARS);
		if (line.length <= remaining) {
			out.push(line);
			remaining -= line.length + 1; // +1 for the newline when joined
			continue;
		}

		// Partially fill the remaining budget and stop.
		out.push(truncateWithEllipsis(line, remaining));
		remaining = 0;
		break;
	}

	return out;
}

// Parses and validates a user-supplied list of status names for agent-wait-any.
// Returns the DEFAULT_WAIT_STATES if the input is empty, or an error message
// if any status name is not in ALL_AGENT_STATUSES.
//
// `input?: string[]` — `?` means the parameter itself is optional (may be
// undefined if the caller didn't pass it at all).
// Return type `{ values: AgentStatus[]; error?: string }` — an inline object
// type: always has `values`, optionally has `error`.
function normalizeWaitStates(input?: string[]): { values: AgentStatus[]; error?: string } {
	if (!input || input.length === 0) {
		return { values: DEFAULT_WAIT_STATES };
	}

	// input.map((value) => value.trim())
	// ▸ `.map(callback)` transforms every element of the array using the callback
	//   and returns a NEW array with the results.  ["  running  "] → ["running"]
	//
	// .filter(Boolean)
	// ▸ `.filter(callback)` keeps only elements where the callback returns truthy.
	// ▸ `Boolean` is a function that returns false for "", null, undefined, 0.
	//   Here it removes empty strings left after trimming.
	//
	// new Set(…) deduplicates; [...new Set(…)] spreads the Set back into an array.
	// Spread `[...set]` = "give me all elements of this Set as an array".
	const trimmed = [...new Set(input.map((value) => value.trim()).filter(Boolean))];
	if (trimmed.length === 0) {
		return { values: DEFAULT_WAIT_STATES };
	}

	// `new Set<AgentStatus>(array)` — constructs a Set pre-populated from the
	// array, giving O(1) .has() lookups instead of O(n) .includes() on the array.
	const known = new Set<AgentStatus>(ALL_AGENT_STATUSES);
	const invalid = trimmed.filter((value) => !known.has(value as AgentStatus));
	if (invalid.length > 0) {
		return {
			values: [],
			error: `Unknown status value(s): ${invalid.join(", ")}`,
		};
	}

	return {
		values: trimmed as AgentStatus[],
	};
}

// Returns the last `count` non-empty lines from `text`.
function tailLines(text: string, count: number): string[] {
	return splitLines(text).slice(-count);
}

// ─── Shell / process helpers ──────────────────────────────────────────────────

// Runs `command` with `args` synchronously and returns a normalised result.
// Never throws – errors are returned as { ok: false, error }.
//
// `options?: { cwd?: string; input?: string }`
// ▸ The whole `options` parameter is optional (`?` after the name).
// ▸ The type is an inline anonymous object type — no need to name it separately.
// ▸ Both fields inside it are also optional.
//
// `options?.cwd`  — optional chaining:
// ▸ If `options` is undefined (caller didn't pass it), this is undefined.
// ▸ If `options` exists, this is `options.cwd` (which may itself be undefined).
// ▸ Without `?.` you'd get "Cannot read property 'cwd' of undefined".
//
// `result.stdout ?? ""`  — nullish coalescing:
// ▸ spawnSync can return null for stdout/stderr if there was no output.
// ▸ `?? ""` replaces null or undefined with an empty string.
// ▸ `||` would also replace 0 and false — `??` is safer here.
function run(command: string, args: string[], options?: { cwd?: string; input?: string }): CommandResult {
	const result = spawnSync(command, args, {
		cwd: options?.cwd,
		input: options?.input,
		encoding: "utf8",
	});

	if (result.error) {
		return {
			ok: false,
			status: result.status,
			stdout: result.stdout ?? "",
			stderr: result.stderr ?? "",
			error: result.error.message,
		};
	}

	return {
		ok: result.status === 0,
		status: result.status,
		stdout: result.stdout ?? "",
		stderr: result.stderr ?? "",
	};
}

// Like run(), but throws a descriptive Error if the command exits non-zero.
// `throw new Error(…)` — creates an Error object and throws it, which unwinds
// the call stack until a `catch` block (or a `.catch()` handler) catches it.
// Template literal: `Command failed: ${command} ${args.join(" ")} …`
// ▸ args.join(" ") converts the string array ["git","-C","…"] into "git -C …"
// ▸ The whole backtick string evaluates all ${…} placeholders inline.
// `.trim()` removes any leading/trailing whitespace from the resulting string.
function runOrThrow(command: string, args: string[], options?: { cwd?: string; input?: string }): CommandResult {
	const result = run(command, args, options);
	if (!result.ok) {
		const reason = result.error ? `error=${result.error}` : `exit=${result.status}`;
		throw new Error(`Command failed: ${command} ${args.join(" ")} (${reason})\n${result.stderr || result.stdout}`.trim());
	}
	return result;
}

// ─── Git helpers ──────────────────────────────────────────────────────────────

// Resolves the root of the git repository that contains `cwd`.
// Falls back to `cwd` itself if not inside a git repo.
function resolveGitRoot(cwd: string): string {
	const result = run("git", ["-C", cwd, "rev-parse", "--show-toplevel"]);
	if (result.ok) {
		const root = result.stdout.trim();
		if (root.length > 0) return resolve(root);
	}
	return resolve(cwd);
}

// ─── Path helpers for the shared state directory ──────────────────────────────

// Returns the "state root" – the folder that anchors .pi/side-agents/.
// Normally the git repo root; overridden by PI_SIDE_AGENTS_ROOT for tests.
function getStateRoot(ctx: ExtensionContext): string {
	const fromEnv = process.env[ENV_STATE_ROOT];
	if (fromEnv) return resolve(fromEnv);
	return resolveGitRoot(ctx.cwd);
}

// Returns the directory that holds the registry and runtime folders:
// <stateRoot>/.pi/side-agents/
function getMetaDir(stateRoot: string): string {
	return join(stateRoot, ".pi", "side-agents");
}

// Full path to registry.json.
function getRegistryPath(stateRoot: string): string {
	return join(getMetaDir(stateRoot), "registry.json");
}

// Full path to registry.lock (held for the duration of each registry mutation).
function getRegistryLockPath(stateRoot: string): string {
	return join(getMetaDir(stateRoot), "registry.lock");
}

// Returns the per-agent runtime directory: <metaDir>/runtime/<agentId>/
// This is where kickoff.md, backlog.log, exit.json and launch.sh live.
function getRuntimeDir(stateRoot: string, agentId: string): string {
	return join(getMetaDir(stateRoot), "runtime", agentId);
}

// Returns the base folder used to archive old runtime dirs before reuse:
// <metaDir>/runtime-archive/<agentId>/
function getRuntimeArchiveBaseDir(stateRoot: string, agentId: string): string {
	return join(getMetaDir(stateRoot), "runtime-archive", agentId);
}

// Generates a timestamp string safe for use in directory names
// (replaces ":" and "." which are problematic on some filesystems).
function runtimeArchiveStamp(): string {
	return nowIso().replace(/[:.]/g, "-");
}

// Ensures a clean runtime directory exists for the agent.
// If a runtime dir already exists (left over from a previous run), it is moved
// to runtime-archive/<agentId>/<timestamp>/ before a fresh dir is created.
// As a last resort, the old dir is deleted if archiving also fails.
async function prepareFreshRuntimeDir(stateRoot: string, agentId: string): Promise<PrepareRuntimeDirResult> {
	const runtimeDir = getRuntimeDir(stateRoot, agentId);
	// Happy path: no previous runtime dir – just create and return.
	if (!(await fileExists(runtimeDir))) {
		await ensureDir(runtimeDir);
		return { runtimeDir };
	}

	// Build a unique archive path that won't collide even under parallel starts.
	const archiveBaseDir = getRuntimeArchiveBaseDir(stateRoot, agentId);
	const archiveDir = join(
		archiveBaseDir,
		`${runtimeArchiveStamp()}-${process.pid}-${Math.random().toString(16).slice(2, 8)}`,
	);

	try {
		await ensureDir(archiveBaseDir);
		// Atomic rename: the old dir disappears and the new one appears in one step.
		await fs.rename(runtimeDir, archiveDir);
		await ensureDir(runtimeDir);
		return {
			runtimeDir,
			archivedRuntimeDir: archiveDir,
		};
	} catch (archiveErr) {
		// Archiving failed (e.g. cross-device rename). Try a plain delete instead.
		const archiveErrMessage = stringifyError(archiveErr);
		try {
			await fs.rm(runtimeDir, { recursive: true, force: true });
			await ensureDir(runtimeDir);
		} catch (cleanupErr) {
			throw new Error(
				`Failed to prepare runtime dir ${runtimeDir}: archive failed (${archiveErrMessage}); cleanup failed (${stringifyError(cleanupErr)})`,
			);
		}

		return {
			runtimeDir,
			warning: `Failed to archive existing runtime dir for ${agentId}: ${archiveErrMessage}. Removed stale runtime directory instead.`,
		};
	}
}

// ─── Generic filesystem helpers ───────────────────────────────────────────────

// Returns true if the path exists (file or directory), false otherwise.
async function fileExists(path: string): Promise<boolean> {
	try {
		await fs.stat(path);
		return true;
	} catch {
		return false;
	}
}

// Creates `path` and all missing parent directories, silently if they exist.
async function ensureDir(path: string): Promise<void> {
	await fs.mkdir(path, { recursive: true });
}

// Reads a JSON file and parses it. Returns undefined on any error
// (missing file, bad JSON, etc.) so callers don't have to catch.
//
// `<T>` — generic type parameter. The caller decides what shape to expect:
//   readJsonFile<RegistryFile>(path)  → returns Promise<RegistryFile | undefined>
//   readJsonFile<ExitMarker>(path)    → returns Promise<ExitMarker | undefined>
//
// Promise<T | undefined> — the Promise produces either a T value or undefined.
//
// `async` + `await`: fs.readFile is async (reads from disk without blocking).
// `await` pauses here until the file contents arrive, then assigns to `raw`.
//
// `try { … } catch { … }` — if ANYTHING inside the try block throws
// (file not found, permission denied, malformed JSON, …) we jump to catch
// and return undefined instead of crashing the caller.
//
// `JSON.parse(raw) as T` — `as T` is a type assertion: we tell TypeScript
// "trust me, the parsed object has the shape T." TypeScript cannot verify this
// at runtime; if the JSON has the wrong shape, later code will just get undefined
// or wrong values for optional fields.
async function readJsonFile<T>(path: string): Promise<T | undefined> {
	try {
		const raw = await fs.readFile(path, "utf8");
		return JSON.parse(raw) as T;
	} catch {
		return undefined;
	}
}

// Writes `content` to `path` atomically by writing to a temp file first and
// then renaming it. This prevents readers from seeing a partially-written file.
async function atomicWrite(path: string, content: string): Promise<void> {
	await ensureDir(dirname(path));
	// Include PID and random bytes in the temp name to avoid collisions.
	const tmp = `${path}.tmp.${process.pid}.${Math.random().toString(16).slice(2)}`;
	await fs.writeFile(tmp, content, "utf8");
	await fs.rename(tmp, path);
}

// Acquires an exclusive file-based lock at `lockPath`, runs `fn`, then releases
// the lock. If another process holds the lock, retries with exponential backoff
// for up to 10 seconds, with stale-lock detection (dead PID or age > 30 s).
//
// `<T>` — generic: works for any return type. Callers get back whatever `fn`
// returns, fully type-safe.  withFileLock<RegistryFile>(…) → Promise<RegistryFile>
//
// `fn: () => Promise<T>` — a callback parameter.
// ▸ `()` = takes no arguments
// ▸ `=> Promise<T>` = returns a Promise of T (so fn must be an async function)
// ▸ The caller passes their actual work as a lambda: async () => { … }
//
// `while (true) { … }` — an infinite loop. The only way out is:
//   `return await fn()` on success, or `throw` on timeout.
async function withFileLock<T>(lockPath: string, fn: () => Promise<T>): Promise<T> {
	await ensureDir(dirname(lockPath));

	const started = Date.now();
	while (true) {
		try {
			// "wx" = create exclusively; throws EEXIST if the file already exists.
			const handle = await fs.open(lockPath, "wx");
			try {
				// Write our PID into the lock file so peers can detect stale locks.
				await handle.writeFile(JSON.stringify({ pid: process.pid, createdAt: nowIso() }) + "\n", "utf8");
			} catch {
				// best effort
			}

			try {
				return await fn();
			} finally {
				// Always release the lock, even if fn() throws.
				await handle.close().catch(() => {});
				await fs.unlink(lockPath).catch(() => {});
			}
		} catch (err: any) {
			if (err?.code !== "EEXIST") throw err; // unexpected error

			try {
				const st = await fs.stat(lockPath);
				const ageMs = Date.now() - st.mtimeMs;

				// Lock is older than 30 s – unconditionally stale.
				if (ageMs > 30_000) {
					await fs.unlink(lockPath).catch(() => {});
					continue;
				}

				// Lock is 2-30 s old – check whether the holder PID is still alive.
				if (ageMs > 2_000) {
					try {
						const raw = await fs.readFile(lockPath, "utf8");
						const data = JSON.parse(raw);
						if (typeof data.pid === "number") {
							try {
								process.kill(data.pid, 0); // signal 0 = existence check only
							} catch {
								// PID doesn't exist → stale lock from a crashed process.
								await fs.unlink(lockPath).catch(() => {});
								continue;
							}
						}
					} catch {
						// Can't read/parse lock – fall through to normal retry/timeout.
					}
				}
			} catch {
				// ignore stat errors
			}

			if (Date.now() - started > 10_000) {
				throw new Error(`Timed out waiting for lock ${lockPath}`);
			}
			// Random jitter (40-120 ms) avoids thundering-herd when many agents start at once.
			await sleep(40 + Math.random() * 80);
		}
	}
}

// ─── Registry read / write / mutate ──────────────────────────────────────────

// Reads registry.json from disk. Returns an empty registry if the file is
// missing, unreadable, or has a schema version mismatch.
async function loadRegistry(stateRoot: string): Promise<RegistryFile> {
	const registryPath = getRegistryPath(stateRoot);
	const parsed = await readJsonFile<RegistryFile>(registryPath);
	if (!parsed || typeof parsed !== "object") return emptyRegistry();
	if (parsed.version !== REGISTRY_VERSION || typeof parsed.agents !== "object" || parsed.agents === null) {
		return emptyRegistry();
	}
	return parsed;
}

// Writes `registry` to registry.json atomically.
async function saveRegistry(stateRoot: string, registry: RegistryFile): Promise<void> {
	const registryPath = getRegistryPath(stateRoot);
	await atomicWrite(registryPath, JSON.stringify(registry, null, 2) + "\n");
}

// The canonical way to modify the registry:
// 1. Acquire the file lock so no two processes write at the same time.
// 2. Load the latest registry from disk.
// 3. Call `mutator` which can make any changes to the in-memory object.
// 4. If anything changed, write it back atomically.
// 5. Release the lock and return the (possibly updated) registry.
//
// `mutator: (registry: RegistryFile) => Promise<void> | void`
// ▸ `mutator` is a *callback parameter* — a function passed as an argument.
// ▸ `(registry: RegistryFile)` = the callback receives one argument: the registry.
// ▸ `=> Promise<void> | void` = the callback may be async (returns a Promise)
//   OR synchronous (returns nothing). The `| void` union handles both cases.
// ▸ The caller supplies the actual logic inline:
//     await mutateRegistry(stateRoot, async (registry) => {
//       registry.agents["foo"] = { … };   // mutate the object directly
//     });
//   After this call returns, the change has been persisted to disk.
async function mutateRegistry(stateRoot: string, mutator: (registry: RegistryFile) => Promise<void> | void): Promise<RegistryFile> {
	const lockPath = getRegistryLockPath(stateRoot);
	return withFileLock(lockPath, async () => {
		const registry = await loadRegistry(stateRoot);
		const before = JSON.stringify(registry);
		await mutator(registry);
		const after = JSON.stringify(registry);
		// Skip the write if nothing changed (avoids spurious disk I/O).
		if (after !== before) {
			await saveRegistry(stateRoot, registry);
		}
		return registry;
	});
}

// ─── Slug / agent ID generation ───────────────────────────────────────────────

/** Sanitize a raw string into a kebab-case slug suitable for branch names and agent IDs. */
// Lowercases, replaces non-alphanumeric runs with hyphens, strips leading/trailing
// hyphens, then keeps only the first 3 hyphen-separated words.
function sanitizeSlug(raw: string): string {
	return raw
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, "-")
		.replace(/^-+|-+$/g, "")
		.split("-")
		.filter(Boolean)
		.slice(0, 3)
		.join("-");
}

/** Turn a task description into a slug by taking the first 3 meaningful words. */
// Strips punctuation, filters common stop-words, and joins the first 3 words
// with hyphens. Used as a fallback when the LLM is unavailable or fails.
function slugFromTask(task: string): string {
	const stopWords = new Set(["a", "an", "the", "to", "in", "on", "at", "of", "for", "and", "or", "is", "it", "be", "do", "with"]);
	const words = task
		.replace(/[^a-zA-Z0-9\s]/g, " ")
		.split(/\s+/)
		.map((w) => w.toLowerCase())
		.filter((w) => w.length > 0 && !stopWords.has(w));
	const slug = words.slice(0, 3).join("-");
	return slug || "agent";
}

/** Generate a slug via LLM, falling back to heuristic extraction from task text. */
// Calls the LLM with a one-line instruction to summarise the task as a kebab slug.
// The response is sanitized; if it comes out empty or the API fails, falls back
// to slugFromTask() and includes a warning in the result.
async function generateSlug(ctx: ExtensionContext, task: string): Promise<{ slug: string; warning?: string }> {
	if (!ctx.model) {
		return { slug: slugFromTask(task), warning: "No model available for slug generation; used heuristic fallback." };
	}

	try {
		const userMessage: Message = {
			role: "user",
			content: [
				{
					type: "text",
					text: task,
				},
			],
			timestamp: Date.now(),
		};

		const apiKey = await ctx.modelRegistry.getApiKey(ctx.model);
		const response = await complete(
			ctx.model,
			{
				systemPrompt:
					"Generate a 2-3 word kebab-case slug summarizing the given task. Reply with ONLY the slug, nothing else. Examples: fix-auth-leak, add-retry-logic, update-readme",
				messages: [userMessage],
			},
			{ apiKey, maxTokens: 30 }, // 30 tokens is plenty for a short slug
		);

		// Extract the text content from the response blocks and sanitize it.
		const raw = response.content
			.filter((block): block is { type: "text"; text: string } => block.type === "text")
			.map((block) => block.text)
			.join("")
			.trim();

		const slug = sanitizeSlug(raw);
		if (slug) return { slug };

		return { slug: slugFromTask(task), warning: "LLM returned empty slug; used heuristic fallback." };
	} catch (err) {
		return {
			slug: slugFromTask(task),
			warning: `Slug generation failed: ${stringifyError(err)}. Used heuristic fallback.`,
		};
	}
}

/** Collect all agent IDs currently known in the registry or checked out as side-agent branches. */
// Also scans live git worktrees for "side-agent/*" branches so that IDs created
// by a previous process (that may have crashed before writing the registry) are
// still considered "in use" and won't be reused.
//
// `Set<string>` — a collection of unique strings. Adding the same ID twice
// has no effect (deduplication is automatic).
// `Object.keys(registry.agents)` — returns an array of all key strings from
// the agents dictionary, e.g. ["fix-auth", "update-readme"].
// `new Set<string>(array)` — constructs the Set pre-populated from an array.
function existingAgentIds(registry: RegistryFile, repoRoot: string): Set<string> {
	const ids = new Set<string>(Object.keys(registry.agents));

	const listed = run("git", ["-C", repoRoot, "worktree", "list", "--porcelain"]);
	if (listed.ok) {
		for (const line of listed.stdout.split(/\r?\n/)) {
			if (!line.startsWith("branch ")) continue;
			const branchRef = line.slice("branch ".length).trim();
			if (!branchRef || branchRef === "(detached)") continue;
			// Convert "refs/heads/side-agent/foo" → "foo"
			const branch = branchRef.startsWith("refs/heads/") ? branchRef.slice("refs/heads/".length) : branchRef;
			if (branch.startsWith("side-agent/")) {
				ids.add(branch.slice("side-agent/".length));
			}
		}
	}

	return ids;
}

/** Deduplicate a slug against existing IDs by appending -2, -3, etc. */
// If "fix-auth" is taken it tries "fix-auth-2", "fix-auth-3", and so on.
function deduplicateSlug(slug: string, existing: Set<string>): string {
	if (!existing.has(slug)) return slug;
	for (let i = 2; ; i++) {
		const candidate = `${slug}-${i}`;
		if (!existing.has(candidate)) return candidate;
	}
}

// ─── Worktree lock helpers ────────────────────────────────────────────────────
// The active.lock file inside a worktree's .pi/ folder marks it as in-use.
// It contains the agent ID, session ID, PID, and tmux window info so that
// the orphan-lock scanner can determine whether the holder is still alive.

// Creates (or overwrites) the active.lock file with the given `payload`.
async function writeWorktreeLock(worktreePath: string, payload: Record<string, unknown>): Promise<void> {
	const lockPath = join(worktreePath, ".pi", "active.lock");
	await ensureDir(dirname(lockPath));
	await atomicWrite(lockPath, JSON.stringify(payload, null, 2) + "\n");
}

// Reads the existing active.lock, merges `patch` into it, and writes it back.
// Used to add tmux window info after the window is created.
async function updateWorktreeLock(worktreePath: string, patch: Record<string, unknown>): Promise<void> {
	const lockPath = join(worktreePath, ".pi", "active.lock");
	const current = (await readJsonFile<Record<string, unknown>>(lockPath)) ?? {};
	await writeWorktreeLock(worktreePath, { ...current, ...patch });
}

// Deletes the active.lock file when the agent finishes or fails.
// Ignores errors (the file may already be gone).
async function cleanupWorktreeLockBestEffort(worktreePath?: string): Promise<void> {
	if (!worktreePath) return;
	const lockPath = join(worktreePath, ".pi", "active.lock");
	await fs.unlink(lockPath).catch(() => {});
}

// Returns the set of absolute paths of all worktrees registered with git
// for `repoRoot` (includes the main worktree and all linked worktrees).
function listRegisteredWorktrees(repoRoot: string): Set<string> {
	const result = runOrThrow("git", ["-C", repoRoot, "worktree", "list", "--porcelain"]);
	const set = new Set<string>();
	for (const line of result.stdout.split(/\r?\n/)) {
		if (line.startsWith("worktree ")) {
			set.add(resolve(line.slice("worktree ".length).trim()));
		}
	}
	return set;
}

// ─── Worktree slot discovery & orphan-lock scanning ───────────────────────────

// Describes one numbered sibling directory used as an agent worktree slot.
type WorktreeSlot = {
	index: number;  // numeric suffix, e.g. 1 for "myrepo-agent-worktree-0001"
	path: string;   // absolute path
};

// A lock file found in a worktree that has no matching registry entry.
type OrphanWorktreeLock = {
	worktreePath: string;
	lockPath: string;
	lockAgentId?: string;      // agent ID stored inside the lock file (if any)
	lockPid?: number;          // PID stored inside the lock file (if any)
	lockTmuxWindowId?: string; // tmux window ID stored inside the lock file (if any)
	blockers: string[];        // reasons the lock cannot be safely reclaimed
};

// Result of scanning all worktree slots for orphan locks.
type OrphanWorktreeLockScan = {
	reclaimable: OrphanWorktreeLock[]; // safe to delete
	blocked: OrphanWorktreeLock[];     // live pid or tmux window still present
};

// Finds all directories in the parent folder of `repoRoot` whose names match
// "<repoBaseName>-agent-worktree-NNNN" and returns them sorted by index.
async function listWorktreeSlots(repoRoot: string): Promise<WorktreeSlot[]> {
	const parent = dirname(repoRoot);
	const prefix = `${basename(repoRoot)}-agent-worktree-`;
	// Build a regex that matches exactly the expected naming pattern.
	const re = new RegExp(`^${prefix.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(\\d{4})$`);

	const entries = await fs.readdir(parent, { withFileTypes: true });
	const slots: WorktreeSlot[] = [];
	for (const entry of entries) {
		if (!entry.isDirectory()) continue;
		const match = entry.name.match(re);
		if (!match) continue;
		const index = Number(match[1]);
		if (!Number.isFinite(index)) continue;
		slots.push({
			index,
			path: join(parent, entry.name),
		});
	}
	slots.sort((a, b) => a.index - b.index);
	return slots;
}

// Safely parses a PID value from a lock file field: accepts both numbers and
// digit strings, rejects anything negative, non-integer, or non-finite.
function parseOptionalPid(value: unknown): number | undefined {
	if (typeof value === "number" && Number.isFinite(value) && Number.isInteger(value) && value > 0) {
		return value;
	}
	if (typeof value === "string" && /^\d+$/.test(value)) {
		const parsed = Number(value);
		if (Number.isFinite(parsed) && parsed > 0) return parsed;
	}
	return undefined;
}

// Returns true if `pid` is a currently running process.
// Uses signal 0 which is an existence check; EPERM means the process exists
// but we don't have permission to signal it (still counts as alive).
function isPidAlive(pid?: number): boolean {
	if (pid === undefined) return false;
	try {
		process.kill(pid, 0);
		return true;
	} catch (err: any) {
		return err?.code === "EPERM"; // process exists but we lack permission
	}
}

// Returns a human-readable description of an orphan lock for display in /agents.
function summarizeOrphanLock(lock: OrphanWorktreeLock): string {
	const details: string[] = [];
	if (lock.lockAgentId) details.push(`agent:${lock.lockAgentId}`);
	if (lock.lockTmuxWindowId) details.push(`tmux:${lock.lockTmuxWindowId}`);
	if (lock.lockPid !== undefined) details.push(`pid:${lock.lockPid}`);
	if (details.length === 0) return lock.worktreePath;
	return `${lock.worktreePath} (${details.join(" ")})`;
}

// Scans all worktree slots for lock files that have no matching registry entry.
// Classifies each orphan as reclaimable (no live pid / tmux) or blocked (still live).
async function scanOrphanWorktreeLocks(repoRoot: string, registry: RegistryFile): Promise<OrphanWorktreeLockScan> {
	const slots = await listWorktreeSlots(repoRoot);
	const reclaimable: OrphanWorktreeLock[] = [];
	const blocked: OrphanWorktreeLock[] = [];

	for (const slot of slots) {
		const lockPath = join(slot.path, ".pi", "active.lock");
		if (!(await fileExists(lockPath))) continue;

		const raw = (await readJsonFile<Record<string, unknown>>(lockPath)) ?? {};
		const lockAgentId = typeof raw.agentId === "string" ? raw.agentId : undefined;
		// If the lock's agent ID is still tracked in the registry, skip it.
		if (lockAgentId && registry.agents[lockAgentId]) {
			continue;
		}

		const lockPid = parseOptionalPid(raw.pid);
		const lockTmuxWindowId = typeof raw.tmuxWindowId === "string" ? raw.tmuxWindowId : undefined;

		// Collect reasons the lock may not be safely reclaimed.
		const blockers: string[] = [];
		if (isPidAlive(lockPid)) {
			blockers.push(`pid ${lockPid} is still alive`);
		}
		if (lockTmuxWindowId && tmuxWindowExists(lockTmuxWindowId)) {
			blockers.push(`tmux window ${lockTmuxWindowId} is active`);
		}

		const candidate: OrphanWorktreeLock = {
			worktreePath: slot.path,
			lockPath,
			lockAgentId,
			lockPid,
			lockTmuxWindowId,
			blockers,
		};

		if (blockers.length > 0) {
			blocked.push(candidate);
		} else {
			reclaimable.push(candidate);
		}
	}

	return { reclaimable, blocked };
}

// Deletes the lock files for all reclaimable orphan locks.
// Returns lists of successfully removed paths and failures.
async function reclaimOrphanWorktreeLocks(locks: OrphanWorktreeLock[]): Promise<{
	removed: string[];
	failed: Array<{ lockPath: string; error: string }>;
}> {
	const removed: string[] = [];
	const failed: Array<{ lockPath: string; error: string }> = [];

	for (const lock of locks) {
		try {
			await fs.unlink(lock.lockPath);
			removed.push(lock.lockPath);
		} catch (err: any) {
			if (err?.code === "ENOENT") continue; // already gone – that's fine
			failed.push({ lockPath: lock.lockPath, error: stringifyError(err) });
		}
	}

	return { removed, failed };
}

// ─── Pi file synchronisation ──────────────────────────────────────────────────

// Copies (as symlinks) any files from the parent repo's .pi/ folder that start
// with "side-agent-" into the child worktree's .pi/ folder.
// This is how the agent-setup skill's child-finish skill gets delivered to the worktree.
async function syncParallelAgentPiFiles(parentRepoRoot: string, worktreePath: string): Promise<void> {
	const parentPiDir = join(parentRepoRoot, ".pi");
	if (!(await fileExists(parentPiDir))) return;

	// Find all files/dirs whose name starts with "side-agent-".
	const sourceEntries = await fs.readdir(parentPiDir, { withFileTypes: true });
	const names = sourceEntries
		.filter((entry) => entry.name.startsWith("side-agent-"))
		.map((entry) => entry.name);
	if (names.length === 0) return;

	const worktreePiDir = join(worktreePath, ".pi");
	await ensureDir(worktreePiDir);

	for (const name of names) {
		const source = join(parentPiDir, name);
		const target = join(worktreePiDir, name);

		let shouldLink = true;
		try {
			const st = await fs.lstat(target);
			if (st.isSymbolicLink()) {
				// If the symlink already points to the right source, leave it alone.
				const existing = await fs.readlink(target);
				if (resolve(dirname(target), existing) === resolve(source)) {
					shouldLink = false;
				}
			}
			if (shouldLink) {
				// Remove whatever is there (old symlink or directory) before re-linking.
				await fs.rm(target, { recursive: true, force: true });
			}
		} catch {
			// Target doesn't exist yet – that's fine, we'll create the symlink below.
		}

		if (shouldLink) {
			await fs.symlink(source, target);
		}
	}
}

// ─── Worktree allocation ──────────────────────────────────────────────────────

// Finds or creates a git worktree for the new agent:
// 1. Scans existing numbered slot directories (myrepo-agent-worktree-NNNN).
// 2. Picks the first slot that is unlocked and has no local changes, OR creates
//    a new slot directory with the next available index.
// 3. Checks the worktree out at the current HEAD on a new branch "side-agent/<id>".
// 4. Writes the active.lock file and syncs .pi/ files into the worktree.
async function allocateWorktree(options: {
	repoRoot: string;
	stateRoot: string;
	agentId: string;
	parentSessionId?: string;
}): Promise<AllocateWorktreeResult> {
	// Object destructuring — pulls four named fields out of `options` into
	// individual local variables. Equivalent to:
	//   const repoRoot = options.repoRoot;
	//   const stateRoot = options.stateRoot;   … and so on.
	const { repoRoot, stateRoot, agentId, parentSessionId } = options;

	const warnings: string[] = [];
	const branch = `side-agent/${agentId}`;
	// We check out at the current HEAD so the agent starts from the same commit
	// as the parent.
	const mainHead = runOrThrow("git", ["-C", repoRoot, "rev-parse", "HEAD"]).stdout.trim();

	const registry = await loadRegistry(stateRoot);
	const slots = await listWorktreeSlots(repoRoot);
	const registered = listRegisteredWorktrees(repoRoot);

	let chosen: WorktreeSlot | undefined;
	let maxIndex = 0;

	for (const slot of slots) {
		maxIndex = Math.max(maxIndex, slot.index);
		const lockPath = join(slot.path, ".pi", "active.lock");

		// Skip locked slots (another agent is using them).
		if (await fileExists(lockPath)) {
			const lock = await readJsonFile<Record<string, unknown>>(lockPath);
			const lockAgentId = typeof lock?.agentId === "string" ? lock.agentId : undefined;
			if (!lockAgentId || !registry.agents[lockAgentId]) {
				warnings.push(`Locked worktree is not tracked in registry: ${slot.path}`);
			}
			continue;
		}

		// For registered worktrees, verify they have no uncommitted changes.
		const isRegistered = registered.has(resolve(slot.path));
		if (isRegistered) {
			const status = run("git", ["-C", slot.path, "status", "--porcelain"]);
			if (!status.ok) {
				warnings.push(`Could not inspect unlocked worktree, skipping: ${slot.path}`);
				continue;
			}
			if (status.stdout.trim().length > 0) {
				warnings.push(`Unlocked worktree has local changes, skipping: ${slot.path}`);
				continue;
			}
		} else {
			// For unregistered directories, skip if non-empty (could be user data).
			const entries = await fs.readdir(slot.path).catch(() => []);
			if (entries.length > 0) {
				warnings.push(`Unlocked slot is not a registered worktree and not empty, skipping: ${slot.path}`);
				continue;
			}
		}

		chosen = slot;
		break; // first suitable slot wins
	}

	// No suitable existing slot found – allocate a new one with the next index.
	if (!chosen) {
		const next = maxIndex + 1 || 1;
		const parent = dirname(repoRoot);
		const name = `${basename(repoRoot)}-agent-worktree-${String(next).padStart(4, "0")}`;
		chosen = { index: next, path: join(parent, name) };
	}

	const chosenPath = chosen.path;
	const chosenRegistered = registered.has(resolve(chosenPath));

	if (chosenRegistered) {
		// Reuse an existing worktree: hard-reset it to HEAD, clean untracked files,
		// and switch to the new agent branch.
		const oldBranchResult = run("git", ["-C", chosenPath, "branch", "--show-current"]);
		const oldBranch = oldBranchResult.ok ? oldBranchResult.stdout.trim() : "";

		run("git", ["-C", chosenPath, "merge", "--abort"]); // ignore errors (no merge in progress)
		runOrThrow("git", ["-C", chosenPath, "reset", "--hard", mainHead]);
		runOrThrow("git", ["-C", chosenPath, "clean", "-fd"]);
		runOrThrow("git", ["-C", chosenPath, "checkout", "-B", branch, mainHead]);

		// Attempt to delete the old branch if it is fully merged; -d (not -D) is safe.
		if (oldBranch && oldBranch !== branch) {
			run("git", ["-C", repoRoot, "branch", "-d", oldBranch]);
		}
	} else {
		// Create a brand-new worktree from scratch.
		if (await fileExists(chosenPath)) {
			const entries = await fs.readdir(chosenPath).catch(() => []);
			if (entries.length > 0) {
				throw new Error(`Cannot use worktree slot ${chosenPath}: directory exists and is not empty`);
			}
		}
		await ensureDir(dirname(chosenPath));
		runOrThrow("git", ["-C", repoRoot, "worktree", "add", "-B", branch, chosenPath, mainHead]);
	}

	// Set up the worktree's .pi/ directory.
	await ensureDir(join(chosenPath, ".pi"));
	await syncParallelAgentPiFiles(repoRoot, chosenPath);
	// Mark the slot as occupied.
	await writeWorktreeLock(chosenPath, {
		agentId,
		sessionId: parentSessionId,
		parentSessionId,
		pid: process.pid,
		branch,
		startedAt: nowIso(),
	});

	return {
		worktreePath: chosenPath,
		slotIndex: chosen.index,
		branch,
		warnings,
	};
}

// ─── Kickoff prompt construction ──────────────────────────────────────────────

// Builds the text that is written to kickoff.md and sent to the child Pi session.
// If `includeSummary` is true and a model is available, an LLM call distils
// the relevant bits of the parent's conversation history and appends them.
// Falls back to the raw task text if the LLM call fails or returns "NONE".
async function buildKickoffPrompt(ctx: ExtensionContext, task: string, includeSummary: boolean): Promise<{ prompt: string; warning?: string }> {
	const parentSession = ctx.sessionManager.getSessionFile();
	const sessionSuffix = parentSession ? `\n\nParent Pi session: ${parentSession}` : "";
	if (!includeSummary || !ctx.model) {
		// No summary requested or no model configured – just use the raw task.
		return { prompt: task + sessionSuffix };
	}

	// Grab all "message" entries from the current conversation branch.
	const branch = ctx.sessionManager.getBranch();
	const messages = branch
		// .filter(callback) keeps elements where callback returns true.
		// (entry): entry is SessionEntry & { type: "message" }
		// ▸ This is a "type guard" — a special filter signature.
		// ▸ After filtering, TypeScript *knows* every surviving entry has
		//   `type === "message"`, so `.map((entry) => entry.message)` below
		//   is safe without a cast.
		// `SessionEntry & { type: "message" }` is an intersection type:
		// ▸ The value must satisfy BOTH SessionEntry AND have type="message".
		.filter((entry): entry is SessionEntry & { type: "message" } => entry.type === "message")
		// .map(callback) transforms each element and returns a new array.
		// Here: extracts the `.message` field from each filtered entry.
		.map((entry) => entry.message);

	if (messages.length === 0) {
		return { prompt: task }; // nothing to summarise
	}

	try {
		// Convert internal messages to LLM format and serialise them as text.
		const llmMessages = convertToLlm(messages);
		const conversationText = serializeConversation(llmMessages);

		// Ask the LLM to distil only the parts relevant to the child task.
		const userMessage: Message = {
			role: "user",
			content: [
				{
					type: "text",
					text: `## Parent conversation\n\n${conversationText}\n\n## Child task\n\n${task}`,
				},
			],
			timestamp: Date.now(),
		};

		const apiKey = await ctx.modelRegistry.getApiKey(ctx.model);
		const response = await complete(
			ctx.model,
			{ systemPrompt: SUMMARY_SYSTEM_PROMPT, messages: [userMessage] },
			{ apiKey },
		);

		// Collect and normalise the LLM's text output.
		// LLM responses are arrays of "blocks" (text, images, tool calls …).
		// We want only the text blocks, extract their `.text` strings, and
		// join them into one big string separated by newlines.
		//
		// Chained array methods — each returns a new array/value:
		//   .filter(…)   → keeps only text blocks (type guard narrows the type)
		//   .map(…)      → extracts the .text string from each block
		//   .join("\n")  → concatenates all strings with "\n" between them
		const summary = normalizeGeneratedSummary(
			response.content
				.filter((block): block is { type: "text"; text: string } => block.type === "text")
				.map((block) => block.text)
				.join("\n"),
		);

		if (!summary) {
			// LLM said "NONE" or returned nothing useful – fall back.
			return { prompt: task + sessionSuffix };
		}

		// Build a structured kickoff prompt that includes the summary.
		const prompt = [
			task,
			"",
			"## Parent session",
			parentSession ? `- ${parentSession}` : "- (unknown)",
			"",
			"## Relevant parent context",
			summary,
		].join("\n");

		return { prompt };
	} catch (err) {
		return {
			prompt: task + sessionSuffix,
			warning: `Failed to generate context summary: ${stringifyError(err)}. Started child with raw task only.`,
		};
	}
}

// ─── Launch script ────────────────────────────────────────────────────────────

// Generates the bash script that the tmux window runs.
// Sequence:
//   1. Export environment variables so the child Pi inherits them.
//   2. (Optionally) run .pi/side-agent-start.sh for project-specific setup.
//   3. Launch `pi` with the kickoff prompt read from kickoff.md.
//   4. Write exit.json with the exit code and timestamp.
//   5. Wait for a keypress, then kill its own tmux window.
function buildLaunchScript(params: {
	agentId: string;
	parentSessionId?: string;
	parentRepoRoot: string;
	stateRoot: string;
	worktreePath: string;
	tmuxWindowId: string;
	promptPath: string;
	exitFile: string;
	modelSpec?: string;
	runtimeDir: string;
	kubeconfig?: string;
}): string {
	// Forward KUBECONFIG if the parent session has one set.
	const kubeconfigExport = params.kubeconfig
		? `export KUBECONFIG=${shellQuote(params.kubeconfig)}\n`
		: "";
	return `#!/usr/bin/env bash
set -euo pipefail

AGENT_ID=${shellQuote(params.agentId)}
PARENT_SESSION=${shellQuote(params.parentSessionId ?? "")}
PARENT_REPO=${shellQuote(params.parentRepoRoot)}
STATE_ROOT=${shellQuote(params.stateRoot)}
WORKTREE=${shellQuote(params.worktreePath)}
WINDOW_ID=${shellQuote(params.tmuxWindowId)}
PROMPT_FILE=${shellQuote(params.promptPath)}
EXIT_FILE=${shellQuote(params.exitFile)}
MODEL_SPEC=${shellQuote(params.modelSpec ?? "")}
RUNTIME_DIR=${shellQuote(params.runtimeDir)}
START_SCRIPT=\"$WORKTREE/.pi/side-agent-start.sh\"
CHILD_SKILLS_DIR=\"$WORKTREE/.pi/side-agent-skills\"

export ${ENV_AGENT_ID}=\"$AGENT_ID\"
export ${ENV_PARENT_SESSION}=\"$PARENT_SESSION\"
export ${ENV_PARENT_REPO}=\"$PARENT_REPO\"
export ${ENV_STATE_ROOT}=\"$STATE_ROOT\"
export ${ENV_RUNTIME_DIR}=\"$RUNTIME_DIR\"
${kubeconfigExport}
write_exit() {
  local code="$1"
  printf '{"exitCode":%d,"finishedAt":"%s"}\n' "$code" "$(date -Is)" > "$EXIT_FILE"
}

cd "$WORKTREE"

if [[ -x "$START_SCRIPT" ]]; then
  set +e
  "$START_SCRIPT" "$PARENT_REPO" "$WORKTREE" "$AGENT_ID"
  start_exit=$?
  set -e
  if [[ "$start_exit" -ne 0 ]]; then
    echo "[side-agent] start script failed with code $start_exit"
    write_exit "$start_exit"
    read -n 1 -s -r -p "[side-agent] Press any key to close this tmux window..." || true
    echo
    tmux kill-window -t "$WINDOW_ID" || true
    exit "$start_exit"
  fi
fi

PI_CMD=(pi)
if [[ -n "$MODEL_SPEC" ]]; then
  PI_CMD+=(--model "$MODEL_SPEC")
fi
if [[ -d "$CHILD_SKILLS_DIR" ]]; then
  # agent-setup writes the child-only finish skill here; load it explicitly.
  PI_CMD+=(--skill "$CHILD_SKILLS_DIR")
fi

set +e
"\${PI_CMD[@]}" "$(cat "$PROMPT_FILE")"
exit_code=$?
set -e

write_exit "$exit_code"

if [[ "$exit_code" -eq 0 ]]; then
  echo "[side-agent] Agent finished."
else
  echo "[side-agent] Agent exited with code $exit_code."
fi

read -n 1 -s -r -p "[side-agent] Press any key to close this tmux window..." || true
echo

tmux kill-window -t "$WINDOW_ID" || true
`;
}

// ─── Tmux helpers ─────────────────────────────────────────────────────────────

// Throws if tmux is not installed or if we are not currently inside a tmux session.
function ensureTmuxReady(): void {
	const version = run("tmux", ["-V"]);
	if (!version.ok) {
		throw new Error("tmux is required for /agent but was not found or is not working");
	}

	const session = run("tmux", ["display-message", "-p", "#S"]);
	if (!session.ok) {
		throw new Error("/agent must be run from inside tmux (current tmux session was not detected)");
	}
}

// Returns the name of the tmux session the current process is attached to.
function getCurrentTmuxSession(): string {
	const result = runOrThrow("tmux", ["display-message", "-p", "#S"]);
	const value = result.stdout.trim();
	if (!value) throw new Error("Failed to determine current tmux session");
	return value;
}

// Creates a new tmux window (in the background, so it doesn't steal focus)
// and returns its stable window ID and numeric index.
function createTmuxWindow(tmuxSession: string, name: string): { windowId: string; windowIndex: number } {
	const result = runOrThrow("tmux", [
		"new-window",
		"-d",           // detached – don't switch to the new window
		"-t",
		`${tmuxSession}:`,
		"-P",           // print info about the new window
		"-F",
		"#{window_id} #{window_index}", // format: "@5 3"
		"-n",
		name,
	]);
	const out = result.stdout.trim();
	const [windowId, indexRaw] = out.split(/\s+/);
	const windowIndex = Number(indexRaw);
	if (!windowId || !Number.isFinite(windowIndex)) {
		throw new Error(`Unable to parse tmux window identity: ${out}`);
	}
	return { windowId, windowIndex };
}

// Returns true if the tmux window identified by `windowId` (@N) still exists.
function tmuxWindowExists(windowId: string): boolean {
	const result = run("tmux", ["display-message", "-p", "-t", windowId, "#{window_id}"]);
	return result.ok && result.stdout.trim() === windowId;
}

// Redirects all output from the tmux pane to `logPath` using tmux pipe-pane.
// "-o" appends instead of overwriting if something is already piped.
function tmuxPipePaneToFile(windowId: string, logPath: string): void {
	runOrThrow("tmux", ["pipe-pane", "-t", windowId, "-o", `cat >> ${shellQuote(logPath)}`]);
}

// Types `line` followed by Enter into the tmux pane (simulates keyboard input).
function tmuxSendLine(windowId: string, line: string): void {
	runOrThrow("tmux", ["send-keys", "-t", windowId, line, "C-m"]);
}

// Sends Ctrl+C to the tmux pane (interrupt signal).
function tmuxInterrupt(windowId: string): void {
	run("tmux", ["send-keys", "-t", windowId, "C-c"]);
}

// Pastes `prompt` into the tmux pane via the tmux buffer (handles multi-line
// text safely without shell escaping issues) and presses Enter.
function tmuxSendPrompt(windowId: string, prompt: string): void {
	// Load the text into tmux's paste buffer via stdin.
	const loaded = run("tmux", ["load-buffer", "-"], { input: prompt });
	if (!loaded.ok) {
		throw new Error(`Failed to send input to tmux window ${windowId}: ${loaded.stderr || loaded.error || "unknown error"}`);
	}
	// Paste the buffer into the pane, then press Enter.
	runOrThrow("tmux", ["paste-buffer", "-d", "-t", windowId]);
	runOrThrow("tmux", ["send-keys", "-t", windowId, "C-m"]);
}

// Captures the last TMUX_BACKLOG_CAPTURE_LINES lines of scrollback from the
// pane and returns the last `lines` of them as plain strings.
function tmuxCaptureTail(windowId: string, lines = 10): string[] {
	const captured = run("tmux", ["capture-pane", "-p", "-t", windowId, "-S", `-${TMUX_BACKLOG_CAPTURE_LINES}`]);
	if (!captured.ok) return [];
	return tailLines(captured.stdout, lines);
}

/** Capture the currently visible tmux pane content (no scrollback). */
// Returns only what is rendered on screen right now (no history buffer).
function tmuxCaptureVisible(windowId: string): string[] {
	const captured = run("tmux", ["capture-pane", "-p", "-t", windowId]);
	if (!captured.ok) return [];
	return splitLines(captured.stdout);
}

// ─── Runtime refresh (status polling) ────────────────────────────────────────

type RefreshRuntimeResult = {
	removeFromRegistry: boolean; // true when the agent exited successfully (code 0)
};

// Inspects a single agent's runtime state and updates the record in-place.
// Checks (in order):
//   1. Already "done"? → clean up lock and remove from registry.
//   2. Exit marker file exists? → set status to done/failed.
//   3. tmux window still alive? → set status to "running" if still in early phase.
//   4. tmux window gone without an exit marker? → "crashed".
async function refreshOneAgentRuntime(stateRoot: string, record: AgentRecord): Promise<RefreshRuntimeResult> {
	if (record.status === "done") {
		await cleanupWorktreeLockBestEffort(record.worktreePath);
		return { removeFromRegistry: true };
	}

	// Check whether the launch script wrote exit.json.
	if (record.exitFile && (await fileExists(record.exitFile))) {
		const exit = (await readJsonFile<ExitMarker>(record.exitFile)) ?? {};
		if (typeof exit.exitCode === "number") {
			record.exitCode = exit.exitCode;
			record.finishedAt = exit.finishedAt ?? record.finishedAt ?? nowIso();
			const changed = await setRecordStatus(stateRoot, record, exit.exitCode === 0 ? "done" : "failed");
			if (!changed) {
				record.updatedAt = nowIso();
			}
			await cleanupWorktreeLockBestEffort(record.worktreePath);
			if (exit.exitCode === 0) {
				return { removeFromRegistry: true }; // successful agents are pruned
			}
			return { removeFromRegistry: false }; // failed agents stay for inspection
		}
	}

	if (!record.tmuxWindowId) {
		return { removeFromRegistry: false };
	}

	const live = tmuxWindowExists(record.tmuxWindowId);
	if (live) {
		// Promote from early startup statuses to "running" once the window exists.
		if (record.status === "allocating_worktree" || record.status === "spawning_tmux") {
			await setRecordStatus(stateRoot, record, "running");
		}
		return { removeFromRegistry: false };
	}

	// tmux window is gone but no exit marker was written → the process crashed.
	if (!isTerminalStatus(record.status)) {
		record.finishedAt = record.finishedAt ?? nowIso();
		await setRecordStatus(stateRoot, record, "crashed");
		if (!record.error) {
			record.error = "tmux window disappeared before an exit marker was recorded";
		}
		await cleanupWorktreeLockBestEffort(record.worktreePath);
	}

	return { removeFromRegistry: false };
}

// Refreshes a single agent by ID:
// mutates the registry record in place, removes it if done, and returns
// a snapshot of the (updated) record, or undefined if the agent is unknown/removed.
async function refreshAgent(stateRoot: string, agentId: string): Promise<AgentRecord | undefined> {
	let snapshot: AgentRecord | undefined;
	await mutateRegistry(stateRoot, async (registry) => {
		const record = registry.agents[agentId];
		if (!record) return;
		const refreshed = await refreshOneAgentRuntime(stateRoot, record);
		if (refreshed.removeFromRegistry) {
			delete registry.agents[agentId];
			return;
		}
		// Deep-copy so the returned snapshot is isolated from future mutations.
		snapshot = JSON.parse(JSON.stringify(record)) as AgentRecord;
	});
	return snapshot;
}

// Refreshes all agents in the registry in one pass and returns the updated registry.
async function refreshAllAgents(stateRoot: string): Promise<RegistryFile> {
	// mutateRegistry takes an async callback. We pass one with `async (registry) => { … }`.
	// The callback receives the loaded registry, mutates it, and mutateRegistry
	// saves it if anything changed, then returns the final registry object.
	return mutateRegistry(stateRoot, async (registry) => {
		// Object.entries(obj) → array of [key, value] pairs.
		// for (const [agentId, record] of …) — array destructuring in a for…of:
		//   each iteration unpacks the [key, value] pair into two variables.
		for (const [agentId, record] of Object.entries(registry.agents)) {
			const refreshed = await refreshOneAgentRuntime(stateRoot, record);
			if (refreshed.removeFromRegistry) {
				delete registry.agents[agentId];
			}
		}
	});
}

// Returns up to `lines` recent lines of output for an agent:
// 1. Prefers the live tmux pane (avoids TUI footer noise in the log file).
// 2. Falls back to backlog.log when the window is gone.
async function getBacklogTail(record: AgentRecord, lines = 10): Promise<string[]> {
	// Prefer the visible tmux pane — it shows what's actually on screen
	// and avoids noise from TUI footer redraws that pollute the backlog file.
	if (record.tmuxWindowId && tmuxWindowExists(record.tmuxWindowId)) {
		const visible = tmuxCaptureVisible(record.tmuxWindowId);
		const result = sanitizeBacklogLines(collectRecentBacklogLines(visible, lines));
		if (result.length > 0) return result;
	}

	// Fall back to the backlog log file (e.g. tmux window gone but file remains).
	if (record.logPath && (await fileExists(record.logPath))) {
		try {
			const raw = await fs.readFile(record.logPath, "utf8");
			const tailed = sanitizeBacklogLines(selectBacklogTailLines(raw, lines));
			if (tailed.length > 0) return tailed;
		} catch {
			// fall through
		}
	}

	return [];
}

// ─── UI helpers ───────────────────────────────────────────────────────────────

// Sends a formatted message to the Pi chat (when in TUI mode) or prints to
// stdout (when running headless, e.g. in tests).
function renderInfoMessage(pi: ExtensionAPI, ctx: ExtensionContext, title: string, lines: string[]): void {
	const content = [title, "", ...lines].join("\n");
	if (ctx.hasUI) {
		pi.sendMessage({
			customType: "side-agents-report",
			content,
			display: true,
		});
	} else {
		console.log(content);
	}
}

// ─── Command argument parsing ─────────────────────────────────────────────────

// Parses the raw argument string of /agent or alias commands.
// Extracts an optional "-model <spec>" flag and returns the remainder as `task`.
function parseAgentCommandArgs(raw: string): { task: string; model?: string } {
	let rest = raw;
	let model: string | undefined;

	const modelMatch = rest.match(/(?:^|\s)-model\s+(\S+)/);
	if (modelMatch) {
		model = modelMatch[1];
		rest = rest.replace(modelMatch[0], " "); // remove the flag from the string
	}

	return {
		task: rest.trim(),
		model,
	};
}

// ─── Global settings / aliases ────────────────────────────────────────────────

// Shape of one alias entry in ~/.pi/agent/settings.json.
type AliasConfig = { model?: string };

// Returns the path to the global Pi agent settings file.
function getGlobalSettingsPath(): string {
	return join(process.env.HOME ?? "~", ".pi", "agent", "settings.json");
}

// Reads alias definitions from ~/.pi/agent/settings.json.
// Returns an empty object if the file is missing or has no "aliases" field.
function loadAliases(): Record<string, AliasConfig> {
	try {
		const raw = readFileSync(getGlobalSettingsPath(), "utf-8");
		const settings = JSON.parse(raw);
		if (settings && typeof settings.aliases === "object" && settings.aliases !== null) {
			return settings.aliases as Record<string, AliasConfig>;
		}
	} catch {
		// settings.json missing or no aliases field — that's fine
	}
	return {};
}

// ─── Model resolution ─────────────────────────────────────────────────────────

// Valid thinking-budget suffixes that can be appended to a model spec with a colon.
const THINKING_LEVELS = new Set(["off", "minimal", "low", "medium", "high", "xhigh"]);

// Splits "modelId:thinkingLevel" into its two parts.
// If the part after the last colon is not a known thinking level, the whole
// string is treated as the pattern (no thinking suffix).
function splitModelPatternAndThinking(raw: string): { pattern: string; thinking?: string } {
	const trimmed = raw.trim();
	const colon = trimmed.lastIndexOf(":");
	if (colon <= 0 || colon === trimmed.length - 1) return { pattern: trimmed };

	const suffix = trimmed.slice(colon + 1);
	if (!THINKING_LEVELS.has(suffix)) return { pattern: trimmed };

	return {
		pattern: trimmed.slice(0, colon),
		thinking: suffix,
	};
}

// Appends ":thinkingLevel" to `modelSpec` if `thinking` is set.
function withThinking(modelSpec: string, thinking?: string): string {
	return thinking ? `${modelSpec}:${thinking}` : modelSpec;
}

// Converts the user-supplied `requested` model string into a fully-qualified
// "provider/modelId[:thinking]" string that Pi's --model flag accepts.
// Handles several cases:
//   - empty/undefined → inherit parent model
//   - already has "/" → use as-is (already qualified)
//   - bare model ID → look up in the model registry to find the provider
//   - ambiguous (multiple providers) → warn and pass through raw
async function resolveModelSpecForChild(
	ctx: ExtensionContext,
	requested?: string,
): Promise<{ modelSpec?: string; warning?: string }> {
	const currentModelSpec = ctx.model ? `${ctx.model.provider}/${ctx.model.id}` : undefined;
	if (!requested || requested.trim().length === 0) {
		return { modelSpec: currentModelSpec }; // inherit parent
	}

	const trimmed = requested.trim();
	if (trimmed.includes("/")) {
		return { modelSpec: trimmed }; // already fully qualified
	}

	const { pattern, thinking } = splitModelPatternAndThinking(trimmed);

	// Fast path: matches the currently active model ID exactly.
	if (ctx.model && pattern === ctx.model.id) {
		return {
			modelSpec: withThinking(`${ctx.model.provider}/${ctx.model.id}`, thinking),
		};
	}

	try {
		const available = (await ctx.modelRegistry.getAvailable()) as Array<{ provider: string; id: string }>;
		const exact = available.filter((model) => model.id === pattern);

		// Exactly one match → resolved unambiguously.
		if (exact.length === 1) {
			const match = exact[0];
			return {
				modelSpec: withThinking(`${match.provider}/${match.id}`, thinking),
			};
		}

		if (exact.length > 1) {
			// Multiple providers offer this model ID – prefer the same provider as
			// the parent session if possible.
			if (ctx.model) {
				const preferred = exact.find((model) => model.provider === ctx.model?.provider);
				if (preferred) {
					return {
						modelSpec: withThinking(`${preferred.provider}/${preferred.id}`, thinking),
					};
				}
			}

			// Ambiguous – pass through and warn.
			const providers = [...new Set(exact.map((model) => model.provider))].sort();
			return {
				modelSpec: trimmed,
				warning: `Model '${pattern}' matches multiple providers (${providers.join(", ")}); child was started with raw pattern '${trimmed}'. Use provider/model to force a specific provider.`,
			};
		}
	} catch {
		// Best effort only; keep raw model pattern.
	}

	return { modelSpec: trimmed }; // unknown model – pass through and let Pi validate
}

// ─── Agent ID normalisation ───────────────────────────────────────────────────

// Extracts and trims the first whitespace-separated token from `raw`.
// Tool callers sometimes accidentally include trailing whitespace or extra text.
function normalizeAgentId(raw: string): string {
	const trimmed = raw.trim();
	if (!trimmed) return "";
	const firstToken = trimmed.split(/\s+/, 1)[0];
	return firstToken ?? "";
}

// ─── Core: start an agent ─────────────────────────────────────────────────────

// End-to-end orchestration of spawning a new child agent:
//   1. Validate tmux is available.
//   2. Generate / deduplicate the agent ID (slug).
//   3. Register the agent in registry.json.
//   4. Allocate a git worktree.
//   5. Prepare a fresh runtime directory.
//   6. Build the kickoff prompt (optionally with LLM-generated context).
//   7. Create a tmux window and write the launch script.
//   8. Send the `cd` + `bash launch.sh` commands into the tmux pane.
//   9. Finalise the registry record with all runtime details.
//  On any error, kills the spawned tmux window and marks the agent as "failed".
async function startAgent(pi: ExtensionAPI, ctx: ExtensionContext, params: StartAgentParams): Promise<StartAgentResult> {
	ensureTmuxReady();

	const stateRoot = getStateRoot(ctx);
	const repoRoot = resolveGitRoot(stateRoot);
	const parentSessionId = ctx.sessionManager.getSessionFile();
	const now = nowIso();

	// These are declared outside the try so they are accessible in the catch block.
	let agentId = "";
	let spawnedWindowId: string | undefined;
	let allocatedWorktreePath: string | undefined;
	let allocatedBranch: string | undefined;
	let aggregatedWarnings: string[] = [];

	try {
		await ensureDir(getMetaDir(stateRoot));

		// Step 1: determine the slug (either from branchHint or LLM generation).
		let slug: string;
		if (params.branchHint) {
			slug = sanitizeSlug(params.branchHint);
			if (!slug) slug = slugFromTask(params.task);
		} else {
			const generated = await generateSlug(ctx, params.task);
			slug = generated.slug;
			if (generated.warning) aggregatedWarnings.push(generated.warning);
		}

		// Step 2: deduplicate the slug against existing agent IDs / branches,
		// then insert a placeholder record so the slot is claimed immediately.
		await mutateRegistry(stateRoot, async (registry) => {
			const existing = existingAgentIds(registry, repoRoot);
			agentId = deduplicateSlug(slug, existing);
			registry.agents[agentId] = {
				id: agentId,
				parentSessionId,
				task: params.task,
				model: params.model,
				status: "allocating_worktree",
				startedAt: now,
				updatedAt: now,
			};
		});

		// Step 3: set up the git worktree.
		const worktree = await allocateWorktree({
			repoRoot,
			stateRoot,
			agentId,
			parentSessionId,
		});
		allocatedWorktreePath = worktree.worktreePath;
		allocatedBranch = worktree.branch;
		// Spread `[...worktree.warnings]` creates a *copy* of the warnings array
		// so that later `.push()` calls on aggregatedWarnings don't accidentally
		// mutate the original array inside the worktree result object.
		aggregatedWarnings = [...worktree.warnings];

		// Step 4: prepare a clean runtime directory (archiving any old one).
		const runtimePrep = await prepareFreshRuntimeDir(stateRoot, agentId);
		const runtimeDir = runtimePrep.runtimeDir;
		if (runtimePrep.archivedRuntimeDir) {
			aggregatedWarnings.push(`Archived existing runtime dir for ${agentId}: ${runtimePrep.archivedRuntimeDir}`);
		}
		if (runtimePrep.warning) {
			aggregatedWarnings.push(runtimePrep.warning);
		}

		// Compute the paths of all runtime files.
		const promptPath = join(runtimeDir, "kickoff.md");
		const logPath = join(runtimeDir, "backlog.log");
		const exitFile = join(runtimeDir, "exit.json");
		const launchScriptPath = join(runtimeDir, "launch.sh");
		await atomicWrite(logPath, ""); // create an empty log file now so pipe-pane can append

		// Step 5: update the registry with the worktree and runtime paths.
		await mutateRegistry(stateRoot, async (registry) => {
			const record = registry.agents[agentId];
			if (!record) return;
			record.worktreePath = worktree.worktreePath;
			record.branch = worktree.branch;
			record.runtimeDir = runtimeDir;
			record.promptPath = promptPath;
			record.logPath = logPath;
			record.exitFile = exitFile;
			await setRecordStatus(stateRoot, record, "spawning_tmux");
			record.warnings = [...(record.warnings ?? []), ...worktree.warnings];
		});

		// Step 6: build the kickoff prompt (with optional LLM context summary).
		const kickoff = await buildKickoffPrompt(ctx, params.task, params.includeSummary);
		if (kickoff.warning) aggregatedWarnings.push(kickoff.warning);

		// Write the prompt to disk so the launch script can `cat` it.
		await atomicWrite(promptPath, kickoff.prompt + "\n");

		// Also log the prompt to the backlog file for later retrieval.
		try {
			await mutateRegistry(stateRoot, async (registry) => {
				const record = registry.agents[agentId];
				if (!record) return;
				await appendKickoffPromptToBacklog(stateRoot, record, kickoff.prompt);
			});
		} catch {
			// Best effort fallback when registry lock/update fails; write directly
			// to the known backlog path without requiring registry mutation.
			await appendKickoffPromptToBacklog(
				stateRoot,
				{
					id: agentId,
					task: params.task,
					status: "spawning_tmux",
					startedAt: now,
					updatedAt: nowIso(),
					runtimeDir,
					logPath,
				},
				kickoff.prompt,
			);
		}

		// Step 7: resolve the model spec and create the tmux window.
		const resolvedModel = await resolveModelSpecForChild(ctx, params.model);
		const modelSpec = resolvedModel.modelSpec;
		if (resolvedModel.warning) aggregatedWarnings.push(resolvedModel.warning);

		const tmuxSession = getCurrentTmuxSession();
		const { windowId, windowIndex } = createTmuxWindow(tmuxSession, `agent-${agentId}`);
		spawnedWindowId = windowId;

		// Record the tmux window ID in the worktree lock so orphan scanning can
		// check whether the window is still alive.
		await updateWorktreeLock(worktree.worktreePath, {
			tmuxWindowId: windowId,
			tmuxWindowIndex: windowIndex,
		});

		// Step 8: write launch.sh, make it executable, then start it in the pane.
		const launchScript = buildLaunchScript({
			agentId,
			parentSessionId,
			parentRepoRoot: repoRoot,
			stateRoot,
			worktreePath: worktree.worktreePath,
			tmuxWindowId: windowId,
			promptPath,
			exitFile,
			modelSpec,
			runtimeDir,
			kubeconfig: process.env.KUBECONFIG,
		});
		await atomicWrite(launchScriptPath, launchScript);
		await fs.chmod(launchScriptPath, 0o755);

		// Redirect pane output to the backlog log file.
		tmuxPipePaneToFile(windowId, logPath);
		// Run cd in the interactive pane shell first so Ctrl+Z in child Pi drops
		// back to the child worktree prompt (not the parent worktree).
		tmuxSendLine(windowId, `cd ${shellQuote(worktree.worktreePath)}`);
		tmuxSendLine(windowId, `bash ${shellQuote(launchScriptPath)}`);

		// Step 9: finalise the registry record with all runtime details.
		await mutateRegistry(stateRoot, async (registry) => {
			const record = registry.agents[agentId];
			if (!record) return;
			record.tmuxSession = tmuxSession;
			record.tmuxWindowId = windowId;
			record.tmuxWindowIndex = windowIndex;
			record.worktreePath = worktree.worktreePath;
			record.branch = worktree.branch;
			record.runtimeDir = runtimeDir;
			record.promptPath = promptPath;
			record.logPath = logPath;
			record.exitFile = exitFile;
			record.model = modelSpec;
			await setRecordStatus(stateRoot, record, "running");
			// `record.warnings ?? []` — if record.warnings is undefined, use [].
			// `[...array1, ...array2]` — spread both arrays into one new array.
			// Result: a single flat array containing all old warnings followed by
			// all new ones. The original arrays are not modified.
			record.warnings = [...(record.warnings ?? []), ...aggregatedWarnings];
		});

		const started: StartAgentResult = {
			id: agentId,
			tmuxWindowId: windowId,
			tmuxWindowIndex: windowIndex,
			worktreePath: worktree.worktreePath,
			branch: worktree.branch,
			warnings: aggregatedWarnings,
			prompt: kickoff.prompt,
		};
		// Emit a hidden session entry carrying the full kickoff prompt so the
		// parent session has a record of what was sent.
		emitKickoffPromptMessage(pi, started);

		return started;
	} catch (err) {
		// Clean up: kill the tmux window if one was already created.
		if (spawnedWindowId) {
			run("tmux", ["kill-window", "-t", spawnedWindowId]);
		}

		// Mark the agent as "failed" in the registry so it shows up in /agents.
		if (agentId) {
			await mutateRegistry(stateRoot, async (registry) => {
				const record = registry.agents[agentId];
				if (!record) return;
				record.error = stringifyError(err);
				record.finishedAt = nowIso();
				const changed = await setRecordStatus(stateRoot, record, "failed");
				if (!changed) {
					record.updatedAt = nowIso();
				}
				if (allocatedWorktreePath) record.worktreePath = allocatedWorktreePath;
				if (allocatedBranch) record.branch = allocatedBranch;
				record.warnings = [...(record.warnings ?? []), ...aggregatedWarnings];
			});
		}

		throw err;
	}
}

// ─── agent-check payload builder ─────────────────────────────────────────────

// Refreshes the agent's runtime state and assembles the response object
// returned by the agent-check tool and the agent-wait-any polling loop.
// Returns { ok: false, error } when the agent ID is unknown.
async function agentCheckPayload(stateRoot: string, agentId: string): Promise<Record<string, unknown>> {
	const normalizedId = normalizeAgentId(agentId);
	if (!normalizedId) {
		return {
			ok: false,
			error: "No agent id was provided",
		};
	}

	const record = await refreshAgent(stateRoot, normalizedId);
	if (!record) {
		return {
			ok: false,
			error: `Unknown agent id: ${normalizedId}`,
		};
	}

	// Fetch the last ~10 lines of output for the LLM to inspect.
	const backlog = await getBacklogTail(record, 10);

	return {
		ok: true,
		agent: {
			id: record.id,
			status: record.status,
			tmuxWindowId: record.tmuxWindowId,
			tmuxWindowIndex: record.tmuxWindowIndex,
			worktreePath: record.worktreePath,
			branch: record.branch,
			task: summarizeTask(record.task), // truncated to TASK_PREVIEW_MAX_CHARS
			startedAt: record.startedAt,
			finishedAt: record.finishedAt,
			exitCode: record.exitCode,
			error: record.error,
			warnings: record.warnings ?? [],
		},
		backlog,
	};
}

// ─── agent-send ───────────────────────────────────────────────────────────────

// Sends a follow-up prompt to a running agent's tmux pane.
// If `prompt` starts with "!", sends Ctrl+C first to interrupt the current
// operation, then waits briefly before pasting the rest of the prompt.
// Updates the agent status back to "running" in the registry.
async function sendToAgent(stateRoot: string, agentId: string, prompt: string): Promise<{ ok: boolean; message: string }> {
	const normalizedId = normalizeAgentId(agentId);
	if (!normalizedId) {
		return { ok: false, message: "No agent id was provided" };
	}

	const record = await refreshAgent(stateRoot, normalizedId);
	if (!record) {
		return { ok: false, message: `Unknown agent id: ${normalizedId}` };
	}
	if (!record.tmuxWindowId) {
		return { ok: false, message: `Agent ${normalizedId} has no tmux window id recorded` };
	}
	if (!tmuxWindowExists(record.tmuxWindowId)) {
		return { ok: false, message: `Agent ${normalizedId} tmux window is not active` };
	}

	let payload = prompt;
	if (payload.startsWith("!")) {
		// "!" prefix → interrupt first, then send the rest.
		tmuxInterrupt(record.tmuxWindowId);
		payload = payload.slice(1).trimStart();
		if (payload.length > 0) {
			// Brief pause so Pi can finish handling the interrupt and return to an
			// interactive prompt before the follow-up text lands in the pane.
			await sleep(300);
		}
	}
	if (payload.length > 0) {
		tmuxSendPrompt(record.tmuxWindowId, payload);
	}

	// Mark the agent as running (it was probably in "waiting_user").
	await mutateRegistry(stateRoot, async (registry) => {
		const current = registry.agents[normalizedId];
		if (!current) return;
		if (!isTerminalStatus(current.status)) {
			const changed = await setRecordStatus(stateRoot, current, "running");
			if (!changed) {
				current.updatedAt = nowIso();
			}
		}
	});

	return { ok: true, message: `Sent prompt to ${normalizedId}` };
}

// ─── Child-side status updates ────────────────────────────────────────────────

// Called by the child Pi process (via pi.on events) to update its own status
// in the shared registry. Does nothing when running in the parent context
// (ENV_AGENT_ID is only set inside the launch script).
async function setChildRuntimeStatus(ctx: ExtensionContext, nextStatus: AgentStatus): Promise<void> {
	const agentId = process.env[ENV_AGENT_ID];
	if (!agentId) return; // not a child agent process

	const stateRoot = getStateRoot(ctx);
	await mutateRegistry(stateRoot, async (registry) => {
		const record = registry.agents[agentId];
		if (!record) return;
		if (isTerminalStatus(record.status)) return; // never overwrite terminal statuses

		const changed = await setRecordStatus(stateRoot, record, nextStatus);
		if (!changed) {
			record.updatedAt = nowIso();
		}
	});
}

// ─── agent-wait-any ───────────────────────────────────────────────────────────

// Polls the status of all provided agent IDs every second until one of them
// reaches a status in `waitStates` (default: waiting_user, failed, crashed).
// If an ID that was previously known disappears (auto-pruned after exit 0),
// it is reported as "done".
// Returns the agentCheckPayload() of the first matching agent.
async function waitForAny(
	stateRoot: string,
	ids: string[],
	signal?: AbortSignal,
	waitStatesInput?: string[],
): Promise<Record<string, unknown>> {
	const uniqueIds = [...new Set(ids.map((id) => normalizeAgentId(id)).filter(Boolean))];
	if (uniqueIds.length === 0) {
		return { ok: false, error: "No agent ids were provided" };
	}

	const waitStates = normalizeWaitStates(waitStatesInput);
	if (waitStates.error) {
		return { ok: false, error: waitStates.error };
	}
	// Set.has() is O(1) — much faster than array.includes() for repeated checks.
	const waitStateSet = new Set<AgentStatus>(waitStates.values);

	let firstPass = true;
	// Track which IDs we have seen at least once so we can detect auto-pruning.
	const knownIds = new Set<string>();

	// `while (true)` — infinite polling loop. Exits only via `return` or a thrown error.
	while (true) {
		// `signal?.aborted` — optional chain on an AbortSignal.
		// AbortSignal is a browser/Node standard: the caller can cancel a long
		// operation by calling controller.abort(), which sets signal.aborted = true.
		if (signal?.aborted) {
			return { ok: false, error: "agent-wait-any aborted" };
		}

		const unknownOnFirstPass: string[] = [];
		let knownCount = 0;

		for (const id of uniqueIds) {
			const checked = await agentCheckPayload(stateRoot, id);
			const ok = checked.ok === true;
			if (!ok) {
				// Agent was known before but disappeared — it exited successfully
				// and was auto-pruned from the registry. Report it as done.
				if (knownIds.has(id)) {
					return {
						ok: true,
						agent: { id, status: "done" },
						backlog: [],
					};
				}
				if (firstPass) unknownOnFirstPass.push(id);
				continue;
			}

			knownIds.add(id);
			knownCount += 1;
			// `checked.agent` is typed as `unknown` (the payload is a generic Record).
			// `as any` disables TypeScript type-checking for this expression —
			// we're saying "I know what's in here, let me access .status freely."
			// The outer `as AgentStatus | undefined` then re-applies a specific type.
			// This is necessary because the payload shape isn't statically known here.
			const status = (checked.agent as any)?.status as AgentStatus | undefined;
			if (!status) continue;
			if (waitStateSet.has(status)) {
				return checked; // this agent reached a wait state → return immediately
			}
		}

		// Fail immediately if any provided ID was unrecognised on the very first
		// poll — unknown agents will never become known, so waiting is pointless.
		if (firstPass && unknownOnFirstPass.length > 0) {
			return {
				ok: false,
				error: `Unknown agent id(s): ${unknownOnFirstPass.join(", ")}`,
			};
		}

		firstPass = false;
		await sleep(1000); // poll interval
	}
}

// ─── Child session linking ────────────────────────────────────────────────────

// Called on session_start / session_switch inside the child Pi process.
// Writes the child's own session file path into the shared registry and
// into the worktree active.lock, so the parent can open the child's session.
// Also appends a CHILD_LINK_ENTRY_TYPE entry to the child's own session so
// it is recorded exactly once.
async function ensureChildSessionLinked(pi: ExtensionAPI, ctx: ExtensionContext): Promise<void> {
	const agentId = process.env[ENV_AGENT_ID];
	if (!agentId) return; // not a child agent process

	const stateRoot = getStateRoot(ctx);
	const childSession = ctx.sessionManager.getSessionFile();
	const parentSession = process.env[ENV_PARENT_SESSION];

	await mutateRegistry(stateRoot, async (registry) => {
		const existing = registry.agents[agentId];
		if (!existing) {
			// Agent record missing (e.g. registry was cleared) – re-create a minimal one.
			registry.agents[agentId] = {
				id: agentId,
				parentSessionId: parentSession,
				childSessionId: childSession,
				task: "(child session linked without parent registry record)",
				status: "running",
				startedAt: nowIso(),
				updatedAt: nowIso(),
			};
			return;
		}

		existing.childSessionId = childSession;
		existing.parentSessionId = existing.parentSessionId ?? parentSession;
		let statusChanged = false;
		if (!isTerminalStatus(existing.status)) {
			statusChanged = await setRecordStatus(stateRoot, existing, "running");
		}
		if (!statusChanged) {
			existing.updatedAt = nowIso();
		}
	});

	// Also update the worktree lock file with the child session.
	const lockPath = join(ctx.cwd, ".pi", "active.lock");
	if (await fileExists(lockPath)) {
		const lock = (await readJsonFile<Record<string, unknown>>(lockPath)) ?? {};
		lock.sessionId = childSession;
		lock.agentId = agentId;
		await atomicWrite(lockPath, JSON.stringify(lock, null, 2) + "\n");
	}

	// Append a session entry (once) so the child session records its own identity.
	// `.some(callback)` — returns true if AT LEAST ONE element passes the test.
	// Stops as soon as it finds a match (short-circuits — doesn't scan the rest).
	// Here: "does any session entry have the CHILD_LINK_ENTRY_TYPE customType?"
	// `entry as { customType?: string }` — type assertion: we widen the type so
	// TypeScript lets us read `.customType` which isn't on the base SessionEntry type.
	const hasLinkEntry = ctx.sessionManager.getEntries().some((entry) => {
		if (entry.type !== "custom") return false;
		const customEntry = entry as { customType?: string };
		return customEntry.customType === CHILD_LINK_ENTRY_TYPE;
	});

	if (!hasLinkEntry) {
		pi.appendEntry(CHILD_LINK_ENTRY_TYPE, {
			agentId,
			parentSession,
			linkedAt: Date.now(),
		});
	}
}

// Returns true when the current Pi process is running as a child agent
// (i.e. the ENV_AGENT_ID environment variable is set by the launch script).
function isChildRuntime(): boolean {
	return Boolean(process.env[ENV_AGENT_ID]);
}

// ─── Status transition detection ─────────────────────────────────────────────

// Compares the current set of agent records against the in-memory snapshots
// from the previous poll and returns a list of status changes.
// Also synthesises a "→ done" transition for any agent that disappeared
// (auto-pruned after a clean exit) since the last snapshot.
// Stores the new snapshots for the next comparison.
function collectStatusTransitions(stateRoot: string, agents: AgentRecord[]): StatusTransitionNotice[] {
	const previous = statusSnapshotsByStateRoot.get(stateRoot);
	const next = new Map<string, AgentStatusSnapshot>();
	const transitions: StatusTransitionNotice[] = [];

	for (const record of agents) {
		const currentSnapshot: AgentStatusSnapshot = {
			status: record.status,
			tmuxWindowIndex: record.tmuxWindowIndex,
		};
		next.set(record.id, currentSnapshot);

		const previousSnapshot = previous?.get(record.id);
		// No change or first time seen → no transition to emit.
		if (!previousSnapshot || previousSnapshot.status === record.status) continue;
		transitions.push({
			id: record.id,
			fromStatus: previousSnapshot.status,
			toStatus: record.status,
			tmuxWindowIndex: record.tmuxWindowIndex ?? previousSnapshot.tmuxWindowIndex,
		});
	}

	if (previous) {
		// Detect agents that were in a non-terminal state but are now gone from
		// the registry (they exited with code 0 and were auto-removed).
		// Map.entries() returns [key, value] pairs, just like Object.entries() for plain objects.
		// Destructuring `[agentId, previousSnapshot]` unpacks each pair inline.
		for (const [agentId, previousSnapshot] of previous.entries()) {
			if (next.has(agentId)) continue; // still present
			if (isTerminalStatus(previousSnapshot.status)) continue; // already terminal
			transitions.push({
				id: agentId,
				fromStatus: previousSnapshot.status,
				toStatus: "done",
				tmuxWindowIndex: previousSnapshot.tmuxWindowIndex,
			});
		}
	}

	statusSnapshotsByStateRoot.set(stateRoot, next);
	// On the very first poll there are no previous snapshots, so nothing to compare.
	if (!previous) return [];
	return transitions.sort((a, b) => a.id.localeCompare(b.id));
}

// ─── Status formatting ────────────────────────────────────────────────────────

// A minimal theme interface used locally: applies a colour role to text.
type ThemeForeground = { fg: (role: "warning" | "muted" | "accent" | "error", text: string) => string };

// Returns the status word, coloured if a theme is provided.
function formatStatusWord(status: AgentStatus, theme?: ThemeForeground): string {
	if (!theme) return status;
	return theme.fg(statusColorRole(status), status);
}

// Returns a dim/muted label prefix (e.g. "win:", "task:"), coloured if a theme is provided.
function formatLabelPrefix(prefix: string, theme?: ThemeForeground): string {
	if (!theme) return prefix;
	return theme.fg("muted", prefix);
}

// Formats a single status transition as a human-readable string, e.g.:
// "side-agent fix-auth: running -> waiting_user (tmux #3)"
function formatStatusTransitionMessage(transition: StatusTransitionNotice, theme?: ThemeForeground): string {
	const win = transition.tmuxWindowIndex !== undefined ? ` (tmux #${transition.tmuxWindowIndex})` : "";
	const from = formatStatusWord(transition.fromStatus, theme);
	const to = formatStatusWord(transition.toStatus, theme);
	return `side-agent ${transition.id}: ${from} -> ${to}${win}`;
}

// Emits one chat message per transition and, for failure transitions, also
// shows a TUI notification badge. Does nothing in child agent processes.
function emitStatusTransitions(pi: ExtensionAPI, ctx: ExtensionContext, transitions: StatusTransitionNotice[]): void {
	if (isChildRuntime()) return; // children don't emit status messages to parents

	for (const transition of transitions) {
		const message = formatStatusTransitionMessage(transition, ctx.hasUI ? ctx.ui.theme : undefined);
		pi.sendMessage(
			{
				customType: STATUS_UPDATE_MESSAGE_TYPE,
				content: message,
				display: true,
				details: {
					agentId: transition.id,
					fromStatus: transition.fromStatus,
					toStatus: transition.toStatus,
					tmuxWindowIndex: transition.tmuxWindowIndex,
					emittedAt: Date.now(),
				},
			},
			{
				triggerTurn: false,  // don't start a new LLM turn just for this message
				deliverAs: "followUp",
			},
		);

		// Show an error badge in the TUI for hard failures.
		if (ctx.hasUI && (transition.toStatus === "failed" || transition.toStatus === "crashed")) {
			ctx.ui.notify(message, "error");
		}

		// Colorize the tmux window in the status bar to reflect the new status.
		if (transition.tmuxWindowIndex !== undefined) {
			const target = `:${transition.tmuxWindowIndex}`;
			const style =
				transition.toStatus === "waiting_user" ? "bg=yellow,fg=black" :
				transition.toStatus === "failed" || transition.toStatus === "crashed" ? "bg=red,fg=white" :
				"default";
			run("tmux", ["set-window-option", "-t", target, "window-status-style", style]);
			run("tmux", ["set-window-option", "-t", target, "window-status-current-style", style]);
		}
	}
}

// Emits a hidden session entry that carries the full kickoff prompt so the
// parent's conversation history has a record of what was sent to the child.
function emitKickoffPromptMessage(pi: ExtensionAPI, started: StartAgentResult): void {
	const win = started.tmuxWindowIndex !== undefined ? ` (tmux #${started.tmuxWindowIndex})` : "";
	const content = `side-agent ${started.id}: kickoff prompt${win}\n\n${started.prompt}`;
	pi.sendMessage(
		{
			customType: PROMPT_UPDATE_MESSAGE_TYPE,
			content,
			display: false, // hidden from the chat view; visible in session history
			details: {
				agentId: started.id,
				tmuxWindowId: started.tmuxWindowId,
				tmuxWindowIndex: started.tmuxWindowIndex,
				worktreePath: started.worktreePath,
				branch: started.branch,
				prompt: started.prompt,
				emittedAt: Date.now(),
			},
		},
		{ triggerTurn: false },
	);
}

// ─── Status bar rendering ─────────────────────────────────────────────────────

// Refreshes all agents, detects status transitions (emitting messages for each),
// and updates the TUI status bar with a compact one-line summary of all agents.
// Each agent is shown as "<id>:<shortStatus>@<tmuxWindow>", coloured by role.
// Clears the status bar when no agents remain.
async function renderStatusLine(pi: ExtensionAPI, ctx: ExtensionContext, options?: { emitTransitions?: boolean }): Promise<void> {
	if (!ctx.hasUI) return; // no UI → nothing to render

	const stateRoot = getStateRoot(ctx);
	const refreshed = await refreshAllAgents(stateRoot);
	const agents = Object.values(refreshed.agents).sort((a, b) => a.id.localeCompare(b.id));

	if (options?.emitTransitions ?? true) {
		// Normal path: detect and emit any status changes since last poll.
		const transitions = collectStatusTransitions(stateRoot, agents);
		if (transitions.length > 0) {
			emitStatusTransitions(pi, ctx, transitions);
		}
	} else if (!statusSnapshotsByStateRoot.has(stateRoot)) {
		// First call with emitTransitions=false (e.g. before_agent_start):
		// initialise the snapshot map without emitting anything.
		collectStatusTransitions(stateRoot, agents);
	}

	if (agents.length === 0) {
		// No active agents → remove the status bar entry.
		if (lastRenderedStatusLine !== undefined) {
			ctx.ui.setStatus(STATUS_KEY, undefined);
			lastRenderedStatusLine = undefined;
		}
		return;
	}

	// Build the status bar string: "agent1:run@2 agent2:wait@5"
	const theme = ctx.ui.theme;
	const line = visible
		.map((record) => {
			const win = record.tmuxWindowIndex !== undefined ? `@${record.tmuxWindowIndex}` : "";
			const entry = `${record.id}:${statusShort(record.status)}${win}`;
			return theme.fg(statusColorRole(record.status), entry);
		})
		.join(" ");

	// Skip the write if the line hasn't changed (avoids flickering).
	if (line === lastRenderedStatusLine) return;
	ctx.ui.setStatus(STATUS_KEY, line);
	lastRenderedStatusLine = line;
}

// Starts the background interval that keeps the status bar up to date (every 2.5 s).
// Also triggers an immediate render on first call.
// Stores the latest ctx/pi references so the interval callback always uses current values.
function ensureStatusPoller(pi: ExtensionAPI, ctx: ExtensionContext): void {
	statusPollContext = ctx;
	statusPollApi = pi;
	if (!ctx.hasUI) return;

	if (!statusPollTimer) {
		// setInterval(callback, ms) — calls `callback` every `ms` milliseconds.
		// Returns a NodeJS.Timeout handle we store in statusPollTimer.
		statusPollTimer = setInterval(() => {
			// Skip if a previous render is still in flight (avoids queuing up renders).
			if (statusPollInFlight || !statusPollContext || !statusPollApi) return;
			statusPollInFlight = true;
			// `void` before the call means "run this async function but don't
			// await it and don't care about its return value."
			// `.catch(() => {})` — swallows any error so the interval never dies.
			// `.finally(() => { … })` — runs after success OR error, resetting the flag.
			void renderStatusLine(statusPollApi, statusPollContext)
				.catch(() => {})
				.finally(() => {
					statusPollInFlight = false;
				});
		}, 2500);
		// unref() tells Node.js: "don't keep the process alive just for this timer."
		// Without it, if Pi tries to exit, the interval would prevent shutdown.
		statusPollTimer.unref();
	}

	// Render immediately instead of waiting for the first interval tick.
	// Again `void` + `.catch(() => {})` = fire-and-forget, ignore errors.
	void renderStatusLine(pi, ctx).catch(() => {});
}

// ─── Extension entry point ────────────────────────────────────────────────────

// `export default` makes this function the module's single public export.
// When Pi loads the extension file it does:
//   const ext = await import("./side-agents.js");
//   ext.default(pi);    ← calls this function with the live API object
// Everything registered inside this function (commands, tools, listeners) is
// active for the lifetime of the Pi process.
export default function sideAgentsExtension(pi: ExtensionAPI) {

	// Child sessions (PI_SIDE_AGENT_ID is set) must not be able to spawn further
	// agents — only the root parent session registers spawn-related tools/commands.
	const isChild = !!process.env[ENV_AGENT_ID];

	// /agent [-model <spec>] <task>
	// Spawns a new background child agent for the given task.
	// Not registered in child sessions to enforce a single level of parallelism.
	if (!isChild) pi.registerCommand("agent", {
		description: "Spawn a background child agent in its own tmux window/worktree: /agent [-model <provider/id>] <task>",
		handler: async (args, ctx) => {
			const parsed = parseAgentCommandArgs(args);
			if (!parsed.task) {
				ctx.hasUI && ctx.ui.notify("Usage: /agent [-model <provider/id>] <task>", "error");
				return;
			}

			try {
				ctx.hasUI && ctx.ui.notify("Starting side-agent…", "info");
				const started = await startAgent(pi, ctx, {
					task: parsed.task,
					model: parsed.model,
					includeSummary: true, // /agent always tries to distil parent context
				});

				// Build the info lines shown in chat after a successful start.
				const lines = [
					`id: ${started.id}`,
					`tmux window: ${started.tmuxWindowId} (#${started.tmuxWindowIndex})`,
					`worktree: ${started.worktreePath}`,
					`branch: ${started.branch}`,
				];
				for (const warning of started.warnings) {
					lines.push(`warning: ${warning}`);
				}
				lines.push("", "prompt:");
				for (const line of started.prompt.split(/\r?\n/)) {
					lines.push(`  ${line}`);
				}
				renderInfoMessage(pi, ctx, "side-agent started", lines);
				await renderStatusLine(pi, ctx).catch(() => {});
			} catch (err) {
				ctx.hasUI && ctx.ui.notify(`Failed to start agent: ${stringifyError(err)}`, "error");
			}
		},
	}); // end /agent

	// /agents
	// Lists all tracked agents with their status, worktree, and task preview.
	// Offers to clean up failed/crashed agents and reclaim orphan worktree locks.
	pi.registerCommand("agents", {
		description: "List tracked side agents",
		handler: async (_args, ctx) => {
			const stateRoot = getStateRoot(ctx);
			const repoRoot = resolveGitRoot(stateRoot);
			let registry = await refreshAllAgents(stateRoot);
			const records = Object.values(registry.agents).sort((a, b) => a.id.localeCompare(b.id));
			let orphanLocks = await scanOrphanWorktreeLocks(repoRoot, registry);

			if (records.length === 0 && orphanLocks.reclaimable.length === 0 && orphanLocks.blocked.length === 0) {
				ctx.hasUI && ctx.ui.notify("No tracked side agents yet.", "info");
				return;
			}

			const lines: string[] = [];
			const failedIds: string[] = [];

			if (records.length === 0) {
				lines.push("(no tracked agents)");
			} else {
				const theme = ctx.hasUI ? ctx.ui.theme : undefined;
				// Array.entries() returns [index, value] pairs.
				// Destructuring gives us `index` (0, 1, 2 …) and `record` simultaneously.
				// Used below to check `index < records.length - 1` (insert blank between agents).
				for (const [index, record] of records.entries()) {
					const win = record.tmuxWindowIndex !== undefined ? `#${record.tmuxWindowIndex}` : "-";
					const worktreeName = record.worktreePath ? basename(record.worktreePath) || record.worktreePath : "-";
					const statusWord = formatStatusWord(record.status, theme);
					const winPrefix = formatLabelPrefix("win:", theme);
					const worktreePrefix = formatLabelPrefix("worktree:", theme);
					const taskPrefix = formatLabelPrefix("task:", theme);
					lines.push(`${record.id}  ${statusWord}  ${winPrefix}${win}  ${worktreePrefix}${worktreeName}`);
					lines.push(`  ${taskPrefix} ${summarizeTask(record.task)}`);
					if (record.error) lines.push(`  error: ${record.error}`);
					if (record.status === "failed" || record.status === "crashed") {
						failedIds.push(record.id);
					}
					if (index < records.length - 1) {
						lines.push(""); // blank line between agents
					}
				}
			}

			// Report orphan worktree locks at the bottom of the list.
			if (orphanLocks.reclaimable.length > 0 || orphanLocks.blocked.length > 0) {
				if (lines.length > 0) lines.push("");
				lines.push("orphan worktree locks:");
				for (const lock of orphanLocks.reclaimable) {
					lines.push(`  reclaimable: ${summarizeOrphanLock(lock)}`);
				}
				for (const lock of orphanLocks.blocked) {
					lines.push(`  blocked: ${summarizeOrphanLock(lock)} (${lock.blockers.join("; ")})`);
				}
			}

			renderInfoMessage(pi, ctx, "side-agents", lines);

			// Offer to remove failed/crashed agents from the registry.
			if (failedIds.length > 0 && ctx.hasUI) {
				const confirmed = await ctx.ui.confirm(
					"Clean up failed agents?",
					`Remove ${failedIds.length} failed/crashed agent(s) from registry: ${failedIds.join(", ")}`,
				);
				if (confirmed) {
					registry = await mutateRegistry(stateRoot, async (next) => {
						for (const id of failedIds) {
							delete next.agents[id];
						}
					});
					ctx.ui.notify(`Removed ${failedIds.length} agent(s): ${failedIds.join(", ")}`, "info");
				}
			}

			// Re-scan after potential cleanup so the orphan-lock offer is accurate.
			orphanLocks = await scanOrphanWorktreeLocks(repoRoot, registry);

			// Offer to delete reclaimable orphan lock files.
			if (orphanLocks.reclaimable.length > 0 && ctx.hasUI) {
				const preview = orphanLocks.reclaimable.slice(0, 6).map((lock) => `- ${summarizeOrphanLock(lock)}`);
				if (orphanLocks.reclaimable.length > preview.length) {
					preview.push(`- ... and ${orphanLocks.reclaimable.length - preview.length} more`);
				}

				const confirmed = await ctx.ui.confirm(
					"Reclaim orphan worktree locks?",
					[
						`Remove ${orphanLocks.reclaimable.length} orphan worktree lock(s)?`,
						"Only lock files with no tracked registry agent and no live pid/tmux signal are included.",
						"",
						...preview,
					].join("\n"),
								);
				if (confirmed) {
					const reclaimed = await reclaimOrphanWorktreeLocks(orphanLocks.reclaimable);
					if (reclaimed.failed.length === 0) {
						ctx.ui.notify(`Reclaimed ${reclaimed.removed.length} orphan worktree lock(s).`, "info");
					} else {
						ctx.ui.notify(
							`Reclaimed ${reclaimed.removed.length} orphan lock(s); failed ${reclaimed.failed.length}.`,
							"warning",
						);
					}
				}
			}

			if (orphanLocks.blocked.length > 0 && ctx.hasUI) {
				ctx.ui.notify(
					`Found ${orphanLocks.blocked.length} orphan lock(s) that look live; leaving them untouched.`,
					"warning",
				);
			}
		},
	});

	// /agent-list-tools
	// Prints all registered tools split into active (sent to LLM) and inactive.
	pi.registerCommand("agent-list-tools", {
		description: "List all registered tools and which are active",
		handler: async (_args, ctx) => {
			const all = pi.getAllTools();
			const activeNames = new Set(pi.getActiveTools());
			const active = all.filter((t) => activeNames.has(t.name));
			const inactive = all.filter((t) => !activeNames.has(t.name));
			const lines: string[] = [];
			lines.push(`active (${active.length}):`);
			for (const t of active) lines.push(`  ${t.name}`);
			if (inactive.length > 0) {
				lines.push(`inactive (${inactive.length}):`);
				for (const t of inactive) lines.push(`  ${t.name}`);
			}
			renderInfoMessage(pi, ctx, "tools", lines);
		},
	});

	// Register alias commands from ~/.pi/agent/settings.json.
	// Each alias becomes a /commandName that calls startAgent() with a preset model.
	// Not registered in child sessions to enforce a single level of parallelism.
	const aliases = isChild ? {} : loadAliases();
	for (const [aliasName, aliasCfg] of Object.entries(aliases)) {
		const model = aliasCfg.model;
		pi.registerCommand(aliasName, {
			description: `Alias for /agent${model ? ` -model ${model}` : ""}: /${aliasName} <task>`,
			handler: async (args, ctx) => {
				if (!args?.trim()) {
					ctx.hasUI && ctx.ui.notify(`Usage: /${aliasName} <task>`, "error");
					return;
				}
				try {
					ctx.hasUI && ctx.ui.notify("Starting side-agent…", "info");
					const started = await startAgent(pi, ctx, {
						task: args.trim(),
						model,
						includeSummary: true,
					});
					const lines = [
						`id: ${started.id}`,
						`tmux window: ${started.tmuxWindowId} (#${started.tmuxWindowIndex})`,
						`worktree: ${started.worktreePath}`,
						`branch: ${started.branch}`,
					];
					for (const warning of started.warnings) {
						lines.push(`warning: ${warning}`);
					}
					lines.push("", "prompt:");
					for (const line of started.prompt.split(/\r?\n/)) {
						lines.push(`  ${line}`);
					}
					renderInfoMessage(pi, ctx, "side-agent started", lines);
					await renderStatusLine(pi, ctx).catch(() => {});
				} catch (err) {
					ctx.hasUI && ctx.ui.notify(`Failed to start agent: ${stringifyError(err)}`, "error");
				}
			},
		});
	}

	// agent-start tool: used by the LLM to spawn a child agent programmatically.
	// `includeSummary: false` because the LLM already crafts a precise description.
	// Not registered in child sessions to enforce a single level of parallelism.
	if (!isChild) pi.registerTool({
		name: "agent-start",
		label: "Agent Start",
		description:
			"Start a background side agent in tmux/worktree. Lifecycle: child implements the change or asks for clarification -> wait-state and yield -> parent inspects (agent-check or agent-wait-any), reviews work, reacts -> eventually, parent asks child to wrap up (send 'LGTM, merge'), sends /quit when child is done. Provide a short kebab-case branchHint (max 3 words) for the agent's branch name. Returns { ok: true, id, task, tmuxWindowId, tmuxWindowIndex, worktreePath, branch, warnings[] } on success, or { ok: false, error } on failure.",
		parameters: Type.Object({
			description: Type.String({ description: "Task description for child agent kickoff prompt (include all necessary context)" }),
			branchHint: Type.String({ description: "Short kebab-case branch slug, max 3 words (e.g. fix-auth-leak)" }),
			model: Type.Optional(Type.String({ description: "Model as provider/modelId (optional)" })),
		}),
		async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
			try {
				const started = await startAgent(pi, ctx, {
					task: params.description,
					branchHint: params.branchHint,
					model: params.model,
					includeSummary: false,
				});
				return {
					content: [
						{
							type: "text",
							text: JSON.stringify(
								{
									ok: true,
									id: started.id,
									// Truncate task preview in the tool response to avoid token waste.
									task: params.description.length > 200 ? params.description.slice(0, 200) + "…" : params.description,
									tmuxWindowId: started.tmuxWindowId,
									tmuxWindowIndex: started.tmuxWindowIndex,
									worktreePath: started.worktreePath,
									branch: started.branch,
									warnings: started.warnings,
								},
								null,
								2,
							),
						},
					],
					details: undefined,
				};
			} catch (err) {
				return {
					content: [{ type: "text", text: JSON.stringify({ ok: false, error: stringifyError(err) }, null, 2) }],
					details: undefined,
				};
			}
		},
	});

	// agent-check tool: used by the LLM to inspect one agent's current state.
	pi.registerTool({
		name: "agent-check",
		label: "Agent Check",
		description:
			"Check a given side agent status and return compact recent output. Returns { ok: true, agent: { id, status, tmuxWindowId, tmuxWindowIndex, worktreePath, branch, task, startedAt, finishedAt?, exitCode?, error?, warnings[] }, backlog: string[] }, or { ok: false, error } if the agent id is unknown or a registry error occurs. backlog is sanitized/truncated for LLM safety; task is a compact preview. Statuses: allocating_worktree | spawning_tmux | running | waiting_user | failed | crashed. Agents that exit with code 0 are auto-removed from registry.",
		parameters: Type.Object({
			id: Type.String({ description: "Agent id" }),
		}),
		async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
			try {
				const payload = await agentCheckPayload(getStateRoot(ctx), params.id);
				return {
					content: [{ type: "text", text: JSON.stringify(payload, null, 2) }],
					details: undefined,
				};
			} catch (err) {
				return {
					content: [{ type: "text", text: JSON.stringify({ ok: false, error: stringifyError(err) }, null, 2) }],
					details: undefined,
				};
			}
		},
	});

	// agent-wait-any tool: blocks (polls) until one of the listed agents reaches
	// a wait state, then returns that agent's check payload.
	pi.registerTool({
		name: "agent-wait-any",
		label: "Agent Wait Any",
		description:
			"Wait for an agent to finish its work. Returns the agent's status payload (same shape as agent-check) once it completes (done), yields (waiting_user), fails, or crashes.",
		parameters: Type.Object({
			ids: Type.Array(Type.String({ description: "Agent id" }), { description: "Agent ids to wait for" }),
		}),
		async execute(_toolCallId, params, signal, _onUpdate, ctx) {
			try {
				const payload = await waitForAny(getStateRoot(ctx), params.ids, signal);
				return {
					content: [{ type: "text", text: JSON.stringify(payload, null, 2) }],
					details: undefined,
				};
			} catch (err) {
				return {
					content: [{ type: "text", text: JSON.stringify({ ok: false, error: stringifyError(err) }, null, 2) }],
					details: undefined,
				};
			}
		},
	});

	// agent-send tool: used by the LLM to steer a running or waiting child agent.
	pi.registerTool({
		name: "agent-send",
		label: "Agent Send",
		description:
			"Send a steering/follow-up prompt to a child agent's tmux pane. Returns { ok: boolean, message: string }.",
		parameters: Type.Object({
			id: Type.String({ description: "Agent id" }),
			prompt: Type.String({ description: "Prompt text to send (prefix with '!' to interrupt first instead of organic steering, '/' for slash commands like /quit)" }),
		}),
		async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
			try {
				const payload = await sendToAgent(getStateRoot(ctx), params.id, params.prompt);
				return {
					content: [{ type: "text", text: JSON.stringify(payload, null, 2) }],
					details: undefined,
				};
			} catch (err) {
				return {
					content: [{ type: "text", text: JSON.stringify({ ok: false, error: stringifyError(err) }, null, 2) }],
					details: undefined,
				};
			}
		},
	});

	// agent-list-tools tool: lets the LLM introspect which tools are available,
	// so it can decide whether a subtask requires capabilities best handled by
	// a dedicated child agent versus inline tool calls.
	pi.registerTool({
		name: "agent-list-tools",
		label: "Agent List Tools",
		description:
			"List all registered tools and which ones are currently active (sent to the LLM each turn). Use this to decide whether a subtask needs a child agent with different capabilities.",
		parameters: Type.Object({}),
		async execute(_toolCallId, _params, _signal, _onUpdate, _ctx) {
			const all = pi.getAllTools();
			const activeNames = new Set(pi.getActiveTools());
			return {
				content: [
					{
						type: "text",
						text: JSON.stringify(
							{
								active: all.filter((t) => activeNames.has(t.name)).map((t) => ({ name: t.name, description: t.description })),
								inactive: all.filter((t) => !activeNames.has(t.name)).map((t) => ({ name: t.name, description: t.description })),
							},
							null,
							2,
						),
					},
				],
				details: undefined,
			};
		},
	});

	// pi.on(eventName, callback) — registers an event listener.
	// The callback is an arrow function: async (_event, ctx) => { … }
	// ▸ `_event` is prefixed with `_` to signal it is intentionally unused
	//   (TypeScript would warn about an unused parameter otherwise).
	// ▸ `ctx` is the ExtensionContext for this event (current session state).
	// ▸ The callback is `async` so it can `await` async operations inside.
	//
	// session_start: fired when Pi's session starts.
	// In child processes: links the session to the registry.
	// In parent processes: starts the status bar poller.
	pi.on("session_start", async (_event, ctx) => {
		await ensureChildSessionLinked(pi, ctx).catch(() => {});
		ensureStatusPoller(pi, ctx);
	});

	// session_switch: fired when the user switches to a different conversation branch.
	// Same dual purpose as session_start.
	pi.on("session_switch", async (_event, ctx) => {
		await ensureChildSessionLinked(pi, ctx).catch(() => {});
		ensureStatusPoller(pi, ctx);
	});

	// agent_start: fired just before the LLM starts generating a response turn.
	// In the child process: marks the agent as "running" in the registry.
	pi.on("agent_start", async (_event, ctx) => {
		await setChildRuntimeStatus(ctx, "running").catch(() => {});
	});

	// agent_end: fired after the LLM finishes a response turn (Pi is now idle,
	// waiting for the next user message).
	// In the child process: marks the agent as "waiting_user" so the parent knows
	// the child has yielded and is ready for review/feedback.
	pi.on("agent_end", async (_event, ctx) => {
		await setChildRuntimeStatus(ctx, "waiting_user").catch(() => {});
	});

	// before_agent_start: fired before each LLM turn in the parent process.
	// Updates the stored ctx reference and silently re-renders the status bar
	// (without emitting transition messages, to avoid noise mid-turn).
	pi.on("before_agent_start", async (_event, ctx) => {
		statusPollContext = ctx;
		statusPollApi = pi;
		await renderStatusLine(pi, ctx, { emitTransitions: false }).catch(() => {});
	});
}
