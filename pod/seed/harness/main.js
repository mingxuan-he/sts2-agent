// Pod harness: one process = one session = one context window.
// The supervisor restarts this forever; /pod files are the only memory.
//
// This code is YOURS. The stop logic, the logging, the context assembly,
// the tools — all editable. Changes take effect next session.
//
// Exit codes (supervisor convention): 0 finished, 2 stalled, 3 token cap, 1 crash.

import fs from "node:fs";
import path from "node:path";
import { Agent } from "@earendil-works/pi-agent-core";
import { createModels, createProvider, envApiKeyAuth } from "@earendil-works/pi-ai";
import { openAICompletionsApi } from "@earendil-works/pi-ai/api/openai-completions.lazy";
import { makeTools } from "./tools.js";

const POD = process.env.POD_ROOT ?? "/pod";
const BOOTSTRAP = process.env.BOOTSTRAP_PATH ?? "/opt/bootstrap.md";
const GAME_URL = process.env.GAME_URL ?? "http://game:8300";
const TOKEN_CAP = parseInt(process.env.SESSION_TOKEN_CAP ?? "200000", 10);
const STALL_LIMIT = 3;

// ── model: any OpenAI-compatible endpoint, fully env-configured ────────────

const modelDef = {
  id: process.env.MODEL_ID ?? "qwen/qwen3.6-35b-a3b",
  name: process.env.MODEL_ID ?? "qwen/qwen3.6-35b-a3b",
  api: "openai-completions",
  provider: "llm",
  baseUrl: process.env.MODEL_BASE_URL ?? "https://openrouter.ai/api/v1",
  reasoning: false,
  input: ["text"],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: parseInt(process.env.MODEL_CONTEXT_WINDOW ?? "262144", 10),
  maxTokens: parseInt(process.env.MODEL_MAX_TOKENS ?? "16384", 10),
};

const models = createModels();
models.setProvider(
  createProvider({
    id: "llm",
    name: "LLM endpoint",
    baseUrl: modelDef.baseUrl,
    auth: { apiKey: envApiKeyAuth("LLM API key", ["MODEL_API_KEY", "OPENROUTER_API_KEY"]) },
    models: [modelDef],
    api: openAICompletionsApi(),
  }),
);
const model = models.getModel("llm", modelDef.id);

// ── session state + JSONL log ──────────────────────────────────────────────

const sessionId = new Date().toISOString().replace(/[:.]/g, "-");
const session = { finished: false, reason: null, lastError: null, stalls: 0, tokens: { input: 0, output: 0 }, capWarned: false };

fs.mkdirSync(path.join(POD, "sessions"), { recursive: true });
const logStream = fs.createWriteStream(path.join(POD, "sessions", `${sessionId}.jsonl`), { flags: "a" });
function log(entry) {
  logStream.write(JSON.stringify({ t: Date.now(), ...entry }) + "\n");
}

// ── context assembly ───────────────────────────────────────────────────────

function readIfExists(p) {
  try {
    return fs.readFileSync(p, "utf-8");
  } catch {
    return "";
  }
}

const bootstrap = readIfExists(BOOTSTRAP)
  .replaceAll("$GAME_URL", GAME_URL)
  .replaceAll("$SESSION_TOKEN_CAP", String(TOKEN_CAP));
const promptMd = readIfExists(path.join(POD, "PROMPT.md"));
const handoff = readIfExists(path.join(POD, "HANDOFF.md"));

const systemPrompt = [bootstrap, "\n---\n\n# Your PROMPT.md\n", promptMd].join("\n");

const openingMessage = [
  `Session ${sessionId}. Token budget: ${TOKEN_CAP}.`,
  handoff ? `\nHandoff from your previous session:\n\n${handoff}` : "\nNo handoff found — this may be your first session.",
  "\nDecide how to spend this session, then work. End with finish_session.",
].join("\n");

// ── agent ──────────────────────────────────────────────────────────────────

const agent = new Agent({
  initialState: {
    systemPrompt,
    model,
    tools: makeTools({ gameUrl: GAME_URL, session, pod: POD }),
  },
  streamFn: models.streamSimple.bind(models),
  sessionId,

  afterToolCall: async ({ toolCall, result, isError }) => {
    log({
      type: "tool",
      name: toolCall.name,
      args: toolCall.arguments,
      isError: !!isError,
      result: result?.content?.map((c) => c.text ?? "").join("\n") ?? String(result),
    });
  },

  shouldStopAfterTurn: async ({ message, toolResults }) => {
    if (session.finished) {
      session.reason = "finished";
      return true;
    }

    const spent = session.tokens.input + session.tokens.output;
    if (spent >= TOKEN_CAP) {
      if (!session.capWarned) {
        session.capWarned = true;
        agent.steer({
          role: "user",
          content:
            "Your token budget is exhausted. This is your final turn: call finish_session " +
            "with a handoff for the next session NOW.",
          timestamp: Date.now(),
        });
        return false;
      }
      session.reason = "cap";
      return true;
    }
    if (!session.capWarned && spent >= TOKEN_CAP * 0.85) {
      agent.steer({
        role: "user",
        content: `Budget warning: ~${spent} of ${TOKEN_CAP} tokens spent. Wrap up and write your handoff soon.`,
        timestamp: Date.now(),
      });
      return false;
    }

    if ((toolResults?.length ?? 0) === 0) {
      session.stalls += 1;
      if (session.stalls >= STALL_LIMIT) {
        session.reason = "stalled";
        return true;
      }
      agent.followUp({
        role: "user",
        content:
          "Your last turn called no tools. Nobody reads prose here — act through tools, " +
          "or end the session with finish_session.",
        timestamp: Date.now(),
      });
      return false;
    }
    session.stalls = 0;
    return false;
  },
});

agent.subscribe((event) => {
  if (event.type === "message_end") {
    const m = event.message;
    if (m.role === "assistant") {
      if (m.stopReason === "error") {
        session.lastError = m.errorMessage ?? "provider error";
      }
      if (m.usage) {
        session.tokens.input += m.usage.input ?? 0;
        session.tokens.output += m.usage.output ?? 0;
      }
      const textParts = (Array.isArray(m.content) ? m.content : [])
        .filter((c) => c.type === "text")
        .map((c) => c.text)
        .join("\n");
      log({ type: "assistant", text: textParts, usage: m.usage });
    } else if (m.role === "user") {
      log({ type: "user", text: typeof m.content === "string" ? m.content : JSON.stringify(m.content) });
    }
  }
});

// ── run ────────────────────────────────────────────────────────────────────

log({ type: "session_start", sessionId, model: modelDef.id, tokenCap: TOKEN_CAP });
try {
  await agent.prompt(openingMessage);
  await agent.waitForIdle();
} catch (err) {
  log({ type: "crash", error: String(err?.stack ?? err) });
  console.error(err);
  await closeLog();
  process.exit(1);
}

// A provider/stream error (bad key, outage) is a crash, not a stall: let the
// supervisor apply exponential backoff instead of burning the session cap.
if (!session.finished && session.lastError) {
  log({ type: "session_end", reason: "error", error: session.lastError, tokens: session.tokens });
  console.error(`[harness] provider error: ${session.lastError}`);
  await closeLog();
  process.exit(1);
}

session.reason ??= session.finished ? "finished" : "stalled";
log({ type: "session_end", reason: session.reason, tokens: session.tokens });
console.log(`[harness] session ${sessionId} ended: ${session.reason} (${session.tokens.input} in / ${session.tokens.output} out)`);
await closeLog();
process.exit({ finished: 0, stalled: 2, cap: 3 }[session.reason] ?? 0);

function closeLog() {
  return new Promise((resolve) => logStream.end(resolve));
}
