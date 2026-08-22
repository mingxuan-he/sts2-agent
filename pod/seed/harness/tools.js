// Tool definitions for the pod agent. All of this is yours to extend:
// add tools here (they register on the next session), or just write
// scripts in /pod/bin and call them through `bash`.

import { execFile } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { Type } from "typebox";

const MAX_TOOL_OUTPUT = 50_000; // chars; keeps one tool result from flooding context

function text(t) {
  return { content: [{ type: "text", text: truncate(t) }] };
}

function truncate(t) {
  if (t.length <= MAX_TOOL_OUTPUT) return t;
  return t.slice(0, MAX_TOOL_OUTPUT) + `\n...[truncated, ${t.length} chars total]`;
}

async function gameCall(gameUrl, method, route, body) {
  const res = await fetch(gameUrl + route, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const payload = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(`game service ${res.status}: ${JSON.stringify(payload.detail ?? payload)}`);
  }
  return payload;
}

function renderState(payload) {
  let out = `run: ${payload.run} | seed: ${payload.seed} | done: ${payload.done}\n`;
  if (payload.error) out += `ACTION REJECTED: ${payload.error}\n`;
  out += typeof payload.state === "string" ? payload.state : JSON.stringify(payload.state, null, 1);
  return out;
}

export function makeTools({ gameUrl, session, pod }) {
  const POD = pod ?? "/pod";
  const game_new_run = {
    name: "game_new_run",
    description:
      "Start a new game run. The server assigns the seed. " +
      "Characters: Ironclad, Silent, Defect, Necrobinder, Regent.",
    parameters: Type.Object({
      character: Type.String(),
      ascension: Type.Optional(Type.Integer({ minimum: 0, maximum: 20 })),
    }),
    execute: async (_id, p) => {
      const payload = await gameCall(gameUrl, "POST", "/runs", {
        character: p.character,
        ascension: p.ascension ?? 0,
      });
      return text(renderState(payload));
    },
  };

  const game_state = {
    name: "game_state",
    description:
      "Get the current observation for a run. format='json' returns the raw state object.",
    parameters: Type.Object({
      run: Type.String(),
      format: Type.Optional(Type.Union([Type.Literal("text"), Type.Literal("json")])),
    }),
    execute: async (_id, p) => {
      const payload = await gameCall(gameUrl, "GET", `/runs/${p.run}/state?format=${p.format ?? "text"}`);
      return text(renderState(payload));
    },
  };

  const game_action = {
    name: "game_action",
    description:
      "Take an action in a run. The observation's [Actions] line lists legal verbs and args. " +
      "Illegal actions return ACTION REJECTED and change nothing.",
    parameters: Type.Object({
      run: Type.String(),
      action: Type.String(),
      args: Type.Optional(Type.Object({}, { additionalProperties: true })),
    }),
    execute: async (_id, p) => {
      const payload = await gameCall(gameUrl, "POST", `/runs/${p.run}/action`, {
        action: p.action,
        args: p.args ?? {},
      });
      return text(renderState(payload));
    },
  };

  const game_detail = {
    name: "game_detail",
    description:
      "Full detail views for a run: deck (all cards + descriptions), piles (draw sorted/discard/exhaust, " +
      "combat only), relics, potions, map (whole act incl. boss).",
    parameters: Type.Object({
      run: Type.String(),
      endpoint: Type.Union(["deck", "piles", "relics", "potions", "map"].map((e) => Type.Literal(e))),
    }),
    execute: async (_id, p) => {
      const payload = await gameCall(gameUrl, "GET", `/runs/${p.run}/${p.endpoint}`);
      return text(payload[p.endpoint] ?? JSON.stringify(payload));
    },
  };

  const game_list_runs = {
    name: "game_list_runs",
    description: "List runs currently open on the game service (yours may span sessions).",
    parameters: Type.Object({}),
    execute: async () => {
      const payload = await gameCall(gameUrl, "GET", "/runs");
      return text(JSON.stringify(payload, null, 1));
    },
  };

  const read_file = {
    name: "read_file",
    description: "Read a file (your home is /pod).",
    parameters: Type.Object({ path: Type.String() }),
    execute: async (_id, p) => text(fs.readFileSync(p.path, "utf-8")),
  };

  const write_file = {
    name: "write_file",
    description: "Write a file, creating parent directories. Overwrites.",
    parameters: Type.Object({ path: Type.String(), content: Type.String() }),
    execute: async (_id, p) => {
      fs.mkdirSync(path.dirname(p.path), { recursive: true });
      fs.writeFileSync(p.path, p.content);
      return text(`wrote ${p.content.length} chars to ${p.path}`);
    },
  };

  const bash = {
    name: "bash",
    description: "Run a shell command (cwd /pod). Use for ls, grep, moving files, running your scripts.",
    parameters: Type.Object({
      command: Type.String(),
      timeout_s: Type.Optional(Type.Integer({ minimum: 1, maximum: 600 })),
    }),
    execute: (_id, p) =>
      new Promise((resolve, reject) => {
        execFile(
          "/bin/sh",
          ["-c", p.command],
          { cwd: POD, timeout: (p.timeout_s ?? 60) * 1000, maxBuffer: 10 * 1024 * 1024 },
          (err, stdout, stderr) => {
            const out = `${stdout}${stderr ? `\n[stderr]\n${stderr}` : ""}`.trim();
            if (err && err.killed) reject(new Error(`timed out\n${out}`));
            else if (err) resolve(text(`exit ${err.code}\n${out}`));
            else resolve(text(out || "(no output)"));
          },
        );
      }),
  };

  const finish_session = {
    name: "finish_session",
    description:
      "End this session deliberately. Your handoff is written to /pod/HANDOFF.md and is the " +
      "first thing the next session reads — say what you were doing, what you learned, what's next.",
    parameters: Type.Object({ handoff: Type.String() }),
    execute: async (_id, p) => {
      fs.writeFileSync(path.join(POD, "HANDOFF.md"), p.handoff);
      session.finished = true;
      return { content: [{ type: "text", text: "session ended" }], terminate: true };
    },
  };

  return [
    game_new_run,
    game_state,
    game_action,
    game_detail,
    game_list_runs,
    read_file,
    write_file,
    bash,
    finish_session,
  ];
}
