# context-compact

LLM context window compactor — summarizes old conversation history to free up context space without losing continuity.

## The Problem

Long-running AI agent sessions accumulate thousands of messages. When the context window fills up, naive truncation drops old messages and the agent loses critical context: active tasks, decisions made, identifiers referenced. The conversation breaks.

**context-compact** solves this by summarizing old messages via your LLM of choice, then replacing them with a compact summary. The agent retains continuity while freeing up context space.

## Install

```bash
npm install context-compact
```

## Quick Start — Anthropic SDK

```typescript
import Anthropic from "@anthropic-ai/sdk";
import { compactIfNeeded, type SummarizeFn } from "context-compact";

const client = new Anthropic();

const summarize: SummarizeFn = async (messages, instructions, previousSummary) => {
  const systemPrompt = previousSummary
    ? `${instructions}\n\nPrevious summary:\n${previousSummary}`
    : instructions;

  const response = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 4096,
    system: systemPrompt,
    messages: messages.map((m) => ({
      role: m.role === "assistant" ? "assistant" : "user",
      content: typeof m.content === "string" ? m.content : JSON.stringify(m.content),
    })),
  });

  return response.content[0].type === "text" ? response.content[0].text : "";
};

// In your agent loop:
const result = await compactIfNeeded({
  messages: conversationHistory,
  contextWindowTokens: 200_000, // Claude's context window
  summarize,
});

// Use result.messages as the new conversation history
conversationHistory = result.messages;
```

## Quick Start — OpenAI SDK

```typescript
import OpenAI from "openai";
import { compactIfNeeded, type SummarizeFn } from "context-compact";

const client = new OpenAI();

const summarize: SummarizeFn = async (messages, instructions, previousSummary) => {
  const systemContent = previousSummary
    ? `${instructions}\n\nPrevious summary:\n${previousSummary}`
    : instructions;

  const response = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [
      { role: "system", content: systemContent },
      ...messages.map((m) => ({
        role: m.role as "user" | "assistant" | "system",
        content: typeof m.content === "string" ? m.content : JSON.stringify(m.content),
      })),
    ],
  });

  return response.choices[0].message.content ?? "";
};

const result = await compactIfNeeded({
  messages: conversationHistory,
  contextWindowTokens: 128_000,
  summarize,
});

conversationHistory = result.messages;
```

## How Chunking Works

The summarization model itself has a context limit. You can't send 100k tokens of history to a model with a 128k window — there's no room for the response.

**context-compact** splits old messages into chunks that fit within the summarization model's context, then summarizes sequentially with a running summary carried forward. For very long histories, it can split into parallel parts, summarize each independently, then merge.

```
Messages: [m1, m2, m3, ..., m500]
                    ↓
         Split into chunks
      [m1..m100] [m101..m200] [m201..m300] ...
                    ↓
       Summarize sequentially
      summary₁ → summary₂ → summary₃ → ...
                    ↓
          Final summary + kept messages
```

The chunk size adapts to message size. If messages are large (e.g., code files in tool results), chunks shrink to avoid exceeding the summarization model's limit.

## Identifier Preservation

Agent conversations are full of opaque identifiers that must survive summarization exactly:

- UUIDs: `550e8400-e29b-41d4-a716-446655440000`
- File paths: `/src/components/Auth/LoginForm.tsx`
- URLs: `https://api.example.com/v2/users/42`
- Hashes: `sha256:a3f2b8c...`
- IPs/ports: `192.168.1.100:8080`

By default (`identifierPolicy: "strict"`), the summarization prompt instructs the LLM to preserve these verbatim. You can customize or disable this:

```typescript
// Custom policy
compactIfNeeded({
  // ...
  options: {
    identifierPolicy: "custom",
    identifierInstructions: "Preserve all file paths and URLs exactly.",
  },
});

// Disable
compactIfNeeded({
  // ...
  options: { identifierPolicy: "off" },
});
```

## Token Estimation

Token counting without a tokenizer SDK uses a `chars / 4` heuristic. This underestimates for code and unicode text. The `safetyMargin` option (default: `1.2`) compensates by multiplying the estimate when making threshold decisions.

```typescript
import { estimateTokens } from "context-compact";

const tokens = estimateTokens(messages);
// Use for budgeting, not exact billing
```

## API Reference

### `compactIfNeeded(params)`

Main entry point. Evaluates whether compaction is needed and compacts if so.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `messages` | `Message[]` | required | Full conversation history |
| `contextWindowTokens` | `number` | required | Model's context window size |
| `triggerRatio` | `number` | `0.85` | Fraction of context that triggers compaction |
| `keepRatio` | `number` | `0.5` | Fraction of tokens to keep as recent history |
| `summarize` | `SummarizeFn` | required | Your LLM summarization callback |
| `options` | `CompactionOptions` | `{}` | See options below |

Returns `CompactResult` with `compacted`, `messages`, `summary`, and `stats`.

### `compact(params)`

Lower-level: compact unconditionally.

| Parameter | Type | Description |
|-----------|------|-------------|
| `toSummarize` | `Message[]` | Messages to compress |
| `toKeep` | `Message[]` | Messages to preserve verbatim |
| `summarize` | `SummarizeFn` | Your LLM summarization callback |
| `options` | `CompactionOptions` | See options below |

### `estimateTokens(messages)`

Estimate token count without an API call.

### `repairToolPairing(messages)`

Fix orphaned `tool_result` blocks whose `tool_use` partner was dropped. Returns `{ messages, droppedOrphanCount }`.

### `CompactionOptions`

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `identifierPolicy` | `"strict" \| "off" \| "custom"` | `"strict"` | Identifier preservation mode |
| `identifierInstructions` | `string` | — | Custom instructions for `"custom"` policy |
| `customInstructions` | `string` | — | Extra instructions appended to prompt |
| `maxChunkTokens` | `number` | auto | Max tokens per summarization chunk |
| `safetyMargin` | `number` | `1.2` | Token estimate multiplier |
| `parts` | `number` | `2` | Parallel summarization parts |
| `signal` | `AbortSignal` | — | Cancellation signal |

### `SummarizeFn`

```typescript
type SummarizeFn = (
  messages: Message[],
  instructions: string,
  previousSummary?: string,
) => Promise<string>;
```

You implement this with your LLM SDK. The library calls it with messages to summarize, instructions for the summarization, and an optional running summary from prior chunks.

## Security

Before messages are passed to your `summarize` callback, all `tool_result.details` fields are stripped. These often contain large, untrusted payloads from tool executions (file contents, API responses) that should not reach the summarization model.

## License

MIT
