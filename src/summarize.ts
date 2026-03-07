import type { CompactionOptions, Message, SummarizeFn } from "./types.js";
import { chunkByMaxTokens, splitIntoEqualParts } from "./chunking.js";
import { retryWithBackoff } from "./retry.js";
import { stripToolResultDetails } from "./security.js";
import { estimateMessageTokens, estimateTokens } from "./tokens.js";

const IDENTIFIER_INSTRUCTIONS = `Preserve all opaque identifiers exactly as written (no shortening or reconstruction),
including UUIDs, hashes, IDs, tokens, API keys, hostnames, IPs, ports, URLs, and file names.`;

const MERGE_INSTRUCTIONS = `Merge these partial summaries into a single cohesive summary.
MUST PRESERVE:
- Active tasks and their current status (in-progress, blocked, pending)
- Batch operation progress (e.g., '5/17 items completed')
- The last thing the user requested and what was being done about it
- Decisions made and their rationale
- TODOs, open questions, and constraints
- Any commitments or follow-ups promised
PRIORITIZE recent context over older history. The agent needs to know
what it was doing, not just what was discussed.`;

type InternalOptions = {
  safetyMargin: number;
  maxChunkTokens: number;
  contextWindowTokens: number;
  signal?: AbortSignal;
  tokenCounter?: (text: string) => number;
  stripFields?: string[];
  retryAttempts: number;
  retryMinDelayMs: number;
  retryMaxDelayMs: number;
  onProgress?: (stage: string, current: number, total: number) => void;
};

function resolveInternalOptions(
  options: CompactionOptions & { contextWindowTokens: number },
): InternalOptions {
  return {
    safetyMargin: options.safetyMargin ?? 1.2,
    maxChunkTokens:
      options.maxChunkTokens ??
      Math.floor(options.contextWindowTokens * 0.4),
    contextWindowTokens: options.contextWindowTokens,
    signal: options.signal,
    tokenCounter: options.tokenCounter,
    stripFields: options.stripFields,
    retryAttempts: options.retryAttempts ?? 3,
    retryMinDelayMs: options.retryMinDelayMs ?? 500,
    retryMaxDelayMs: options.retryMaxDelayMs ?? 5000,
    onProgress: options.onProgress,
  };
}

function buildInstructions(options: CompactionOptions): string {
  const parts: string[] = [
    "Summarize the following conversation history concisely. Focus on key decisions, outcomes, and context needed for continuity.",
  ];

  const policy = options.identifierPolicy ?? "strict";
  if (policy === "strict") {
    parts.push(IDENTIFIER_INSTRUCTIONS);
  } else if (policy === "custom" && options.identifierInstructions) {
    parts.push(options.identifierInstructions);
  }

  if (options.customInstructions) {
    parts.push(options.customInstructions);
  }

  return parts.join("\n\n");
}

/**
 * Summarize messages in sequential chunks, passing the running summary forward.
 */
async function summarizeChunks(
  messages: Message[],
  summarize: SummarizeFn,
  instructions: string,
  opts: InternalOptions,
): Promise<string> {
  const sanitized = stripToolResultDetails(messages, opts.stripFields);
  const chunks = chunkByMaxTokens(
    sanitized,
    opts.maxChunkTokens,
    opts.safetyMargin,
    opts.tokenCounter,
  );

  let summary: string | undefined;

  for (let i = 0; i < chunks.length; i++) {
    opts.signal?.throwIfAborted();
    opts.onProgress?.("summarizing", i + 1, chunks.length);

    summary = await retryWithBackoff(
      () => summarize(chunks[i], instructions, summary),
      {
        attempts: opts.retryAttempts,
        minDelayMs: opts.retryMinDelayMs,
        maxDelayMs: opts.retryMaxDelayMs,
        signal: opts.signal,
      },
    );
  }

  return summary ?? "No prior history.";
}

/**
 * Summarize with progressive fallback for oversized messages.
 */
async function summarizeWithFallback(
  messages: Message[],
  summarize: SummarizeFn,
  instructions: string,
  opts: InternalOptions,
): Promise<string> {
  try {
    return await summarizeChunks(messages, summarize, instructions, opts);
  } catch (error) {
    // Re-throw abort errors — don't fall back on user cancellation
    if (opts.signal?.aborted) {
      throw error;
    }

    // Progressive fallback: separate oversized messages
    const oversizeThreshold = opts.contextWindowTokens * 0.5;

    const smallMessages: Message[] = [];
    const oversizedNotes: string[] = [];

    for (const msg of messages) {
      const tokens = estimateMessageTokens(msg, opts.tokenCounter);
      if (tokens > oversizeThreshold) {
        const approxK = Math.round(tokens / 1000);
        oversizedNotes.push(
          `[Large message (~${approxK}k tokens) omitted]`,
        );
      } else {
        smallMessages.push(msg);
      }
    }

    if (smallMessages.length > 0) {
      try {
        const partial = await summarizeChunks(
          smallMessages,
          summarize,
          instructions,
          opts,
        );
        if (oversizedNotes.length > 0) {
          return partial + "\n\n" + oversizedNotes.join("\n");
        }
        return partial;
      } catch (innerError) {
        if (opts.signal?.aborted) {
          throw innerError;
        }
        // Fall through to final fallback
      }
    }

    // Final fallback
    return `Context contained ${messages.length} messages. Summary unavailable due to size limits.`;
  }
}

/**
 * Multi-stage summarization: split into parts, summarize each in parallel,
 * then merge the partial summaries.
 */
export async function summarizeInStages(
  messages: Message[],
  summarize: SummarizeFn,
  options: CompactionOptions & { contextWindowTokens: number },
): Promise<string> {
  const opts = resolveInternalOptions(options);
  const parts = options.parts ?? 2;

  const instructions = buildInstructions(options);
  const totalTokens = estimateTokens(messages, opts.tokenCounter);

  if (parts <= 1 || totalTokens <= opts.maxChunkTokens) {
    return summarizeWithFallback(messages, summarize, instructions, opts);
  }

  const splits = splitIntoEqualParts(messages, parts, opts.tokenCounter).filter(
    (c) => c.length > 0,
  );

  if (splits.length <= 1) {
    return summarizeWithFallback(messages, summarize, instructions, opts);
  }

  opts.onProgress?.("splitting", 0, splits.length);

  // Summarize each part in parallel
  const partialSummaries = await Promise.all(
    splits.map((s) =>
      summarizeWithFallback(s, summarize, instructions, opts),
    ),
  );

  opts.onProgress?.("merging", 0, 1);

  // Merge partial summaries
  const mergeMessages: Message[] = partialSummaries.map((s) => ({
    role: "user" as const,
    content: s,
  }));

  const mergeInstructions = [MERGE_INSTRUCTIONS];
  const policy = options.identifierPolicy ?? "strict";
  if (policy === "strict") {
    mergeInstructions.push(IDENTIFIER_INSTRUCTIONS);
  } else if (policy === "custom" && options.identifierInstructions) {
    mergeInstructions.push(options.identifierInstructions);
  }

  return summarizeWithFallback(
    mergeMessages,
    summarize,
    mergeInstructions.join("\n\n"),
    opts,
  );
}

export { buildInstructions, IDENTIFIER_INSTRUCTIONS, MERGE_INSTRUCTIONS };
