import type {
  CompactResult,
  CompactStats,
  CompactionOptions,
  Message,
  SummarizeFn,
} from "./types.js";
import { computeAdaptiveChunkRatio } from "./chunking.js";
import { repairToolPairing } from "./repair.js";
import { summarizeInStages } from "./summarize.js";
import { estimateTokens } from "./tokens.js";

/**
 * Check whether compaction would trigger, without running it.
 * Returns stats useful for making informed decisions.
 */
export function shouldCompact(params: {
  messages: Message[];
  contextWindowTokens: number;
  triggerRatio?: number;
  keepRatio?: number;
  options?: CompactionOptions;
}): CompactStats {
  const {
    messages,
    contextWindowTokens,
    triggerRatio = 0.85,
    keepRatio = 0.5,
    options = {},
  } = params;

  validateParams(contextWindowTokens, triggerRatio, keepRatio);

  const safetyMargin = options.safetyMargin ?? 1.2;
  const tokenCounter = options.tokenCounter;
  const totalTokens = estimateTokens(messages, tokenCounter);
  const threshold = Math.floor(contextWindowTokens * triggerRatio);
  const wouldTrigger = Math.ceil(totalTokens * safetyMargin) >= threshold;

  const keepFrom = computeKeepFrom(messages, totalTokens, keepRatio, tokenCounter);

  return {
    shouldCompact: wouldTrigger && keepFrom > 0,
    estimatedTokens: totalTokens,
    threshold,
    messagesTotal: messages.length,
    estimatedKeepFrom: keepFrom,
  };
}

/**
 * Evaluate whether compaction is needed and, if so, compact.
 *
 * Compaction triggers when estimated tokens exceed triggerRatio × contextWindowTokens.
 * The older portion of messages is summarized, and the most recent keepRatio fraction
 * is preserved verbatim.
 */
export async function compactIfNeeded(params: {
  messages: Message[];
  contextWindowTokens: number;
  triggerRatio?: number;
  keepRatio?: number;
  summarize: SummarizeFn;
  options?: CompactionOptions;
}): Promise<CompactResult> {
  const {
    messages,
    contextWindowTokens,
    triggerRatio = 0.85,
    keepRatio = 0.5,
    summarize,
    options = {},
  } = params;

  validateParams(contextWindowTokens, triggerRatio, keepRatio);

  const safetyMargin = options.safetyMargin ?? 1.2;
  const tokenCounter = options.tokenCounter;
  const totalTokens = estimateTokens(messages, tokenCounter);
  const threshold = Math.floor(contextWindowTokens * triggerRatio);

  options.onProgress?.("estimating", 1, 1);

  // Apply safety margin to the token estimate for threshold comparison
  if (Math.ceil(totalTokens * safetyMargin) < threshold) {
    return {
      compacted: false,
      messages,
      summary: "",
      stats: {
        tokensBefore: totalTokens,
        tokensAfter: totalTokens,
        messagesCompressed: 0,
        messagesKept: messages.length,
      },
    };
  }

  // Determine split point: keep the most recent keepRatio fraction by tokens
  const keepFrom = computeKeepFrom(messages, totalTokens, keepRatio, tokenCounter);

  // If keepFrom === 0, everything fits in the keep budget — nothing meaningful to summarize
  if (keepFrom === 0) {
    return {
      compacted: false,
      messages,
      summary: "",
      stats: {
        tokensBefore: totalTokens,
        tokensAfter: totalTokens,
        messagesCompressed: 0,
        messagesKept: messages.length,
      },
    };
  }

  // Ensure we always keep at least 1 message
  const effectiveKeepFrom = Math.min(keepFrom, messages.length - 1);

  const toSummarize = messages.slice(0, effectiveKeepFrom);
  const toKeep = messages.slice(effectiveKeepFrom);

  options.onProgress?.("splitting", 1, 1);

  return compact({
    toSummarize,
    toKeep,
    summarize,
    options: {
      ...options,
      maxChunkTokens:
        options.maxChunkTokens ??
        Math.floor(
          contextWindowTokens *
            computeAdaptiveChunkRatio(messages, contextWindowTokens, safetyMargin, tokenCounter),
        ),
    },
    contextWindowTokens,
  });
}

/**
 * Compact unconditionally: summarize `toSummarize` messages and prepend
 * the summary to `toKeep` messages.
 */
export async function compact(params: {
  toSummarize: Message[];
  toKeep: Message[];
  summarize: SummarizeFn;
  options?: CompactionOptions;
  contextWindowTokens: number;
}): Promise<CompactResult> {
  const {
    toSummarize,
    toKeep,
    summarize,
    options = {},
    contextWindowTokens,
  } = params;

  const safetyMargin = options.safetyMargin ?? 1.2;
  const tokenCounter = options.tokenCounter;
  const maxChunkTokens =
    options.maxChunkTokens ??
    Math.floor(
      contextWindowTokens *
        computeAdaptiveChunkRatio(toSummarize, contextWindowTokens, safetyMargin, tokenCounter),
    );

  const tokensBefore = estimateTokens([...toSummarize, ...toKeep], tokenCounter);

  const summary = await summarizeInStages(toSummarize, summarize, {
    ...options,
    maxChunkTokens,
    contextWindowTokens,
  });

  // Determine the role for the summary message to avoid consecutive same-role messages
  let summaryRole: "user" | "assistant" = "user";
  if (toKeep.length > 0 && toKeep[0].role === "user") {
    summaryRole = "assistant";
  }

  const summaryMessage: Message = {
    role: summaryRole,
    content: `[Previous conversation summary]\n\n${summary}`,
    _meta: { synthetic: true },
  };

  // Combine summary + kept messages, then repair tool pairing
  const combined = [summaryMessage, ...toKeep];
  const { messages: repairedMessages, droppedOrphanCount, droppedOrphanUseCount } =
    repairToolPairing(combined);

  const tokensAfter = estimateTokens(repairedMessages, tokenCounter);

  return {
    compacted: true,
    messages: repairedMessages,
    summary,
    stats: {
      tokensBefore,
      tokensAfter,
      messagesCompressed: toSummarize.length,
      messagesKept: toKeep.length,
      droppedOrphanResults: droppedOrphanCount,
      droppedOrphanUses: droppedOrphanUseCount,
    },
  };
}

// ── Private helpers ──────────────────────────────────

function validateParams(
  contextWindowTokens: number,
  triggerRatio: number,
  keepRatio: number,
): void {
  if (contextWindowTokens <= 0) {
    throw new Error(
      `contextWindowTokens must be positive, got ${contextWindowTokens}`,
    );
  }
  if (triggerRatio < 0 || triggerRatio > 1) {
    throw new Error(
      `triggerRatio must be between 0 and 1, got ${triggerRatio}`,
    );
  }
  if (keepRatio < 0 || keepRatio > 1) {
    throw new Error(
      `keepRatio must be between 0 and 1, got ${keepRatio}`,
    );
  }
}

function computeKeepFrom(
  messages: Message[],
  totalTokens: number,
  keepRatio: number,
  tokenCounter?: (text: string) => number,
): number {
  const keepTokenTarget = Math.floor(totalTokens * keepRatio);
  let keepFrom = messages.length;
  let keepTokens = 0;

  for (let i = messages.length - 1; i >= 0; i--) {
    const msgTokens = estimateTokens([messages[i]], tokenCounter);
    if (keepTokens + msgTokens > keepTokenTarget && keepFrom < messages.length) {
      break;
    }
    keepTokens += msgTokens;
    keepFrom = i;
  }

  return keepFrom;
}
