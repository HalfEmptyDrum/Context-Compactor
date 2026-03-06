import type {
  CompactResult,
  CompactionOptions,
  Message,
  SummarizeFn,
} from "./types.js";
import { computeAdaptiveChunkRatio } from "./chunking.js";
import { repairToolPairing } from "./repair.js";
import { summarizeInStages } from "./summarize.js";
import { estimateTokens } from "./tokens.js";

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

  const safetyMargin = options.safetyMargin ?? 1.2;
  const totalTokens = estimateTokens(messages);
  const threshold = Math.floor(contextWindowTokens * triggerRatio);

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
  const keepTokenTarget = Math.floor(totalTokens * keepRatio);
  let keepFrom = messages.length;
  let keepTokens = 0;

  for (let i = messages.length - 1; i >= 0; i--) {
    const msgTokens = estimateTokens([messages[i]]);
    if (keepTokens + msgTokens > keepTokenTarget && keepFrom < messages.length) {
      break;
    }
    keepTokens += msgTokens;
    keepFrom = i;
  }

  // Ensure we keep at least one message and compress at least one
  if (keepFrom === 0) {
    keepFrom = Math.min(1, messages.length);
  }

  const toSummarize = messages.slice(0, keepFrom);
  const toKeep = messages.slice(keepFrom);

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
            computeAdaptiveChunkRatio(messages, contextWindowTokens, safetyMargin),
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
  contextWindowTokens?: number;
}): Promise<CompactResult> {
  const {
    toSummarize,
    toKeep,
    summarize,
    options = {},
    contextWindowTokens,
  } = params;

  const safetyMargin = options.safetyMargin ?? 1.2;
  const effectiveContextWindow = contextWindowTokens ?? 128000;
  const maxChunkTokens =
    options.maxChunkTokens ??
    Math.floor(
      effectiveContextWindow *
        computeAdaptiveChunkRatio(toSummarize, effectiveContextWindow, safetyMargin),
    );

  const tokensBefore = estimateTokens([...toSummarize, ...toKeep]);

  const summary = await summarizeInStages(toSummarize, summarize, {
    ...options,
    maxChunkTokens,
    contextWindowTokens: effectiveContextWindow,
  });

  const summaryMessage: Message = {
    role: "user",
    content: `[Previous conversation summary]\n\n${summary}`,
  };

  // Combine summary + kept messages, then repair tool pairing
  const combined = [summaryMessage, ...toKeep];
  const { messages: repairedMessages } = repairToolPairing(combined);

  const tokensAfter = estimateTokens(repairedMessages);

  return {
    compacted: true,
    messages: repairedMessages,
    summary,
    stats: {
      tokensBefore,
      tokensAfter,
      messagesCompressed: toSummarize.length,
      messagesKept: toKeep.length,
    },
  };
}
