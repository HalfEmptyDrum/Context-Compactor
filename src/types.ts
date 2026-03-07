/**
 * A single content block within a message.
 */
export type ContentBlock = {
  type: string;
  [key: string]: unknown;
};

/**
 * Standard LLM message format compatible with OpenAI/Anthropic APIs.
 * Additional fields are passed through untouched.
 */
export type Message = {
  role: "user" | "assistant" | "system" | "tool";
  content: string | ContentBlock[];
  /** Metadata added by the library (e.g., synthetic summary messages). */
  _meta?: { synthetic?: boolean; [key: string]: unknown };
  [key: string]: unknown;
};

/**
 * User-supplied function that summarizes messages via an LLM call.
 * The library is model-agnostic — you bring your own LLM.
 *
 * @param messages - Messages to summarize
 * @param instructions - Summarization instructions/prompt
 * @param previousSummary - Running summary from prior chunks (if any)
 * @returns The summary text
 */
export type SummarizeFn = (
  messages: Message[],
  instructions: string,
  previousSummary?: string,
) => Promise<string>;

/**
 * Options for controlling compaction behavior.
 */
export type CompactionOptions = {
  /**
   * How to handle identifier preservation in the summarization prompt.
   * - "strict" (default): instruct the LLM to preserve UUIDs, hashes, IPs, URLs, etc.
   * - "off": no identifier instructions
   * - "custom": use identifierInstructions string instead
   */
  identifierPolicy?: "strict" | "off" | "custom";

  /** Custom identifier preservation instructions (used when identifierPolicy = "custom") */
  identifierInstructions?: string;

  /** Additional instructions appended to the summarization prompt */
  customInstructions?: string;

  /**
   * Max tokens per summarization chunk.
   * Default: derived from contextWindowTokens × adaptive chunk ratio
   */
  maxChunkTokens?: number;

  /**
   * Safety margin multiplier for token estimation (default: 1.2).
   * estimateTokens() uses chars/4 heuristic which underestimates for code/unicode.
   */
  safetyMargin?: number;

  /**
   * How many parallel chunks to summarize before merging (default: 2).
   * Increase for very long histories to speed up compaction.
   */
  parts?: number;

  /** AbortSignal passed through to summarize calls */
  signal?: AbortSignal;

  /**
   * Optional custom token counter. When provided, used instead of the
   * built-in chars/4 heuristic. Supply tiktoken or @anthropic-ai/tokenizer
   * for accurate counts.
   */
  tokenCounter?: (text: string) => number;

  /**
   * Additional field names to strip from tool_result blocks during summarization.
   * Default strip list: details, raw_response, stderr, debug_info, raw_output.
   */
  stripFields?: string[];

  /** Number of retry attempts for summarization calls (default: 3). */
  retryAttempts?: number;

  /** Minimum delay in ms for retry backoff (default: 500). */
  retryMinDelayMs?: number;

  /** Maximum delay in ms for retry backoff (default: 5000). */
  retryMaxDelayMs?: number;

  /** Called to report compaction progress. */
  onProgress?: (stage: string, current: number, total: number) => void;
};

/**
 * Result of a compaction operation.
 */
export type CompactResult = {
  /** true if compaction actually ran; false if not needed */
  compacted: boolean;
  /** The new message array to use (summary message + kept messages) */
  messages: Message[];
  /** The generated summary text */
  summary: string;
  /** Statistics about the compaction */
  stats: {
    tokensBefore: number;
    tokensAfter: number;
    messagesCompressed: number;
    messagesKept: number;
    droppedOrphanResults?: number;
    droppedOrphanUses?: number;
  };
};

/**
 * Result of shouldCompact() — stats without running compaction.
 */
export type CompactStats = {
  shouldCompact: boolean;
  estimatedTokens: number;
  threshold: number;
  messagesTotal: number;
  estimatedKeepFrom: number;
};
