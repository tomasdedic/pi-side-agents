/**
 * Query Separator Extension
 *
 * Injects a visual separator before each assistant response (after the first query).
 * The separator is display-only in TUI; the LLM receives a single space as content
 * which it ignores.
 *
 * Change SEP_TEXT below to use a different separator string.
 */

import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { Text } from "@mariozechner/pi-tui";

const SEP_TEXT = ">>>";

export default function (pi: ExtensionAPI) {
  let queryCount = 0;

  pi.registerMessageRenderer("query-separator", (_message, _opts, theme) => {
    return new Text(theme.fg("mdHr", SEP_TEXT), 0, 0);
  });

  pi.on("agent_end", async () => {
    queryCount++;
  });

  pi.on("before_agent_start", async () => {
    if (queryCount === 0) return;
    return {
      message: {
        customType: "query-separator",
        content: " ",
        display: true,
      },
    };
  });
}
