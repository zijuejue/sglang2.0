# SGLang Prefix Reuse Triggering Conditions (Research Notes)

## Scope and branch status
- Repository branch inspected: `work`.
- Attempted to pull latest changes with `git pull --ff-only`, but this branch has no upstream tracking configured in this environment.

---

## TL;DR — when does prefix reuse trigger?
Prefix reuse (KV cache prefix hit) is triggered only when all of the following are true:

1. **Prefix cache is enabled** (`--disable-radix-cache` is not set).
2. **There is a non-empty prompt prefix to match** (after internal truncation rules).
3. **The request falls in the same cache namespace (`extra_key`)** as existing cache entries.
4. **The token prefix matches an existing radix path** (subject to page-size alignment).
5. **Special request modes do not bypass/replace radix lookup** (e.g., streaming session restore path).


## Source code walk-through

### 1) Backend selection + global on/off behavior
In scheduler cache initialization, SGLang builds cache backends from server args:
- `disable=server_args.disable_radix_cache` is propagated into cache init params.
- If chunked prefill is enabled *and* radix is disabled, SGLang uses `ChunkCache`/`SWAChunkCache` instead of radix-tree prefix matching.
- Otherwise it selects `RadixCache` (or variants like C++/HiRadix/Mamba/SWA/LMCache wrappers).

**Source**: `python/sglang/srt/managers/scheduler.py`

### 2) How a request computes the candidate prefix length
Before matching, request preparation computes `fill_ids` and then limits match length:
- `max_prefix_len = input_len - 1` (keeps last token uncached for logprob correctness).
- If `return_logprob` with `logprob_start_len >= 0`, match length is additionally capped.
- Only `fill_ids[:max_prefix_len]` is used for `match_prefix`.

So even with exact same full prompt, effective reusable prefix is usually **at most prompt_len - 1**.

**Source**: `python/sglang/srt/managers/schedule_batch.py` (`Req.init_next_round_input`).

### 3) Exact radix match conditions
`RadixCache.match_prefix` applies key rules:
- Returns empty when cache is disabled or key length is zero.
- If `page_size != 1`, key length is truncated to page-aligned length before matching.
- Matching is over `(token_ids, extra_key)` namespace.

**Source**: `python/sglang/srt/mem_cache/radix_cache.py` (`match_prefix`).

### 4) `extra_key` / `cache_salt` namespace isolation
Radix matching enforces isolation by key namespace:
- Internal key comparison checks `key0.extra_key == key1.extra_key`.
- Child routing key includes `extra_key`, so identical token prefixes in different namespaces do not share nodes.

**Source**: `python/sglang/srt/mem_cache/radix_cache.py` (`_check_extra_key`, `get_child_key`).

For OpenAI-compatible APIs, final namespace key is built by concatenating request `cache_salt` and `extra_key`:
- same text + same salt => shareable namespace.
- same text + different salt => isolated namespace (no reuse).

**Source**: `python/sglang/srt/entrypoints/openai/serving_base.py` (`_compute_extra_key`).

### 5) Streaming session mode uses session slot state first
With session-aware cache wrapper:
- non-streaming requests call regular radix `match_prefix`.
- streaming requests with an existing session slot restore committed KV directly and return those indices.

This means session continuation can reuse via session slot state rather than radix-tree traversal.

**Source**: `python/sglang/srt/mem_cache/session_aware_cache.py` (`match_prefix`).

### 6) Input embeddings path constraints
If `input_embeds` is provided while radix cache is enabled, SGLang raises an error and asks to run with `--disable-radix-cache`.

This path effectively disables normal prefix reuse for such inputs unless cache is globally disabled and alternate behavior is used.

**Source**: `python/sglang/srt/managers/tokenizer_manager.py` (`_tokenize_one_request`).

---

## Corresponding test scripts and what they verify

### A) Core radix behavior (unit)
1. **Basic hit/miss and disabled cache**
   - `test_insert_and_match_basic` verifies:
     - when cache enabled: inserted `[1,2,3]` can be matched as full and partial prefix.
     - when cache disabled: no effective insertion/match benefit.

2. **extra_key isolation**
   - `test_extra_key_isolation` inserts identical token ids with different `extra_key` values and verifies matches are isolated.

3. **Page alignment boundaries**
   - `test_page_alignment_boundary` verifies match length is page-aligned when `page_size > 1`.

**Test file**: `test/registered/radix_cache/test_radix_cache_unit.py`

### B) OpenAI API `cache_salt` behavior
- `test_cache_salt_effectiveness` checks:
  - second request with same `cache_salt` gets cache hit,
  - request with different `cache_salt` does not share cache,
  - matching same new salt again then gets hit.

**Test file**: `test/manual/openai_server/features/test_cache_report.py`

### C) Branching-prefix behavior on long prompts
- `test_prefix_cache_branching` demonstrates a subtle trigger pattern:
  - request 1 warms full sequence,
  - request 2 with different branch still no hit,
  - request 3 then hits at branching boundary.
- Expected hit is page/chunk aligned (`branching_pos // 64 * 64`).

**Test file**: `test/registered/4-gpu-models/test_qwen3_next_models.py`

### D) End-to-end multi-turn hit-rate validation
- `run_multiturn_cache_hit_test`:
  - flushes cache,
  - sends barrier-synchronized rounds,
  - computes expected per-round cached token behavior using server page size and prior prompts.

**Used by**: `test/registered/radix_cache/test_radix_cache_hit.py`
**Kit implementation**: `python/sglang/test/kits/cache_hit_kit.py`

---

## Practical “trigger checklist” for debugging
If you expected a cache hit but got miss, check in order:

1. Did you accidentally set `--disable-radix-cache`?
2. Is your effective matched prefix non-empty after `input_len - 1` rule?
3. Are `cache_salt`/`extra_key` identical between requests?
4. Are you crossing page-size boundaries (especially with short common prefix)?
5. Is this a streaming session request (session slot logic may apply)?
6. Are you using `input_embeds` (requires radix disabled)?

---

## Suggested minimal reproduction commands (for local validation)
```bash
# 1) Launch server with radix cache enabled (default)
python -m sglang.launch_server --model-path <model>

# 2) Send same prompt twice and inspect meta_info.cached_tokens
curl -s http://127.0.0.1:30000/generate -H 'Content-Type: application/json' -d '{"text":"Hello world", "sampling_params":{"max_new_tokens":8}}'

# 3) Send same prompt with different cache_salt via OpenAI API to verify isolation
# (depends on your OpenAI-compatible client setup)
```
