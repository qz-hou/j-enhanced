# Jacobian Plugin Integration Guide

This document describes how to integrate the `jacobian_plugin` module into the original (unmodified) SeaKR repository.

## Files to Copy Directly

Copy the entire directory as-is:
- `jacobian_plugin/` (all files)

## Files Requiring Modifications

### 1. `SEAKR/utils.py`

Add two fields to the `UncertaintyScore` dataclass to support Jacobian scores from vLLM:

```python
@dataclass
class UncertaintyScore:
    logprobs: List[Tuple[int, float]]
    perplexity: float
    ln_entropy: float
    energy_score: float
    eigen_score: float
    jacobian_score: Optional[float] = None          # ADD THIS LINE
    jacobian_available: bool = False                # ADD THIS LINE
```

**Why**: The LLM output from modified vLLM populates these fields. The plugin and host use them to gate retrieval decisions.

### 2. `SEAKR/reasoner.py`

#### 2a. Add import (line 5, after other SEAKR imports)

```python
import jacobian_plugin
```

#### 2b. Initialize plugin in `MultiHopReasoner.__init__` (after line 44, before `self.sentence_solver = "max"`)

```python
        # Initialize Jacobian plugin
        self.jacobian_controller = jacobian_plugin.JacobianController(self.llm_engine, self.jacobian_threshold)
        self.jacobian_adapter = jacobian_plugin.SeaKRAdapter(self.llm_engine)
```

**Why**: These objects coordinate Jacobian-specific retrieval logic and state encoding.

#### 2c. Replace retrieval decision logic in `_should_trigger_retrieval`

In the `jacobian` branch, replace:
```python
if self.decision_mode == "jacobian":
    return (uncertainty.jacobian_available and
            uncertainty.jacobian_score is not None and
            uncertainty.jacobian_score > self.jacobian_threshold)
```

With:
```python
if self.decision_mode == "jacobian":
    decision = self.jacobian_controller.decide_retrieval(
        uncertainty.jacobian_score, uncertainty.jacobian_available
    )
    return decision.should_retrieve
```

**Why**: Centralizes Jacobian retrieval decision logic into the plugin.

#### 2d. Replace document selection in `_select_docs_with_temperature_scope` (lines 300-340 approx.)

Replace the entire body of `_select_docs_with_temperature_scope` with:

```python
    async def _select_docs_with_temperature_scope(
            self,
            candidate_doc_ids: List[str],
            candidate_docs: List[str]) -> Optional[Tuple[List[str], List[str]]]:
        if not candidate_doc_ids or not candidate_docs:
            return None

        try:
            # Prepare prefix prompt using the same logic as LLM calls
            prefix_prompt, _ = self.prepare_llm_input(
                question=self.question,
                cot_step=self.running_steps,
                docs=self.docs,
            )

            # Prepare Jacobian input using the adapter
            jacobian_input = await self.jacobian_adapter.prepare_jacobian_input(
                prefix_prompt=prefix_prompt,
                candidate_doc_ids=candidate_doc_ids,
                candidate_docs=candidate_docs,
                num_select_docs=self.jacobian_doc_topk,
                topk_doc_tokens=self.jacobian_doc_token_topk,
                request_id=f"{self.qid}_{self.llm_call_times}_docscope",
            )

            # Use controller to select documents
            selected = await self.jacobian_controller.select_documents(jacobian_input)
            if selected is not None:
                selected_doc_ids, selected_docs = selected
                self.logger.debug(
                    "Temperature-scope pre-selected docs: %s",
                    selected_doc_ids)
                return selected_doc_ids, selected_docs
            else:
                self.logger.debug(
                    "Temperature-scope pre-selection returned no candidate docs; "
                    "fallback to original candidate-doc generation path.")
                return None
        except Exception as exc:
            self.logger.debug(
                "Temperature-scope pre-selection failed; fallback to original "
                "candidate-doc generation path. Error: %s", exc)
            return None
```

**Why**: Replaces direct vLLM backend access with plugin abstraction. All vLLM-specific logic is now contained in the plugin backend.

## CLI Arguments

No changes required. The original `main_multihop.py` already supports:
- `--jacobian_threshold` (passed to `MultiHopReasoner`)
- `--jacobian_doc_topk` (passed to `MultiHopReasoner`)
- `--jacobian_doc_token_topk` (passed to `MultiHopReasoner`)
- `--decision_mode` (already supports `"jacobian"` mode)
- `--compute_jacobian` (already exists)

## Backend Assumptions

The plugin assumes:
1. **Modified vLLM runtime**: The LLM engine must provide `jacobian_score` and `jacobian_available` in the uncertainty output.
2. **Temperature-scope method**: The vLLM model runner must support `select_candidate_docs_by_temperature_scope(...)` method.
3. **Tokenizer API**: The vLLM engine must expose `get_tokenizer()` method.

## Integration Checklist

- [ ] Copy `jacobian_plugin/` to root of original SeaKR repo
- [ ] Update `SEAKR/utils.py`: Add `jacobian_score` and `jacobian_available` to `UncertaintyScore`
- [ ] Update `SEAKR/reasoner.py`:
  - [ ] Add `import jacobian_plugin`
  - [ ] Initialize controller and adapter in `__init__`
  - [ ] Replace `_should_trigger_retrieval` jacobian branch
  - [ ] Replace `_select_docs_with_temperature_scope` body
- [ ] Verify no additional changes needed in `main_multihop.py`
- [ ] Test with `--decision_mode jacobian --jacobian_threshold <threshold>`

## Remaining Coupling Points

After this integration, the following still require the modified vLLM backend:

1. **Jacobian score extraction in `call_llm` method**
   - The greedy output from `call_llm` must contain `jacobian_score` and `jacobian_available` fields.
   - This requires the modified vLLM engine that computes Jacobian uncertainty on the greedy path.
   - To use with standard vLLM: implement a different backend or disable jacobian mode.

2. **UncertaintyScore dataclass shape**
   - The plugin does not create `UncertaintyScore` objects; it only consumes them.
   - The host is responsible for populating these fields from LLM output.

3. **Model runner internals**
   - `VLLMJacobianBackend._get_temperature_scope_model_runner()` accesses vLLM's private internals.
   - If vLLM's internal structure changes, this method must be updated.

## Future Decoupling (Optional)

To make the plugin fully portable to non-vLLM backends:

1. **Abstract uncertainty output**: Create an interface for uncertainty scores instead of requiring `UncertaintyScore` with jacobian fields.
2. **Backend factory**: Provide a way for host code to inject a custom `JacobianBackend` implementation at initialization time.
3. **Configuration object**: Move hyperparameters into a config object instead of spreading them as constructor arguments.

For now, the current design keeps the host-side changes minimal while isolating all vLLM-specific logic in the backend.
