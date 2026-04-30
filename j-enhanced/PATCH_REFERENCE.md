# Quick Patch Reference: Porting jacobian_plugin to Original SeaKR

## Step-by-Step Patch Summary

### Step 1: Copy jacobian_plugin directory
```
Copy: jacobian_plugin/ → original_seakr/jacobian_plugin/
```

### Step 2: SEAKR/utils.py

**Location**: After the `eigen_score: float` field in `UncertaintyScore` dataclass

**Add**:
```python
jacobian_score: Optional[float] = None
jacobian_available: bool = False
```

**Full context**:
```python
@dataclass
class UncertaintyScore:
    logprobs: List[Tuple[int, float]]
    perplexity: float
    ln_entropy: float
    energy_score: float
    eigen_score: float
    jacobian_score: Optional[float] = None           # ← ADD
    jacobian_available: bool = False                 # ← ADD
```

**Import needed**: Already present (`Optional` is used)

---

### Step 3: SEAKR/reasoner.py

#### 3a. Add import
**Location**: After existing SEAKR imports (around line 4-5)

**Add**:
```python
import jacobian_plugin
```

**Full context**:
```python
from typing import Union, List, Tuple, Optional
from .utils import LLMOutputWithUncertainty, Step, UncertaintyScore, StepStatus
from .retriever import BM25
from .dataset import Dataset
import jacobian_plugin                               # ← ADD
import logging
```

#### 3b. Initialize plugin in __init__
**Location**: Inside `MultiHopReasoner.__init__`, after `self.sentence_solver = "max"` (around line 45)

**Add**:
```python
        # Initialize Jacobian plugin
        self.jacobian_controller = jacobian_plugin.JacobianController(self.llm_engine, self.jacobian_threshold)
        self.jacobian_adapter = jacobian_plugin.SeaKRAdapter(self.llm_engine)
```

**Full context**:
```python
        self.doc_id_list = []
        self.docs = []
        self.final_step_answer = None
        self.final_read_answer = None
        self.sentence_solver = "max"

        # Initialize Jacobian plugin                 # ← ADD
        self.jacobian_controller = jacobian_plugin.JacobianController(self.llm_engine, self.jacobian_threshold)  # ← ADD
        self.jacobian_adapter = jacobian_plugin.SeaKRAdapter(self.llm_engine)  # ← ADD
```

#### 3c. Replace _should_trigger_retrieval jacobian branch
**Location**: In `_should_trigger_retrieval()` method

**Remove**:
```python
        if self.decision_mode == "jacobian":
            return (uncertainty.jacobian_available and
                    uncertainty.jacobian_score is not None and
                    uncertainty.jacobian_score > self.jacobian_threshold)
```

**Replace with**:
```python
        if self.decision_mode == "jacobian":
            decision = self.jacobian_controller.decide_retrieval(
                uncertainty.jacobian_score, uncertainty.jacobian_available
            )
            return decision.should_retrieve
```

**Full context**:
```python
    def _should_trigger_retrieval(self, uncertainty: UncertaintyScore) -> bool:
        if self.decision_mode == "eigen":
            return uncertainty.eigen_score > self.eigen_threshold
        if self.decision_mode == "jacobian":
            decision = self.jacobian_controller.decide_retrieval(  # ← REPLACE
                uncertainty.jacobian_score, uncertainty.jacobian_available
            )
            return decision.should_retrieve                       # ← REPLACE
        if self.decision_mode == "hybrid":
            raise NotImplementedError(...)
```

#### 3d. Replace _select_docs_with_temperature_scope method body
**Location**: Replace entire method body (not just lines, but entire function)

**Method signature** (keep as-is):
```python
    async def _select_docs_with_temperature_scope(
            self,
            candidate_doc_ids: List[str],
            candidate_docs: List[str]) -> Optional[Tuple[List[str], List[str]]]:
```

**New body**:
```python
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

**What to remove** (delete these old methods entirely if they exist in original):
- `_get_temperature_scope_model_runner()` 
- `_encode_without_special_tokens()`

These are now encapsulated in the plugin backend.

---

### Step 4: main_multihop.py

**Status**: No changes needed.

The file already supports all required CLI arguments:
- `--decision_mode jacobian`
- `--jacobian_threshold`
- `--jacobian_doc_topk`
- `--jacobian_doc_token_topk`
- `--compute_jacobian`

---

## Testing After Patch

Run with Jacobian mode:
```bash
python main_multihop.py \
  --dataset_name hotpotqa \
  --retriever_port 9200 \
  --model_name_or_path meta-llama/Llama-2-7b-chat \
  --served_model_name llama2-7b \
  --decision_mode jacobian \
  --jacobian_threshold 0.5 \
  --jacobian_doc_topk 3 \
  --jacobian_doc_token_topk 8 \
  --save_dir ./runs/test_jacobian
```

Expected behavior:
- Retrieval is triggered based on `jacobian_score > jacobian_threshold`
- Retrieved documents are pre-filtered using temperature-scope method
- Fallback to full candidate set if temperature-scope unavailable

---

## Files Summary

| File | Action | Reason |
|------|--------|--------|
| `jacobian_plugin/*` | Copy entire dir | Plugin code (backend-agnostic core, vLLM backend impl) |
| `SEAKR/utils.py` | Add 2 lines | Extend UncertaintyScore dataclass |
| `SEAKR/reasoner.py` | Add 3 sections | Import plugin, init controller/adapter, use plugin APIs |
| `main_multihop.py` | No change | Already supports all jacobian args |

## Total Host-Side LOC Changes

- `SEAKR/utils.py`: 2 lines added
- `SEAKR/reasoner.py`: ~35 lines modified (import, init, 2 method implementations)
- **Total**: ~37 lines in host code, ~95 lines of pure plugin code

This is a minimal, surgical integration that keeps the host framework intact.
