# Post-Porting Validation Checklist

Use this checklist to verify the Jacobian plugin is correctly integrated into the original SeaKR repo.

## Pre-Integration Verification

- [ ] **jacobian_plugin/ copied** to repo root
  - [ ] `jacobian_plugin/__init__.py` exists
  - [ ] `jacobian_plugin/backend.py` exists
  - [ ] `jacobian_plugin/types.py` exists
  - [ ] All 6 files present (scorer, policy, adapter, controller, types, backend)

- [ ] **SEAKR/utils.py modified**
  - [ ] `UncertaintyScore` has `jacobian_score` field
  - [ ] `UncertaintyScore` has `jacobian_available` field
  - [ ] Both are typed as `Optional[float]` and `bool` respectively
  - [ ] No syntax errors: `python -c "from SEAKR.utils import UncertaintyScore"`

- [ ] **SEAKR/reasoner.py modified**
  - [ ] Has `import jacobian_plugin` at top
  - [ ] `MultiHopReasoner.__init__` creates `self.jacobian_controller`
  - [ ] `MultiHopReasoner.__init__` creates `self.jacobian_adapter`
  - [ ] `_should_trigger_retrieval` has jacobian branch calling `jacobian_controller.decide_retrieval()`
  - [ ] `_select_docs_with_temperature_scope` uses `jacobian_adapter.prepare_jacobian_input()`
  - [ ] `_select_docs_with_temperature_scope` uses `jacobian_controller.select_documents()`
  - [ ] Old methods removed: `_get_temperature_scope_model_runner()`, `_encode_without_special_tokens()`
  - [ ] No syntax errors: `python -c "from SEAKR.reasoner import MultiHopReasoner"`

## Import Verification

```bash
# Check all imports resolve
python -c "import jacobian_plugin; print('✓ jacobian_plugin imports OK')"
python -c "from jacobian_plugin import JacobianController; print('✓ JacobianController imports OK')"
python -c "from jacobian_plugin import VLLMJacobianBackend; print('✓ VLLMJacobianBackend imports OK')"
python -c "from SEAKR.reasoner import MultiHopReasoner; print('✓ MultiHopReasoner imports OK')"
python -c "from SEAKR.utils import UncertaintyScore; print('✓ UncertaintyScore imports OK')"
```

## Runtime Verification (Minimal)

### Test 1: Jacobian Mode Initialization
```python
from vllm import AsyncLLMEngine, AsyncEngineArgs
from SEAKR.reasoner import MultiHopReasoner
from SEAKR.dataset import get_dataset
from SEAKR.retriever import BM25

# Mock or real engine setup (not shown)
# engine = AsyncLLMEngine.from_engine_args(args)
# retriever = BM25(...)
# dataset = get_dataset(...)

# Test: Can instantiate with jacobian mode
reasoner = MultiHopReasoner(
    qid="test_001",
    question="What is the capital of France?",
    dataset=dataset,
    llm_engine=engine,
    retriever=retriever,
    decision_mode="jacobian",
    jacobian_threshold=0.5,
    compute_jacobian=False
)

assert reasoner.jacobian_controller is not None
assert reasoner.jacobian_adapter is not None
print("✓ Jacobian mode initialization OK")
```

### Test 2: CLI Argument Parsing
```bash
python main_multihop.py --help | grep -E "decision_mode|jacobian_threshold|jacobian_doc"
# Should show:
# --decision_mode {eigen,jacobian,hybrid}
# --jacobian_threshold JACOBIAN_THRESHOLD
# --jacobian_doc_topk JACOBIAN_DOC_TOPK
# --jacobian_doc_token_topk JACOBIAN_DOC_TOKEN_TOPK
```

### Test 3: Dry-run with Jacobian Mode
```bash
python main_multihop.py \
  --dataset_name hotpotqa \
  --retriever_port 9200 \
  --model_name_or_path meta-llama/Llama-2-7b-chat \
  --served_model_name llama2-7b \
  --decision_mode jacobian \
  --jacobian_threshold 0.5 \
  --max_workers 1 \
  --save_dir ./test_run \
  2>&1 | head -50
# Should start without import/syntax errors
```

## Behavioral Verification (Optional, if you have a working vLLM setup)

- [ ] **Jacobian threshold triggers retrieval**
  - Run a question with `--decision_mode jacobian --jacobian_threshold 0.5`
  - Verify retrieval is triggered when `jacobian_score > 0.5`
  - Check logs for "Temperature-scope pre-selected docs"

- [ ] **Fallback works when Jacobian unavailable**
  - If model runner doesn't have `select_candidate_docs_by_temperature_scope()`, plugin should:
    - Log: "Temperature-scope selector unavailable"
    - Return None from adapter
    - Fall back to original candidate-doc path
  - Verify behavior in logs

- [ ] **Eigen mode still works**
  - Run with `--decision_mode eigen`
  - Verify retrieval still works using eigen score threshold
  - Verify Jacobian plugin doesn't interfere

## Troubleshooting

### ImportError: No module named 'jacobian_plugin'
- Verify `jacobian_plugin/` is in repo root, not nested in SEAKR/
- Check `jacobian_plugin/__init__.py` exists

### AttributeError: 'MultiHopReasoner' object has no attribute 'jacobian_controller'
- Verify `__init__` method has the 2 plugin initialization lines
- Check for indentation errors
- Run: `python -c "from SEAKR.reasoner import MultiHopReasoner; r = MultiHopReasoner(...); print(hasattr(r, 'jacobian_controller'))"`

### decision_mode 'jacobian' not recognized
- Check `_validate_decision_mode()` still has `if self.decision_mode == "jacobian": return` (no NotImplementedError)

### "Temperature-scope selector unavailable"
- This is **expected behavior** if vLLM backend doesn't have Jacobian support
- Plugin will fall back gracefully
- To enable: Ensure using modified vLLM fork with Jacobian implementation

### "Jacobian score not available" in logs
- Verify `--compute_jacobian` flag is passed to engine (if required by backend)
- Verify vLLM is actually computing and populating `jacobian_score`

## Post-Porting Regression Tests

Run existing SeaKR tests (if available):

```bash
# Example: Run evaluation on a small dataset
python eval_multihop.py \
  --results_path ./test_run/results.jsonl \
  --dataset_name hotpotqa
```

Expected: All metrics should be similar to `--decision_mode eigen` baseline (if jacobian threshold is set high to rarely trigger).

## Documentation

- [ ] `INTEGRATE_JACOBIAN_PLUGIN.md` is in repo root (reference for maintainers)
- [ ] `PATCH_REFERENCE.md` is in repo root (exact changes made)
- [ ] `PORTING_INVENTORY.md` is in repo root (what was copied/modified)

---

## Sign-Off

If all checks pass:
- [ ] **Integration complete and verified**
- [ ] **jacobian_plugin is production-ready in original SeaKR**
- [ ] **Backward compatibility maintained** (eigen mode unaffected)
- [ ] **Ready for Jacobian-based retrieval experiments**
