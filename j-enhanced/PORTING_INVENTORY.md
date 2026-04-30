# Porting Inventory: What to Copy from Enhanced Repo to Original SeaKR

## Files to Copy (Copy As-Is)

### jacobian_plugin/ (Complete Directory)
```
jacobian_plugin/
├── __init__.py              ✓ Copy
├── types.py                 ✓ Copy
├── backend.py               ✓ Copy
├── scorer.py                ✓ Copy
├── policy.py                ✓ Copy
├── adapter.py               ✓ Copy
└── controller.py            ✓ Copy
```

**Dependencies**: None outside of Python stdlib + vllm + jacobian_plugin itself

**Note**: These files have no hard dependency on enhanced repo internals. They are clean and portable.

---

## Files to Modify

### SEAKR/utils.py
**Modifications**: Add 2 lines to `UncertaintyScore` dataclass
```python
jacobian_score: Optional[float] = None
jacobian_available: bool = False
```
**Minimal**: Yes  
**Reversible**: Yes (simple field additions)

### SEAKR/reasoner.py
**Modifications**:
1. Add import: `import jacobian_plugin`
2. Add 2 lines in `__init__` for plugin init
3. Replace `_should_trigger_retrieval` jacobian branch (4 lines → 4 lines)
4. Replace `_select_docs_with_temperature_scope` body (~40 lines)
5. Remove: `_get_temperature_scope_model_runner()` method (no longer used)
6. Remove: `_encode_without_special_tokens()` method (moved to plugin backend)

**Minimal**: Yes  
**Reversible**: Yes (can conditionally import/disable jacobian logic)

---

## Files NOT Needed

### vllm_uncertainty/
**Status**: ✗ Do NOT copy  
**Reason**: The enhanced vLLM is tied to your training/inference setup. The original SeaKR repo must use its own vLLM build or compatible fork.

**Migration note**: The plugin assumes a vLLM engine is available that:
- Populates `jacobian_score` and `jacobian_available` in uncertainty output
- Provides `model_runner.select_candidate_docs_by_temperature_scope()` method

If original SeaKR uses a different LLM framework or unmodified vLLM, backend implementation may need adjustment.

### main_simpleqa.py
**Status**: ✗ Do NOT modify  
**Reason**: Single-hop QA does not use Jacobian temperature-scope filtering (no `_select_docs_with_temperature_scope` method).

**Jacobian mode in SimpleQA**: Currently not implemented, and not required by original scope.

### main_multihop.py
**Status**: ✓ Keep as-is  
**Reason**: Already has all necessary CLI arguments. No code changes required.

---

## External Dependencies

The `jacobian_plugin` package requires:

### Python stdlib
- `abc` (abstract base classes)
- `typing` (type hints)

### Third-party
- `vllm` (AsyncLLMEngine, get_tokenizer)

**vLLM version**: Must support:
- `AsyncLLMEngine.get_tokenizer()` method
- `model_runner.select_candidate_docs_by_temperature_scope()` method
  - This is custom, added in the enhanced vLLM build
  - Standard vLLM does not have this method
  - Original SeaKR must use the modified vLLM fork or provide equivalent backend

---

## Summary

### To successfully port jacobian_plugin to original SeaKR:

1. **Copy** (9 files):
   - `jacobian_plugin/` (entire directory)

2. **Modify** (2 files, ~37 LOC total):
   - `SEAKR/utils.py` (+2 lines)
   - `SEAKR/reasoner.py` (~35 lines)

3. **Do NOT copy**:
   - `vllm_uncertainty/`
   - `main_simpleqa.py` (optional, but not needed for Jacobian integration)

4. **Backend assumption**:
   - Must be able to provide a modified vLLM (or alternative LLM backend) that supports:
     - Jacobian uncertainty scoring
     - `select_candidate_docs_by_temperature_scope()` method
   - If not available, Jacobian mode will not function, but plugin gracefully degrades

---

## Portability Summary

| Aspect | Status | Notes |
|--------|--------|-------|
| **Plugin core** | ✓ Portable | No hard dependencies on enhanced repo |
| **Backend** | ⚠ Coupled | Requires modified vLLM; can be swapped via JacobianBackend interface |
| **Host integration** | ✓ Minimal | Only 2 files, ~37 lines total |
| **CLI args** | ✓ Ready | Already in original repo |
| **Data structures** | ⚠ Coupled | UncertaintyScore must have jacobian fields |

The plugin is ready for porting with minimal friction. The main constraint is ensuring the original SeaKR has access to a Jacobian-enabled LLM backend.
