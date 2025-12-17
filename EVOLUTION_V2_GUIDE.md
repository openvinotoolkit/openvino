# Evolution V2 - Usage Guide

## New Features

### 1. **evolve_gemma_v2.py** - Improved Evolution with Logging & Checkpointing

**Features:**
- ✅ Detailed logging to `evolution_progress.log`
- ✅ Periodic checkpoints every 10 iterations (saved to `evolution_checkpoints/`)
- ✅ JSON results tracking (`evolution_results_v2.json`)
- ✅ Memory-efficient: compiles models individually
- ✅ Graceful interrupt handling (Ctrl+C)
- ✅ Progress statistics (improvements vs failures)

**Usage:**
```powershell
. .\setup_dev_env.ps1
python evolve_gemma_v2.py
```

**Configuration** (edit in script):
- `MAX_ITERATIONS`: Number of evolution cycles (default: 1000)
- `MUTATION_RATE`: Percentage of neurons to mutate (default: 0.1 = 10%)
- `CHECKPOINT_INTERVAL`: Save every N iterations (default: 10)
- `DEVICE`: "GPU" or "CPU"

**Output Files:**
- `evolution_progress.log` - Timestamped log of all events
- `evolution_checkpoints/checkpoint_iter_N.xml` - Model snapshots
- `evolution_checkpoints/checkpoint_iter_N.json` - Metadata (MSE, timestamp, etc.)
- `evolution_results_v2.json` - Complete history (for plotting/analysis)
- `gemma_ir_tssn/evolved_checkpoint.xml` - Latest best model

**Resume from Checkpoint:**
To resume, manually load a checkpoint:
1. Copy `checkpoint_iter_N.xml` to `gemma_ir_tssn/openvino_model.xml`
2. Re-run the evolution script

---

### 2. **test_models_individually.py** - Memory-Efficient Testing

**Features:**
- ✅ Tests models ONE AT A TIME (avoids OOM errors)
- ✅ Loads, tests, unloads each model sequentially
- ✅ Comprehensive metrics: accuracy, performance, improvements
- ✅ JSON export of all results

**Usage:**
```powershell
. .\setup_dev_env.ps1
python test_models_individually.py
```

**What It Tests:**
1. **Teacher (Dense)** - Performance baseline
2. **Original TSSN** - Pre-evolution accuracy/speed
3. **Evolved TSSN** - Post-evolution comparison

**Output Metrics:**
- Performance: Latency, FPS, Speedup vs teacher
- Accuracy: MSE, MAE, Max Diff, Cosine Similarity
- Improvements: MSE change %, Performance change %

**Output Files:**
- `individual_test_results.json` - Complete results in JSON format

---

## Workflow

### Full Evolution + Testing:

```powershell
# 1. Setup environment
. .\setup_dev_env.ps1

# 2. Run evolution (can take hours for 1000 iterations)
python evolve_gemma_v2.py

# 3. Test the evolved model
python test_models_individually.py

# 4. Analyze results
Get-Content evolution_progress.log -Tail 50  # View last 50 log entries
Get-Content evolution_results_v2.json        # Full iteration history
Get-Content individual_test_results.json     # Final metrics
```

### Quick Test Only:

```powershell
. .\setup_dev_env.ps1
python test_models_individually.py
```

---

## Monitoring Evolution

### View Live Progress:
```powershell
Get-Content evolution_progress.log -Wait -Tail 20
```

### Check Latest Checkpoint:
```powershell
Get-ChildItem evolution_checkpoints | Sort-Object LastWriteTime -Descending | Select-Object -First 1
```

### Analyze Success Rate:
```powershell
# From evolution_results_v2.json, you can extract:
# - Total iterations
# - Number of improvements (kept=true)
# - Best MSE per iteration
# - Time per iteration
```

---

## Interpreting Results

### Evolution Log:
- `✅ IMPROVEMENT` - Mutation reduced MSE, was kept
- `❌ No improvement` - Mutation was reverted
- Success rate = improvements / total iterations

### Test Results:
- **MSE < 1e-6**: Excellent accuracy
- **MSE < 1e-3**: Good accuracy
- **MSE > 1e-3**: Poor accuracy (needs more evolution)
- **Speedup > 1.2x**: Worthwhile performance gain
- **Improvement % > 0**: Evolution helped!

---

## Troubleshooting

### Out of Memory:
- Reduce `SEQ_LEN` in test scripts (default: 32)
- Use CPU instead of GPU: Change `DEVICE = "CPU"`
- Reduce `NUM_ITERATIONS` in test (default: 20)

### Slow Evolution:
- Use GPU instead of CPU
- Reduce `MAX_ITERATIONS`
- Increase `MUTATION_RATE` for faster exploration (but less precise)

### No Improvements:
- Initial model may already be well-optimized
- Try increasing `MUTATION_RATE` (e.g., 0.2 = 20%)
- Let it run longer (1000+ iterations)

---

## File Structure

```
openvino/
├── evolve_gemma_v2.py                  # New evolution script
├── test_models_individually.py         # New memory-efficient testing
├── evolution_progress.log              # Generated: timestamped log
├── evolution_results_v2.json           # Generated: complete history
├── individual_test_results.json        # Generated: test results
├── evolution_checkpoints/              # Generated: model snapshots
│   ├── checkpoint_iter_10.xml
│   ├── checkpoint_iter_10.json
│   ├── checkpoint_iter_20.xml
│   └── ...
├── gemma_ir_tssn/
│   ├── openvino_model.xml             # Original TSSN model
│   └── evolved_checkpoint.xml         # Latest evolved model
└── gemma_ir/
    └── openvino_model.xml              # Teacher (dense) model
```

---

## Next Steps

1. **Run full evolution overnight**: `python evolve_gemma_v2.py`
2. **Test results in morning**: `python test_models_individually.py`
3. **Analyze logs**: Check `evolution_progress.log` for insights
4. **Iterate**: If improvements found, run more iterations
5. **Scale up**: Try on larger models (Gemma-2B, Gemma-7B)

---

## Advanced: Plotting Results

Use Python to visualize evolution:

```python
import json
import matplotlib.pyplot as plt

with open("evolution_results_v2.json") as f:
    data = json.load(f)

iterations = [r['iteration'] for r in data['iterations']]
mse_values = [r['best_mse'] for r in data['iterations']]

plt.plot(iterations, mse_values)
plt.xlabel("Iteration")
plt.ylabel("Best MSE")
plt.title("Evolution Progress")
plt.yscale('log')
plt.show()
```
