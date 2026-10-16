# Setup — run these on your MacBook, not in Claude's sandbox

## 1. Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install beir==2.0.0 --no-deps
```

## 2. Ollama (for local Qwen)
Make sure Ollama is installed and the model is pulled:
```bash
ollama pull qwen2.5:3.8b   # adjust the tag if your local model name differs -- run `ollama list` to check
```

## 3. Gemini (optional, for a spot-check subset later)
Get a free API key from https://aistudio.google.com/apikey and set it as an
environment variable before running any script that uses it:
```bash
export GEMINI_API_KEY="your-key-here"
```

## 4. Run in order
```bash
python 01_setup_and_index.py --dataset scifact     # downloads SciFact, builds BM25 + dense indexes (~2-5 min)
python 02_test_retrieval.py --dataset scifact      # sanity check: prints nDCG@10 for both retrievers, no reformulation
```

**Stop after step 2 and paste me the output** (especially the nDCG@10 numbers, and
the full traceback if anything errors) before running anything further --
this is the checkpoint where we catch dataset/library issues before building
the 4 actions on top.

### Fallback dataset (pre-committed)
If the full pilot on SciFact later shows little to no headroom (oracle gap
near zero, one action dominating everywhere), the pre-committed second
dataset is **TREC-COVID** -- chosen *before* seeing results, because its
queries are noisier and less well-formed than SciFact's, giving an a priori
reason to expect different reformulation behavior. To run it, just swap the
dataset flag:
```bash
python 01_setup_and_index.py --dataset trec-covid
python 02_test_retrieval.py --dataset trec-covid
```
Everything is cached separately per dataset, so this doesn't overwrite the
SciFact run -- both can exist side by side and be compared.

## Notes
- `bm25s` and `beir` versions can be picky about exact APIs; if you get an
  import error or a method-not-found error, paste the exact error and I'll
  adjust the code rather than guessing blind.
- If `pip install` fails on `beir` needing `pytrec_eval`, run:
  `pip install pytrec_eval-terrier` separately, then retry.
- Everything here runs on CPU/MPS automatically via sentence-transformers;
  you don't need to configure anything for Apple Silicon specifically.