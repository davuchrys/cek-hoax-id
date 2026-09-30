# Cek Hoax ID

A small MLOps project: an Indonesian-language hoax/misinformation classifier, trained with scikit-learn and served as a containerized REST API with FastAPI and Docker, tested and built in CI.

> **Status:** v0.1 baseline. The bundled `data/sample.csv` is a tiny, hand-written demo set used for tests and the Docker image. Train on a real dataset (see below) before drawing any conclusions from the metrics.

## Architecture

```
data/*.csv ──> train.py ──> models/model.joblib ──> FastAPI (/predict) ──> Docker image
                  │                                        ▲
              metrics.json                        pytest + GitHub Actions CI
```

- **Model:** TF-IDF (unigrams + bigrams) + Logistic Regression, wrapped in a single `sklearn.Pipeline` so preprocessing ships with the model.
- **API:** FastAPI with input validation, a `/health` endpoint, and a clear `503` if no model is loaded.
- **Packaging:** the Docker image trains on the bundled sample at build time so it runs out of the box.

## Quick start

```bash
# Run with Docker
docker compose up --build

# Try it
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Sebarkan! Minum ramuan ini sembuhkan kanker dalam 3 hari"}'
# {"label":"hoax","hoax_probability":0.5704}  (low confidence: the demo model sees only 18 training rows)
```

Interactive docs: <http://localhost:8000/docs>

## Local development

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
export PYTHONPATH=src                                # Windows PowerShell: $env:PYTHONPATH="src"

python -m hoax_detector.train --data data/sample.csv --out models/model.joblib
uvicorn hoax_detector.api:app --reload
pytest
```

## API

| Method | Path       | Description                                        |
|--------|------------|----------------------------------------------------|
| GET    | `/health`  | Liveness + whether a model is loaded               |
| POST   | `/predict` | Body `{"text": "..."}` → `{label, hoax_probability}` |

## Data: automatic updates

`data/corpus.csv` grows over time from two RSS feeds (short excerpts only, deduplicated by hash):

| Source | Label | Content |
|--------|-------|---------|
| turnbackhoax.id | hoax (1) | fact-check articles about circulating hoaxes |
| antaranews.com | valid (0) | regular news |

```bash
python -m hoax_detector.data_update fetch                      # pull new items
python -m hoax_detector.train --data data/corpus.csv           # retrain
```

Locally or in Docker, set `AUTO_UPDATE_HOURS=24` (already the default in `docker-compose.yml`) and the API fetches new items, retrains and hot-swaps the model in the background, no restart needed.

The [update-data workflow](.github/workflows/update-data.yml) does both every Monday. The corpus is kept in the Actions cache (not in git) and the retrained model is uploaded as an artifact.

### Adding a public dataset

Merge any labelled CSV (`1` = hoax, `0` = valid) into the same corpus:

```bash
python -m hoax_detector.data_update import Train.csv --text-col text --label-col label --source hf-pauwdanny
```

Candidates (check each license first): [pauwdanny/indonesian_hoax_news_dataset](https://huggingface.co/datasets/pauwdanny/indonesian_hoax_news_dataset) (text is machine-translated English, not usable), [Rifky/indonesian-hoax-news](https://huggingface.co/datasets/Rifky/indonesian-hoax-news), [9uz/IDNHoaxCorpus](https://github.com/9uz/IDNHoaxCorpus), and [nlp-brin-id/id-hoax-report](https://huggingface.co/datasets/nlp-brin-id/id-hoax-report) (MIT, gated: request access).

### Known data biases

- Both sources are stripped of label giveaways (`[SALAH]`, `(ANTARA)`), but the model can still learn *source style* (fact-check phrasing vs. wire-news phrasing) instead of truthfulness.
- The hoax class is small at first (about 10 new items per fetch) and the training texts are debunk articles, not viral chain messages, so short WhatsApp-style text is out of distribution.
- Scraped excerpts may be copyrighted, so `data/corpus.csv` is git-ignored and never published. Rebuild it locally with the commands above.

## UI

Open <http://localhost:8000> for a small light-themed web UI (example texts, last five checks stored in the browser). It shows three outcomes instead of two: likely hoax (≥65%), likely valid (≤35%), and undecided in between.

## Roadmap

- [ ] Train and report metrics on a real, larger dataset with a proper held-out split
- [ ] Compare against an IndoBERT fine-tune (PyTorch) and document the trade-offs
- [ ] Model versioning and experiment tracking (MLflow)
- [ ] Drift monitoring on incoming requests
- [ ] JWT authentication on the API
- [ ] Deploy to a cloud runtime

## Limitations

A text classifier flags stylistic patterns of hoaxes, not factual truth. It should assist human fact-checkers, not replace them.

## License

MIT
