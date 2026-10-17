import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'src'

"""Persistent, local sentence-transformer retrieval with exact source offsets."""
from .common import DATA, OUTPUT, save_json, read_json
import hashlib
import math
import time
from pathlib import Path
from langchain_text_splitters import Tokenizer, split_text_on_tokens

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
MODEL_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
QUESTIONS = [
    ('How long can thawed product stay refrigerated?', '30 days'),
    ('What dose requires secondary clinician review?', '100 micrograms'),
    ('What is an absolute contraindication involving a severe allergy?', 'anaphylaxis'),
    ('When must moderate or severe adverse events be documented?', '24 hours'),
    ('Can thawed vials be refrozen?', 'never be refrozen'),
]


def chunk_document(text, tokenizer, target=400, overlap=50):
    """Balance the tail so every chunk is 300–500 actual embedding tokens.

    Offsets retain verbatim source text. Adjacent chunks share 50 tokens.
    Short documents cannot satisfy the minimum and are rejected explicitly.
    """
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False, verbose=False)['offset_mapping']
    count = len(offsets)
    if count < 300:
        raise ValueError('Guideline source needs at least 300 tokens')
    n = max(1, math.ceil((count - overlap) / (target - overlap)))
    size = math.ceil((count + (n - 1) * overlap) / n)
    if size < 300 or size > 500:
        raise ValueError('Cannot balance chunks within 300–500 tokens')
    # LangChain performs the token-window split. Position IDs let the decoder
    # retain exact source punctuation/spacing instead of normalizing quotations.
    splitter = Tokenizer(chunk_overlap=overlap, tokens_per_chunk=size,
        encode=lambda _: list(range(count)),
        decode=lambda positions: text[offsets[positions[0]][0]:offsets[positions[-1]][1]])
    texts = split_text_on_tokens(text=text, tokenizer=splitter)
    chunks = []
    for i, content in enumerate(texts):
        start = i * (size - overlap)
        end = min(start + size, count)
        actual_count = len(tokenizer.encode(content, add_special_tokens=False))
        if not 300 <= actual_count <= 500:
            raise ValueError('A decoded chunk is outside the required 300–500-token range')
        chunks.append({'id': f'chunk-{i:03d}', 'text': content,
                       'start_token': start, 'end_token': end, 'token_count': actual_count,
                       'start_char': offsets[start][0], 'end_char': offsets[end - 1][1]})
    return chunks


class GuidelineIndex:
    def __init__(self, persist_dir=OUTPUT / 'chroma_store'):
        from sentence_transformers import SentenceTransformer
        import torch
        torch.set_num_threads(min(4, torch.get_num_threads()))
        import chromadb
        from chromadb.config import Settings
        self.persist_dir = Path(persist_dir)
        self.model = SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION, device='cpu')
        # MiniLM's default is 256; BERT positional capacity is 512. Avoid silent truncation.
        self.model.max_seq_length = 512
        self.client = chromadb.PersistentClient(path=str(self.persist_dir), settings=Settings(anonymized_telemetry=False))
        self.collection = None

    def build(self, doc_path=DATA / 'mrna_safety_guidelines.txt'):
        text = Path(doc_path).read_text(encoding='utf-8')
        digest = hashlib.sha256(text.encode()).hexdigest()
        chunks = chunk_document(text, self.model.tokenizer)
        name = 'mrna-' + digest[:16]
        self.collection = self.client.get_or_create_collection(name=name, embedding_function=None,
            metadata={'hnsw:space': 'cosine', 'source_sha256': digest, 'embedding_model': MODEL_NAME})
        stored = self.collection.get(include=['documents', 'metadatas'])
        metadata = [{k: v for k, v in c.items() if k not in ('id', 'text')} | {'source': Path(doc_path).name} for c in chunks]
        stored_text = dict(zip(stored['ids'], stored['documents']))
        if stored_text != {c['id']: c['text'] for c in chunks}:
            stale_ids = set(stored_text) - {c['id'] for c in chunks}
            if stale_ids:
                self.collection.delete(ids=sorted(stale_ids))
            vectors = self.model.encode([c['text'] for c in chunks], normalize_embeddings=True, show_progress_bar=False)
            self.collection.upsert(ids=[c['id'] for c in chunks], documents=[c['text'] for c in chunks],
                embeddings=vectors.tolist(), metadatas=metadata)
        elif dict(zip(stored['ids'], stored['metadatas'])) != {c['id']: m for c, m in zip(chunks, metadata)}:
            self.collection.update(ids=[c['id'] for c in chunks], metadatas=metadata)
        save_json(self.persist_dir / 'manifest.json', {'collection': name, 'model': MODEL_NAME, 'max_seq_length': 512,
                  'model_revision': MODEL_REVISION, 'source_sha256': digest, 'chunks': chunks})
        return self

    def load(self):
        manifest = read_json(self.persist_dir / 'manifest.json')
        if manifest['model'] != MODEL_NAME or manifest.get('model_revision') != MODEL_REVISION:
            raise ValueError('Embedding model mismatch; rebuild the index')
        self.collection = self.client.get_collection(manifest['collection'], embedding_function=None)
        return self

    def query(self, question, k=3):
        if not isinstance(question, str) or not question.strip():
            raise ValueError('Question must be nonempty')
        if self.collection is None:
            raise RuntimeError('Build or load the index before querying')
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= self.collection.count():
            raise ValueError('k must be between 1 and the number of indexed chunks')
        start = time.perf_counter()
        vector = self.model.encode([question], normalize_embeddings=True, show_progress_bar=False)
        result = self.collection.query(query_embeddings=vector.tolist(), n_results=k, include=['documents', 'distances', 'metadatas'])
        elapsed = time.perf_counter() - start
        hits = [{'id': cid, 'text': doc, 'distance': float(distance), 'metadata': meta} for cid, doc, distance, meta in
                zip(result['ids'][0], result['documents'][0], result['distances'][0], result['metadatas'][0])]
        return {'question': question, 'k': k, 'latency_seconds': elapsed, 'hits': hits}


def build_index(doc_path=DATA / 'mrna_safety_guidelines.txt'):
    return GuidelineIndex().build(doc_path)


def query_guidelines(vectorstore, question, k=3):
    result = vectorstore.query(question, k)
    return [hit['text'] for hit in result['hits']], result['latency_seconds']


def evaluate(index):
    results = []
    for question, expected in QUESTIONS:
        result = index.query(question)
        result['expected_phrase'] = expected
        result['top1_contains_expected'] = expected in result['hits'][0]['text']
        result['top3_contains_expected'] = any(expected in h['text'] for h in result['hits'])
        results.append(result)
    # Deliberately unsupported query demonstrates that similarity search still returns hits.
    unsupported = index.query('What is the vaccine efficacy percentage against malaria?')
    unsupported['supported_by_source'] = False
    lines = ['# Retrieval evaluation', '', '| Question | Top result | Seconds |', '|---|---|---:|']
    for r in results:
        cid = r['hits'][0]['id']
        lines.append(f"| {r['question']} | [{cid}](#{cid}) | {r['latency_seconds']:.4f} |")
    lines += ['', '## Retrieval quality', '',
        'The questions cover storage, dose review, contraindications and adverse-event reporting. The source excerpts returned for each question are reproduced below. Broad chunks can contain multiple adjacent topics, so relevance must be checked against the question.', '',
        '## Unsupported query', unsupported['question'], '',
        f"The top result was {unsupported['hits'][0]['id']} (query time {unsupported['latency_seconds']:.4f} seconds). The source supplies no malaria efficacy percentage. Nearest-neighbor search ranks available text even when it cannot answer a question; similarity is not proof of evidence coverage.", '',
        '## Retrieved source chunks']
    seen = set()
    for r in results + [unsupported]:
        hit = r['hits'][0]
        if hit['id'] not in seen:
            seen.add(hit['id'])
            lines += ['', '### ' + hit['id'], '', hit['text']]
    (OUTPUT / 'retrieval_evaluation.md').write_text('\n'.join(lines), encoding='utf-8')
    return results


if __name__ == '__main__':
    index = build_index()
    evaluate(index)
    print('Persisted chunks:', index.collection.count())
