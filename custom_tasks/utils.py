import datasets
from functools import partial
from datasets import load_dataset
from lm_eval.tasks.winogrande.preprocess_winogrande import (
    doc_to_target as winogrande_doc_to_target,
    doc_to_text as winogrande_doc_to_text,
)
from lm_eval.tasks.hellaswag.utils import process_docs as hellaswag_process_docs
import bm25s
import Stemmer
import jinja2
jinja_env = jinja2.Environment()
from llama_index.core.node_parser import SentenceSplitter


dataset = load_dataset('hugo/news-corpus-1', 'corpus')['train']

corpus = [
    example['text']
    for example in dataset
    if example['year'] in ['2023', '2024']
]

splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
corpus_chunks = sum(
    [
        splitter.split_text(example["text"])
        for example in dataset
        if example["year"] in ["2023", "2024"]
    ],
    [],
)

stemmer = Stemmer.Stemmer("english")

corpus_tokens = bm25s.tokenize(corpus, stopwords='en', stemmer=stemmer)
retriever = bm25s.BM25()
retriever.index(corpus_tokens)

corpus_chunks_tokens = bm25s.tokenize(corpus_chunks, stopwords='en', stemmer=stemmer)
retriever_chunks = bm25s.BM25()
retriever_chunks.index(corpus_chunks_tokens)

prompt_template = '''Context:
{context}

{question}'''


def search(query, chunks=False):
    query_tokens = bm25s.tokenize(query, stemmer=stemmer)
    if chunks:
        results, scores = retriever.retrieve(query_tokens, k=5)
    else:
        results, scores = retriever.retrieve(query_tokens, k=1)
    return [corpus[doc_idx[0]] for doc_idx in results]


def doc_to_text(
    example: dict,
    question_field: str = None,
    doc_to_text_template: str = None,
    chunks: bool = False,
) -> str:
    if doc_to_text_template is not None:
        template = jinja_env.from_string(doc_to_text_template)
        question = template.render(**example)
    elif question_field is not None:
        question = example[question_field]
    else:
        raise ValueError('Either `doc_to_text_template` or `question_field` must be set.')

    retrieved_contexts = search(question, chunks)
    prompt = prompt_template.format(
        context='\n'.join(retrieved_contexts),
        question=question,
    )
    return prompt

def _winogrande_doc_to_choice(doc, chunk=False):
    idx = doc["sentence"].index("_")
    options = [doc["option1"], doc["option2"]]
    retrieved_contexts = search(doc['sentence'], chunk)
    prompt = prompt_template.format(
        context='\n'.join(retrieved_contexts),
        question='',
    )
    return [prompt + doc["sentence"][:idx] + opt for opt in options]


doc_to_text_openbookqa_top1 = partial(doc_to_text, question_field='question_stem')
doc_to_text_arc_top1 = partial(doc_to_text, doc_to_text_template="Question: {{question}}\nAnswer:")
doc_to_text_hellaswag_top1 = partial(doc_to_text, question_field='query')
doc_to_text_piqa_top1 = partial(doc_to_text, doc_to_text_template="Question: {{goal}}\nAnswer:")
doc_to_text_boolq_top1 = partial(doc_to_text, doc_to_text_template="{{passage}}\nQuestion: {{question}}?\nAnswer:")
winogrande_doc_to_choice_top1 = partial(_winogrande_doc_to_choice, chunk=False)

doc_to_text_openbookqa_top5 = partial(doc_to_text, question_field='question_stem')
doc_to_text_arc_top5 = partial(doc_to_text, doc_to_text_template="Question: {{question}}\nAnswer:")
doc_to_text_hellaswag_top5 = partial(doc_to_text, question_field='query')
doc_to_text_piqa_top5 = partial(doc_to_text, doc_to_text_template="Question: {{goal}}\nAnswer:")
doc_to_text_boolq_top5 = partial(doc_to_text, doc_to_text_template="{{passage}}\nQuestion: {{question}}?\nAnswer:")
winogrande_doc_to_choice_top5 = partial(_winogrande_doc_to_choice, chunk=True)
