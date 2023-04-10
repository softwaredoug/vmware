from dotenv import load_dotenv
import json
import os
import openai
import csv
from sys import argv

load_dotenv()
openai.api_key = os.getenv('OPEN_AI_KEY')
model = "gpt-3.5-turbo"


def prompt1():
    prompt = """
    Within VMWare technical documentation or blog posts, generate an article body for the query "{query}". Include the title and body of the article.
    """
    return prompt


def prompt2():
    prompt = """
    Given VMWare question answering forum, etc - please generate a forum question (ie stackoverflow) that solves the problem behind the question "{query}".  Please feel free to liberally misspell and give common, alternate spellings of the topics in the question (ie vmware -> VM Ware, elasticsearch -> elastic search, etc)

    Please don't include code in the response.
    """
    return prompt


def prompt3():
    prompt = """
In search relevance, users type in search queries that don't match the vocabulary used in the underlying corpus. This is known as the 'vocabulary problem'. Some important reasons for the vocabulary problem - the author of the article in the corpus does not use the same terminology as the searcher.

Given a corpus of VMWare technical documentation, blog posts, and forum posts, please propose some alternate, equivalent synonyms, alternate forms, similar terms, tags, and other possible expansions for the search query below. Note it's very important to come up with synonyms that expand one word to multiple words or vice-versa.

Please list the response as a numbered list with no explanation. Here's the query:

> {query}
    """
    return prompt


def understand_nl_query(prompt, query):
    prompt = prompt.format(query=query)

    completion = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )
    response = completion.choices[0].message.content
    return response


def queries_to_articles(prompt, reader, cache):
    for row in reader:
        query = row['Query']
        query_id = int(row['QueryId'])
        if query in cache.keys():
            print(f"Found {query} in cache")
            yield query_id, query, cache[query]['article']
        else:
            yield query_id, query, understand_nl_query(prompt, query)


def load_query_database(filename='query_database.1.json'):
    try:
        return json.load(open(filename, 'r'))
    except FileNotFoundError:
        return {}


def articles_for_queries(filename: str):
    query_database = load_query_database(filename)
    if 'prompt' not in query_database.keys():
        query_database['prompt'] = prompt3()
    prompt = query_database['prompt']
    if prompt != prompt2():
        print("Prompt has changed. Should you reprocess all queries?")

    if 'questions' not in query_database.keys():
        query_database['questions'] = {}

    for key in query_database.keys():
        if key not in ['questions', 'prompt']:
            raise ValueError(f"Key {key} is not a valid key")

    with open('data/test.csv', 'r') as f:
        reader = csv.DictReader(f)
        for query_id, query, article in queries_to_articles(prompt, reader, query_database['questions']):
            print("----")
            print(query)
            print(article)
            if query not in query_database['questions'].keys():
                query_database['questions'][query] = {'article': article,
                                                      'query_id': query_id}
                if query_id % 10 == 0:
                    print(f"Completed {query_id} queries")
                    json.dump(query_database, open(filename, 'w'))
        json.dump(query_database, open(filename, 'w'))


if __name__ == '__main__':
    articles_for_queries(argv[1])
