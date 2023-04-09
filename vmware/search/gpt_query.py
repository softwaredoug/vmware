from dotenv import load_dotenv
import json
import os
import openai
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


def queries_to_articles(prompt, queries):
    for query in queries:
        yield query, understand_nl_query(prompt, query)


def load_query_database(filename='query_database.1.json'):
    try:
        return json.load(open(filename, 'r'))
    except FileNotFoundError:
        return {}


def articles_for_queries(filename: str):
    query_database = load_query_database(filename)
    if 'prompt' not in query_database.keys():
        query_database['prompt'] = prompt1()
    prompt = query_database['prompt']
    if prompt != prompt2():
        print("Prompt has changed. Should you reprocess all queries?")

    if 'questions' not in query_database.keys():
        query_database['questions'] = {}

    for key in query_database.keys():
        if key not in ['questions', 'prompt']:
            raise ValueError(f"Key {key} is not a valid key")

    with open('data/test.csv', 'r') as f:
        queries_to_process = [line.split(',')[1] for line in f.readlines()
                              if line.split(',')[1] not in query_database['questions'].keys()]
        print(f"Processing {len(queries_to_process)} queries")
        idx = 0
        for query, article in queries_to_articles(prompt, queries_to_process):
            print(query)
            query_database['questions'][query] = article
            idx += 1
            if idx % 10 == 0:
                print(f"Completed {idx} queries")
                json.dump(query_database, open(filename, 'w'))
        json.dump(query_database, open(filename, 'w'))


if __name__ == '__main__':
    articles_for_queries(argv[1])
