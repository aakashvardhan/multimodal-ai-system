import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class OASST1RetrievalSystem:
    def __init__(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]
        
        # Filter for root prompts (messages without a parent)
        self.root_prompts = [item for item in self.data if item['parent_id'] is None]
        
        self.prompts = [item['text'] for item in self.root_prompts]
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.prompt_vectors = self.vectorizer.fit_transform(self.prompts)
    
    def get_response(self, query, top_k=1):
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.prompt_vectors)
        top_indices = similarities.argsort()[0][-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            prompt = self.root_prompts[idx]
            response = self.get_first_assistant_response(prompt['message_id'])
            results.append({
                'prompt': prompt['text'],
                'response': response,
                'similarity': similarities[0][idx]
            })
        
        return results
    
    def get_first_assistant_response(self, parent_id):
        children = [item for item in self.data if item['parent_id'] == parent_id]
        assistant_responses = [child['text'] for child in children if child['role'] == 'assistant']
        return assistant_responses[0] if assistant_responses else "No assistant response found."

# Usage
data_path = 'path/to/oasst1_data.jsonl'  # Update this to your data file path
retrieval_system = OASST1RetrievalSystem(data_path)

# Example query
query = "What is the capital of France?"
results = retrieval_system.get_response(query, top_k=3)

for i, result in enumerate(results, 1):
    print(f"Result {i}:")
    print(f"Prompt: {result['prompt']}")
    print(f"Response: {result['response']}")
    print(f"Similarity: {result['similarity']:.4f}")
    print()