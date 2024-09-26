import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datasets import load_dataset


class OASST1RetrievalSystem:
    def __init__(self, data_path):
        self.data = data_path["train"].to_pandas()

        # Filter for root prompts (messages without a parent)
        self.root_prompts = self.data[self.data["parent_id"].isna()].to_dict("records")

        self.prompts = [item["text"] for item in self.root_prompts]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.prompt_vectors = self.vectorizer.fit_transform(self.prompts)

    def get_response(self, query, top_k=1):
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.prompt_vectors)
        top_indices = similarities.argsort()[0][-top_k:][::-1]

        results = []
        for idx in top_indices:
            prompt = self.root_prompts[idx]
            response = self.get_first_assistant_response(
                prompt.get("id") or prompt.get("message_id")
            )
            results.append(
                {
                    "prompt": prompt["text"],
                    "response": response,
                    "similarity": similarities[0][idx],
                    "message_id": prompt.get("id")
                    or prompt.get("message_id"),  # Include message_id for debugging
                }
            )

        return results

    def get_first_assistant_response(self, parent_id):
        # Find all children of the given parent_id
        children = self.data[self.data["parent_id"] == parent_id]

        # Filter for assistant responses
        assistant_responses = children[children["role"] == "assistant"]["text"].tolist()

        # Return the first assistant response, or a default message if none found
        return (
            assistant_responses[0]
            if assistant_responses
            else "No assistant response found."
        )


# Usage

ds = load_dataset("OpenAssistant/oasst1")
retrieval_system = OASST1RetrievalSystem(ds)

# Example query
query = "What is astrophotography?"
results = retrieval_system.get_response(query, top_k=3)

for i, result in enumerate(results, 1):
    print(f"Result {i}:")
    print(f"Prompt: {result['prompt']}")
    print(f"Response: {result['response']}")
    print(f"Similarity: {result['similarity']:.4f}")
    print()
