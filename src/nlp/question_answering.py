from transformers import pipeline

# Initialize the question-answering pipeline with a specific model
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
gpt_pipeline = pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B")

def answer_question(question, context=None):
    if context:
        # Use the context to generate a more relevant answer
        input_text = f"Context: {context}\nQuestion: {question}"
    else:
        input_text = question

    # Use GPT-Neo model to generate an answer
    result = gpt_pipeline(input_text, max_length=50, num_return_sequences=1, top_p=0.95, temperature=0.7, truncation=True, pad_token_id=gpt_pipeline.tokenizer.eos_token_id)
    return result[0]['generated_text']