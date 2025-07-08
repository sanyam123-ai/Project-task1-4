# summarize_article.py

from transformers import pipeline
import torch
# Check device availability
device = 0 if torch.cuda.is_available() else -1

# Load summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

# Input text (can be replaced with text from a file or URL)
input_article = """
In a groundbreaking discovery, scientists have identified a new species of ancient human ancestor in a remote cave system in South Africa. 
This hominid, believed to have lived nearly 2 million years ago, shares both primitive and advanced traits, suggesting a complex path in human evolution. 
The international research team used advanced dating techniques and 3D scanning to analyze the fossils, which included skull fragments, teeth, and limb bones. 
The discovery challenges long-held assumptions about the geographic spread and diversity of early human relatives. 
Scientists hope further study will uncover clues about how these ancestors lived, adapted, and possibly interacted with other hominid species.
"""

# Summarize the article
summary = summarizer(input_article, max_length=130, min_length=30, do_sample=False)

# Display results
print("\nOriginal Article:\n", input_article)
print("\nSummarized Text:\n", summary[0]['summary_text'])
