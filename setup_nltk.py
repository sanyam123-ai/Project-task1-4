
import nltk



#download essential NLTK datasets
print("Downloading NLTK data...")

#punkt: for sentence tokenization
nltk.download('punkt')

#stopwards: common words to filter out
nltk.download('stopwards')

#punkt_tab: Additional punctuation data
nltk.download('punkt_tab')

print("NLTK data downloaded successfully!")
