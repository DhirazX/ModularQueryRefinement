from nlp_utils.preprocessor import preprocess

# Read data from "data.txt"
with open("data.txt" , "r", encoding="utf-8") as file:
    text = file.read()

# Preprocess text
preprocessed_text = preprocess(text)

# Print the result
print("Preprocessed Text: ", preprocessed_text)