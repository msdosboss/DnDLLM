from transformers import AutoTokenizer, AutoModel
import argparse
import os
import pickle
import torch
import faiss
import re


def cleanText(text):
    text = text.replace('\n', ' ').strip()
    text = re.sub(r'\s+', ' ', text)  # collapse repeated spaces
    return text


def parseTextFiles(dir="data/htmls/"):
    i = 0
    allChunks = []
    while True:
        fileName = dir + str(i) + ".txt"
        if not os.path.exists(fileName):
            print(fileName + " does not exist")
            break
        with open(fileName, "r") as f:
            raw = f.read()
        fileChunks = raw.split("|||")
        # fileChunks = fileChunks[10:]  # Cuts out the first 10 entries this might not be the best solution
        allChunks.append(fileChunks)
        i += 1

    flatChunks = [cleanText(chunk) for fileChunks in allChunks for chunk in fileChunks]
    # print(flatChunks)
    return flatChunks


def lastTokenPool(last_hidden_states, attention_mask):  # This function is from https://huggingface.co/infly/inf-retriever-v1-1.5b
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]



def embedChunck(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        lastHidden = model(**inputs).last_hidden_state
        embedding = lastTokenPool(lastHidden, inputs['attention_mask'])
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        # embedding = lastHidden[:, 0, :]  # extracting [CLS] token aka a context embedding for the whole sentence
        # embedding = model(**inputs).pooler_output	#the DPR output class does not contain a cls token instead we can just use the pooling of the output tokens to create a sentence token

    return embedding.cpu()  # have to return to the cpu because my FAISS index lib is only cpu right now


def createAndSaveDatabase(model, tokenizer, chunks, device):
    embeddings = torch.cat([embedChunck(chunk, model, tokenizer, device) for chunk in chunks])
    index = createDatabase(embeddings)
    saveDatabase(index, chunks)

    return index


def createDatabase(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.numpy())

    return index


def saveDatabase(index, chunks):
    faiss.write_index(index, "RAGDatabase/index.faiss")

    with open("RAGDatabase/text.pkl", "wb") as f:
        pickle.dump(chunks, f)


def loadDatabaseAndText(indexName, pickleFileName):
    index = faiss.read_index(indexName)
    with open(pickleFileName, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks


def normalizeChunks(chunks, sizeThreshold=512):
    normalizedChunks = []
    currentChunk = ""
    for chunk in chunks:
        if len(currentChunk.split()) + len(chunk.split()) <= sizeThreshold:
            currentChunk = currentChunk + " " + chunk
        else:
            normalizedChunks.append(currentChunk.strip())
            currentChunk = chunk

    if currentChunk:
        normalizedChunks.append(currentChunk.strip())

    return normalizedChunks


def queryDatabase(query, model, tokenizer, index, chunks, k, device):
    queryEmbedding = embedChunck(query, model, tokenizer, device)

    distances, indices = index.search(queryEmbedding.numpy(), k)

    topChunks = [chunks[i] for i in indices[0]]
    return topChunks


def ragIntoPrompt(prompt, model, tokenizer, index, chunks, device, topK=50, sizeThreshold=1024):
    topChunks = queryDatabase(prompt, model, tokenizer, index, chunks, topK, device)
    i = 0
    ragEntries = ""
    while len(ragEntries.split()) < sizeThreshold:
        ragEntries += " " + topChunks[i]
        i += 1

    return "Question:\n" + prompt + "\nRAG information:\n" + ragEntries


def loadRagModelAndTokenizer(encoderName="infly/inf-retriever-v1-1.5b", device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(encoderName)
    model = AutoModel.from_pretrained(encoderName).to(device)

    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RAG Configurations')

    parser.add_argument('--create', type=bool, default=False,
                        help='if true will create new database')
    args = parser.parse_args()

    device = "cuda"

    model, tokenizer = loadRagModelAndTokenizer()

    if args.create is True:
        chunks = parseTextFiles()
        # chunks = normalizeChunks(chunks)
        index = createAndSaveDatabase(model, tokenizer, chunks, device)
    else:
        index, chunks = loadDatabaseAndText("RAGDatabase/index.faiss", "RAGDatabase/text.pkl")

    size_modifier_text = "Large creatures take an –1 penalty to AC due to their size. See: Size Modifiers table."

    print("chunk embedding:")
    print(embedChunck(size_modifier_text, model, tokenizer, device))

    print("prompt embedding:")
    print(embedChunck("Creatures that are large have what happen to AC?", model, tokenizer, device))

    topChunks = queryDatabase("Describe to me how touch attacks work", model, tokenizer, index, chunks, 25, device)
    print(topChunks)
