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


def embedChunck(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        lastHidden = model(**inputs).last_hidden_state
        embedding = lastHidden.mean(dim=1)
        # embedding = lastHidden[:, 0, :]  # extracting [CLS] token aka a context embedding for the whole sentence
        # embedding = model(**inputs).pooler_output	#the DPR output class does not contain a cls token instead we can just use the pooling of the output tokens to create a sentence token

    return embedding.cpu()  # have to return to the cpu because my FAISS index lib is only cpu right now


def createAndSaveDatabase(model, tokenizer, chunks):
    embeddings = torch.cat([embedChunck(chunk, model, tokenizer) for chunk in chunks])
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RAG Configurations')

    parser.add_argument('--create', type=bool, default=False,
                        help='if true will create new database')
    args = parser.parse_args()

    device = "cuda"

    encoderName = "infly/inf-retriever-v1-1.5b"
    # encoderName = "facebook/dpr-ctx_encoder-single-nq-base"
    tokenizer = AutoTokenizer.from_pretrained(encoderName)
    model = AutoModel.from_pretrained(encoderName).to(device)

    if args.create is True:
        chunks = parseTextFiles()
        # chunks = normalizeChunks(chunks)
        index = createAndSaveDatabase(model, tokenizer, chunks)
    else:
        index, chunks = loadDatabaseAndText("RAGDatabase/index.faiss", "RAGDatabase/text.pkl")

    queryEmbedding = embedChunck("rules for grappling in Pathfinder? actions lost", model, tokenizer)

    k = 5
    distances, indices = index.search(queryEmbedding.numpy(), k)

    topChunks = [chunks[i] for i in indices[0]]

    print(topChunks)
