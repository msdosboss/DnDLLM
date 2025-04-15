from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy

def embedChunck(text, model, tokenizer):
	inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
	
	with torch.no_grad():
		#lastHidden = model(**inputs).last_hidden_state
		#embedding = lastHidden[:, 0, :]	#extracting [CLS] token aka a context embedding for the whole sentence
		embedding = model(**inputs).pooler_output	#the DPR output class does not contain a cls token instead we can just use the pooling of the output tokens to create a sentence token 

	return embedding.cpu()	#have to return to the cpu because my FAISS index lib is only cpu right now

def createDatabase(embeddings):
	dim = embeddings.shape[1]
	index = faiss.IndexFlatL2(dim)
	index.add(embeddings.numpy())

	return index

if __name__ == "__main__":
	device = "cuda"

	encoderName = "facebook/dpr-ctx_encoder-single-nq-base"
	tokenizer = AutoTokenizer.from_pretrained(encoderName)
	model = AutoModel.from_pretrained(encoderName).to(device)

	documents = [
	"The capital of France is Paris.",
	"The moon orbits the Earth.",
	"PyTorch is an open-source machine learning framework.",
	"RAG uses retrieval to help generate better answers."
	]

	embeddings = torch.cat([embedChunck(doc, model, tokenizer) for doc in documents])

	index = createDatabase(embeddings)

	queryEmbedding = embedChunck("What is the capital of France", model, tokenizer)

	k = 2
	distances, indices = index.search(queryEmbedding.numpy(), k)

	topDocs = [documents[i] for i in indices[0]]

	print(topDocs)
