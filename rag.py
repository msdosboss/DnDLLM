from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy

def embedChunck(text, model, tokenizer):
	inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
	
	with torch.no_grad():
		lastHidden = model(**inputs).last_hidden_state
		embedding = lastHidden[:, 0, :]	#extracting [CLS] token aka a context embedding for the whole sentence
		#embedding = model(**inputs).pooler_output	#the DPR output class does not contain a cls token instead we can just use the pooling of the output tokens to create a sentence token 

	return embedding.cpu()	#have to return to the cpu because my FAISS index lib is only cpu right now

def createDatabase(embeddings):
	dim = embeddings.shape[1]
	index = faiss.IndexFlatL2(dim)
	index.add(embeddings.numpy())

	return index

if __name__ == "__main__":
	device = "cuda"

	encoderName = "infly/inf-retriever-v1-1.5b"
	#encoderName = "facebook/dpr-ctx_encoder-single-nq-base"
	tokenizer = AutoTokenizer.from_pretrained(encoderName)
	model = AutoModel.from_pretrained(encoderName).to(device)

	documents = [
	"The capital of France is Paris.",
	"The moon orbits the Earth.",
	"PyTorch is an open-source machine learning framework.",
	"RAG uses retrieval to help generate better answers.",
	"Pin a target for time",
	"As a standard action, you can attempt to grapple a foe, hindering his combat options. If you do not have Improved Grapple, grab, or a similar ability, attempting to grapple a foe provokes an attack of opportunity from the target of your maneuver. Humanoid creatures without two free hands attempting to grapple a foe take a –4 penalty on the combat maneuver roll. If successful, both you and the target gain the grappled condition. If you successfully grapple a creature that is not adjacent to you, move that creature to an adjacent open space (if no space is available, your grapple fails). Although both creatures have the grappled condition, you can, as the creature that initiated the grapple, release the grapple as a free action, removing the condition from both you and the target. If you do not release the grapple, you must continue to make a check each round, as a standard action, to maintain the hold. If your target does not break the grapple, you get a +5 circumstance bonus on grapple checks made against the same target in subsequent rounds. Once you are grappling an opponent, a successful check allows you to continue grappling the foe, and also allows you to perform one of the following actions (as part of the standard action spent to maintain the grapple).\nMove \n You can move both yourself and your target up to half your speed. At the end of your movement, you can place your target in any square adjacent to you. If you attempt to place your foe in a hazardous location, such as in a wall of fire or over a pit, the target receives a free attempt to break your grapple with a +4 bonus. \nDamage \nYou can inflict damage to your target equal to your unarmed strike, a natural attack, or an attack made with armor spikes or a light or one-handed weapon. This damage can be either lethal or nonlethal. \nPin \nYou can give your opponent the pinned condition (see Conditions). Despite pinning your opponent, you still only have the grappled condition, but you lose your Dexterity bonus to AC. \nTie Up \n If you have your target pinned, otherwise restrained, or unconscious, you can use rope to tie him up. This works like a pin effect, but the DC to escape the bonds is equal to 20 + your Combat Maneuver Bonus (instead of your CMD). The ropes do not need to make a check every round to maintain the pin. If you are grappling the target, you can attempt to tie him up in ropes, but doing so requires a combat maneuver check at a –10 penalty. If the DC to escape from these bindings is higher than 20 + the target’s CMB, the target cannot escape from the bonds, even with a natural 20 on the check. \nIf You Are Grappled \n If you are grappled, you can attempt to break the grapple as a standard action by making a combat maneuver check (DC equal to your opponent’s CMD; this does not provoke an attack of opportunity) or Escape Artist check (with a DC equal to your opponent’s CMD). If you succeed, you break the grapple and can act normally."
	]

	embeddings = torch.cat([embedChunck(doc, model, tokenizer) for doc in documents])

	index = createDatabase(embeddings)

	queryEmbedding = embedChunck("What is grapple", model, tokenizer)

	k = 2
	distances, indices = index.search(queryEmbedding.numpy(), k)

	topDocs = [documents[i] for i in indices[0]]

	print(topDocs)
