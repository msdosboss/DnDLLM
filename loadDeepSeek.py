import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoConfig
#from gptModel import genTextSimple


def genTextSimple(model, idx, maxNewTokens, contextSize, temp = 0.0, topK = None, eosId = None):
	for _ in range(maxNewTokens):
		idxCond = idx[:, -contextSize:]
		with torch.no_grad():
			outputs = model(idxCond)

		logits = outputs[0]
		logits = logits[:, -1, :]	#hugging faces models return more than just the logits when running the forward step on a model
		if topK is not None:
			topLogits, topPos = torch.topk(logits, topK)
			logits = torch.where(condition = logits < topLogits[:, -1], input = torch.tensor(float('-inf')).to(logits.device), other=logits)

		if temp != 0.0:
			logits = logits / temp
			probas = torch.softmax(logits, dim = -1)
			idxNext = torch.multinomial(probas, num_samples = 1)
		else:
			idxNext = torch.argmax(logits, dim = -1, keepdim = True)

		if idxNext == eosId:
			break
		idx = torch.cat((idx, idxNext), dim = 1)

	return idx


def textToToken(text, tokenizer):
	encoded = tokenizer.encode(text)
	encodedTensor = torch.tensor(encoded).unsqueeze(0)	#add the batch dim
	return encodedTensor

def tokenToText(tokens, tokenizer):
	flat = tokens.squeeze(0)
	return tokenizer.decode(flat.tolist())


if __name__ == "__main__":

	modelName = "huihui-ai/DeepSeek-R1-Distill-Qwen-14B-abliterated-v2"
	#modelName = "gpt2"

	model = AutoModelForCausalLM.from_pretrained(modelName,
		torch_dtype=torch.float16,	#quantize to 16 bit
		device_map="auto",	#allow to go on to both cpu and gpu
		trust_remote_code=True
		)
	tokenizer = AutoTokenizer.from_pretrained(modelName)

	config = AutoConfig.from_pretrained(modelName)

	inputTokens = textToToken("Hello, who are you?", tokenizer) 

	response = genTextSimple(model,
			idx = inputTokens,
			maxNewTokens = 100,
			contextSize = config.max_position_embeddings)

	print(tokenToText(response, tokenizer))
