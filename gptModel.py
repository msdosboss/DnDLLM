import torch
import tiktoken
from attetion import MultiHeadAttention


GPT_CONFIG_124M = {
	"vocabSize": 50257,
	"contextLen": 1024,
	"embDim": 768,
	"nHeads" : 12,
	"nLayers": 12,
	"dropRateEmbedding": 0.1,
	"dropRateShortcut": 0.1,
	"dropRateAttetion": 0.1,
	"qkvBias": False	
}

class GPTModel(torch.nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.tokEmb = torch.nn.Embedding(cfg["vocabSize"], cfg["embDim"])
		self.posEmb = torch.nn.Embedding(cfg["contextLen"], cfg["embDim"])
		self.dropEmb = torch.nn.Dropout(cfg["dropRateEmbedding"])
		self.trfBlocks = torch.nn.Sequential(
			* [TransformerBlock(cfg) for _ in range(cfg["nLayers"])]
		)
		self.finalNorm = LayerNorm(cfg["embDim"])
		self.outHead = torch.nn.Linear(cfg["embDim"], cfg["vocabSize"], bias=False)

	def forward(self, inIdx):
		batchSize, seqLen = inIdx.shape
		tokEmbeds = self.tokEmb(inIdx)
		posEmbeds = self.posEmb(torch.arange(seqLen, device=inIdx.device))
		x = tokEmbeds + posEmbeds
		x = self.dropEmb(x)
		x = self.trfBlocks(x)
		x = self.finalNorm(x)
		logits = self.outHead(x)
		return logits

class TransformerBlock(torch.nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.att = MultiHeadAttention(
			dimIn = cfg["embDim"],
			dimOut = cfg["embDim"],
			contextLen = cfg["contextLen"],
			numHeads = cfg["nHeads"],
			dropout = cfg["dropRateAttetion"],
			qkvBias = cfg["qkvBias"]
		)
		self.ff = FeedForward(cfg)
		self.norm1 = LayerNorm(cfg["embDim"])
		self.norm2 = LayerNorm(cfg["embDim"])
		self.dropShortcut = torch.nn.Dropout(cfg["dropRateShortcut"])

	def forward(self, x):
		shortcut = x
		x = self.norm1(x)
		x = self.att(x)
		x = self.dropShortcut(x)
		x = x + shortcut

		shortcut = x
		x = self.norm2(x)
		x = self.ff(x)
		x = self.dropShortcut(x)
		x = x + shortcut

		return x

class LayerNorm(torch.nn.Module):
	def __init__(self, embDim):
		super().__init__()
		self.eps = 1e-5
		self.scale = torch.nn.Parameter(torch.ones(embDim))
		self.shift = torch.nn.Parameter(torch.zeros(embDim))

	def forward(self, x):
		mean = x.mean(dim = -1, keepdim=True)
		var = x.var(dim = -1, keepdim=True, unbiased=False)
		normX = (x - mean) / torch.sqrt(var + self.eps)
		return self.scale * normX + self.shift

class GELU(torch.nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))

class FeedForward(torch.nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.layers = torch.nn.Sequential(
			torch.nn.Linear(cfg["embDim"], cfg["embDim"] * 4),
			GELU(),
			torch.nn.Linear(cfg["embDim"] * 4, cfg["embDim"])
		)
	def forward(self, x):
		return self.layers(x)

def softmaxWithTemp(logits, temp):
	scaledLogits = logits / temp
	return torch.softmax(scaledLogits, dim=0)

def genTextSimple(model, idx, maxNewTokens, contextSize, temp = 0.0, topK = None, eosId = None):
	for _ in range(maxNewTokens):
		idxCond = idx[:, -contextSize:]
		with torch.no_grad():
			logits = model(idxCond)

		logits = logits[:, -1, :]
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
def geluVsRelu():
	import matplotlib.pyplot as plt
	gelu, relu = GELU(), torch.nn.ReLU()

	x = torch.linspace(-3, 3, 100)
	y_gelu, y_relu = gelu(x), relu(x)
	plt.figure(figsize=(8, 3))
	for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
		plt.subplot(1, 2, i)
		plt.plot(x, y)
		plt.title(f"{label} activation function")
		plt.xlabel("x")
		plt.ylabel(f"{label}(x)")
		plt.grid(True)
	plt.tight_layout()
	plt.show()

if __name__ == "__main__":
	#geluVsRelu()
	tokenizer = tiktoken.get_encoding("gpt2")
	batch = []
	txt1 = "Every effort moves you"
	txt2 = "Every day holds a"

	batch.append(torch.tensor(tokenizer.encode(txt1)))
	batch.append(torch.tensor(tokenizer.encode(txt2)))
	batch = torch.stack(batch, dim=0)
	print(batch)
	batch = batch.to("cuda")	#for some reason you have to .to on batch returns an instance of a new batch object that you then have to reasign unlike the Model dont ask me why 
	torch.manual_seed(123)
	model = GPTModel(GPT_CONFIG_124M)
	model.to("cuda")
	logits = model(batch)
	#print("Output shape: ", logits.shape)
	#print(logits)

	startContext = "Hello, I am"
	encoded = tokenizer.encode = tokenizer.encode(startContext)
	encodedTensor = torch.tensor(encoded).unsqueeze(0)
	encodedTensor = encodedTensor.to("cuda")
	model.eval()
	out = genTextSimple(model, encodedTensor, 6, GPT_CONFIG_124M["contextLen"])

	decodedText = tokenizer.decode(out.squeeze(0).tolist())
	print(decodedText)
