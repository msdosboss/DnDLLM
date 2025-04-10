import os
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoConfig


def genTextSimple(model, idx, maxNewTokens, contextSize, temp = 0.0, topK = None, eosId = None):
	generated = idx
	pastKeyValue = None

	for _ in range(maxNewTokens):
		#idxCond = idx[:, -contextSize:] old slow version
		idxCond = generated[:, -1:] if pastKeyValue is not None else generated[:, -contextSize:]
		with torch.no_grad():
			outputs = model(idxCond, use_cache=True, past_key_values = pastKeyValue)

		logits = outputs[0]
		logits = logits[:, -1, :]	#hugging faces models return more than just the logits when running the forward step on a model
		pastKeyValue = outputs.past_key_values
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
		generated = torch.cat((generated, idxNext), dim = 1)
	
	generated = generated[:, idx.shape[1]:]
	return generated

def formatInput(entry):
	instructionText = f"Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n### Instruction:\n{entry['instruction']}"

	inputText = (
		f"\n\n### Instruction: \n{entry['input']}" if entry["input"] else ""
	)

	return instructionText + inputText

def evaluateModel(model, trainLoader, valLoader, device, evalIter):
	model.eval()
	with torch.no_grad():
		trainLoss = calcLossLoader(trainLoader, model, device, numBatches = evalIter)
		valLoss = calcLossLoader(valLoader, model, device, numBatches = evalIter)
	model.train()
	return trainLoss, valLoss

def calcLossBatch(inputBatch, targetBatch, model, device):
	inputBatch = inputBatch.to(device)
	targetBatch = targetBatch.to(device)
	logits = model(inputBatch)
	loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targetBatch.flatten())
	return loss

def calcLossLoader(dataLoader, model, device, numBatches = None):
	totalLoss = 0
	if len(dataLoader) == 0:
		return float("nan")
	elif numBatches == None:
		numBatches = len(dataLoader)
	else:
		numBatches = min(numBatches, len(dataLoader))
	for i, (inputBatch, targetBatch) in enumerate(dataLoader):
		if i < numBatches:
			loss = calcLossBatch(inputBatch, targetBatch, model, device)
			totalLoss += loss.item()
		else:
			break

	return totalLoss / numBatches

def trainModelSimple(model, trainLoader, valLoader, optimizer, device, numEpochs, evalFreq, evalIter, printSampleIter, startContext, outputDir, saveCkptFreq, tokenizer, files, totalFiles, batchSize = 1024, trainRatio = 0.90):
	trainLosses, valLosses, trackTokensSeen = [], [], []
	tokensSeen, globalStep = 0, -1
	startTime = time.time()

	try:
		for epoch in range(numEpochs):
			torch.cuda.empty_cache()
			print("Training ...")
			model.train()
			for inputBatch, targetBatch in trainLoader:
				optimizer.zero_grad()
				loss = calcLossBatch(inputBatch, targetBatch, model, device)
				loss.backward()	#calcs loss grad
				optimizer.step()	#updates models weights
				tokensSeen += inputBatch.numel()
				globalStep += 1

				if globalStep % evalFreq == 0:
					trainLoss, valLoss = evaluateModel(model, trainLoader, valLoader, device, evalIter)
					trainLosses.append(trainLoss)
					valLosses.append(valLoss)
					trackTokensSeen.append(tokensSeen)
					print(f"Ep {epoch + 1} (step {globalStep:06d}) : " f"Train loss {trainLoss:.3f}, " f"Val loss {valLoss:.3f}")
			
				if globalStep % printSampleIter == 0:
					generateAndPrintSample(model, tokenizer, device, startContext)

				if globalStep % saveCkptFreq == 0:
					fileName = outputDir / f"modelPg{globalStep}.pth"
					saveModelAndOptimizer(model, optimizer, fileName)
					print(f"Saved {fileName}")
			
			printEta(startTime, bookStartTime, index, totalFiles)

	except KeyboardInterrupt:
		fileName = outputDir / f"modelPg{globalStep}.pth"
		saveModelAndOptimizer(model, optimizer, fileName)
		print("saved model after Keyboard Int ")
		

	return trainLosses, valLosses, trackTokensSeen

def textToToken(text, tokenizer):
	encoded = tokenizer.encode(text)
	encodedTensor = torch.tensor(encoded).unsqueeze(0)	#add the batch dim
	return encodedTensor

def tokenToText(tokens, tokenizer):
	flat = tokens.squeeze(0)
	return tokenizer.decode(flat.tolist())

def saveModelAndOptimizer(model, optimizer, fileName):
	torch.save({"modelStateDict": model.state_dict(), "optimizerStateDict": optimizer.state_dict(),}, fileName)

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description = 'GPT Model Training Configuration')

	parser.add_argument('--dataFile', type = str, default = 'training.json',
				help = 'Directory containing training data')

	parser.add_argument('--outputDir', type = str, default = 'modelCheckpoints',
				help = 'Directory where model checkpoints will be saved')

	parser.add_argument('--nEpochs', type = int, default = 1,
				help = 'Number of epochs to train the model')

	parser.add_argument('--printSampleIter', type = int, default = 1000,
				help = 'Iterations between printing sample outputs')

	parser.add_argument('--evalFreq', type = int, default = 1000,
				help = 'Frequency of evaulations during training')

	parser.add_argument('--saveCkptFreq', type = int, default = 100000,
				help = 'Frequence of saving model checkpoints during training')

	parser.add_argument('--lr', type = float, default = 5e-4,
				help = 'Learing rate for the optimizer')

	parser.add_argument('--batchSize', type = int, default = 1024,
				help = 'Batch size for training')

	parser.add_argument('--debug', type = bool, default = False,
				help = 'Uses a very small model for debugging purposes')

	parser.add_argument('--isLoadModel', type = bool, default = False,
				help = 'Bool to decided whether we load model or not')

	parser.add_argument('--loadedModelName', type = str, default = "model.pth",
				help = 'filePath to loaded model')
	args = parser.parse_args()

	modelName = "huihui-ai/DeepSeek-R1-Distill-Llama-8B-abliterated"
	#modelName = "gpt2"
	fileName = "model.pth"

	model = AutoModelForCausalLM.from_pretrained(modelName,
		torch_dtype=torch.float16,	#quantize to 16 bit
		device_map="auto",	#allow to go on to both cpu and gpu
		trust_remote_code=True
		)

	if os.path.exists(fileName):
		checkpoint = torch.load(fileName)
		model.load_state_dict(checkpoint["modelStateDict"])
	

	tokenizer = AutoTokenizer.from_pretrained(modelName)
	optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay = 0.1)

	config = AutoConfig.from_pretrained(modelName)

	customizedCollateFn = partial(customCollate, device = device, textToToken)

	with open(args.dataFile, "r") as f:
		data = json.load(f)

	trainPortion = int(len(data) * .85)
	testPortion = int(len(data))
	valPortion = len(data) - trainPortion - testPortion

	trainData = data[:trainPortion]
	testData = data[trainPortion:testPortion]
	valData = data[trainPortion + testPortion:]

	trainDataSet = InstructionDataset(trainData, tokenizer)
	testDataSet = InstructionDataset(testData, tokenizer)
	valDataSet = InstructionDataset(valData, tokenizer)

	numWorkers = 0

	trainLoader = DataLoader(
		trainDataSet,
		batch_size = args.batchSize,
		collate_fn = customizedCollateFn,
		shuffle = True,
		drop_last = True,
		num_workers = numWorkers
	)

	testLoader = DataLoader(
		testDataSet,
		batch_size = args.batchSize,
		collate_fn = customizedCollateFn,
		shuffle = False,
		drop_last = False,
		num_workers = numWorkers
	)

	valLoader = DataLoader(
		valDataSet,
		batch_size = batchSize,
		collate_fn = customizedCollateFn,
		shuffle = False,
		drop_last = False,
		num_workers = numWorkers
	)

	trainLosses, valLosses, tokensSeen = trainModelSimple(model, trainLoader, valLoader, optimizer, device, numEpochs = args.nEpochs, evalFreq = args.evalFreq, evalIter = 5, startContext = formatInput(val_data[0]), tokenizer=tokenizer)

	inputTokens = textToToken("Write a python function to print hello world", tokenizer) 

	response = genTextSimple(model,
			idx = inputTokens,
			maxNewTokens = 100,
			contextSize = config.max_position_embeddings)

	print(tokenToText(response, tokenizer))

	torch.save({"modelStateDict": model.state_dict()}, fileName)
