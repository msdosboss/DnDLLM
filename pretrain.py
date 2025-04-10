import tiktoken
import torch
import matplotlib.pyplot
from matplotlib.ticker import MaxNLocator
from gptModel import GPTModel
from gptModel import genTextSimple
from tokenizer import createDataLoaderV1
import argparse
import os
from pathlib import Path
import time

def textToToken(text, tokenizer):
	encoded = tokenizer.encode(text, allowed_special = {'<|endoftext|>'})
	encodedTensor = torch.tensor(encoded).unsqueeze(0)	#add the batch dim
	return encodedTensor

def tokenToText(tokens, tokenizer):
	flat = tokens.squeeze(0)
	return tokenizer.decode(flat.tolist())

def readTextFile(filePath):
	with open(filePath, "r", encoding="utf-8") as file:
		textData = file.read()
	return textData

def createDataLoaders(textData, trainRatio, batchSize, maxLen, stride, numWorkers = 0):
	splitIdx = int(trainRatio * len(textData))
	trainLoader = createDataLoaderV1(textData[:splitIdx], batchSize = 2, maxLen = GPT_CONFIG_124M["contextLen"], stride = GPT_CONFIG_124M["contextLen"], dropLast = True, shuffle = True, numWorkers = 0)
	valLoader = createDataLoaderV1(textData[splitIdx:], batchSize = 2, maxLen = GPT_CONFIG_124M["contextLen"], stride = GPT_CONFIG_124M["contextLen"], dropLast = False, shuffle = False, numWorkers = 0)

	return trainLoader, valLoader

def convertTime(seconds):
	hours, rem = divmod(seconds, 3600)
	minutes, seconds = divmod(rem, 60)
	return int(hours), int(minutes), int(seconds)

def printEta(startTime, bookStartTime, index, totalFiles):
	bookEndTime = time.time()
	elapsedTime = bookEndTime - bookStartTime
	totalElapsedTime = bookEndTime - startTime
	booksRemaining = totalFiles - index
	averageTimePerBook = totalElapsedTime / index
	eta = averageTimePerBook * booksRemaining

	bookH, bookM, bookS = convertTime(elapsedTime)
	totalH, totalM, totalS = convertTime(totalElapsedTime)
	etaH, etaM, etaS = convertTime(eta)

	print(f"Book Processed {bookH}h {bookM}m {bookS}s" f"\nTotal time elapsed {totalH}h {totalM}m {totalS}s" f"\nETA for remaining books {etaH}h {etaM}m {etaS}s")

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
	
def generateAndPrintSample(model, tokenizer, device, startContext):
	model.eval()
	context_size = model.posEmb.weight.shape[0]
	encoded = textToToken(startContext, tokenizer).to(device)
	with torch.no_grad():
		tokenIds = genTextSimple(model, encoded, 50, context_size, 1.4, 25)
	decodedText = tokenToText(tokenIds, tokenizer)
	print(decodedText.replace("\n", " "))
	model.train()

def evaluateModel(model, trainLoader, valLoader, device, evalIter):
	model.eval()
	with torch.no_grad():
		trainLoss = calcLossLoader(trainLoader, model, device, numBatches = evalIter)
		valLoss = calcLossLoader(valLoader, model, device, numBatches = evalIter)
	model.train()
	return trainLoss, valLoss

def trainModelSimple(model, optimizer, device, numEpochs, evalFreq, evalIter, printSampleIter, startContext, outputDir, saveCkptFreq, tokenizer, files, batchSize = 1024, trainRatio = 0.90):
	trainLosses, valLosses, trackTokensSeen = [], [], []
	tokensSeen, globalStep = 0, -1
	startTime = time.time()

	try:
		for epoch in range(numEpochs):
			torch.cuda.empty_cache()
			for index, filePath in enumerate(files, 1):
				bookStartTime = time.time()
				textData = readTextFile(filePath) + " <|endoftext|> "
				print(f"Tokenizing file {index} of {totalFiles}: {filePath}")

				trainLoader, valLoader = createDataLoaders(textData, trainRatio, batchSize, maxLen = GPT_CONFIG_124M["contextLen"], stride = GPT_CONFIG_124M["contextLen"], numWorkers = 0)		

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

def plotLosses(epochsSeen, tokensSeen, trainLosses, valLosses):
	fgl, ax1 = matplotlib.pyplot.subplots(figsize=(5, 3))
	ax1.plot(epochsSeen, trainLosses, label = "Training loss")
	ax1.plot(epochsSeen, valLosses, linestyle = "-.", label = "Validation loss")
	ax1.set_xlabel("Epochs")
	ax1.set_ylabel("Loss")
	ax1.legend(loc = "upper right")
	ax1.xaxis.set_major_locator(MaxNLocator(integer = True))
	ax2 = ax1.twiny()
	ax2.plot(tokensSeen, trainLosses, alpha = 0)
	ax2.set_xlabel("Tokens seen")
	fgl.tight_layout()
	matplotlib.pyplot.show()

def saveModelAndOptimizer(model, optimizer, fileName):
	torch.save({"modelStateDict": model.state_dict(), "optimizerStateDict": optimizer.state_dict(),}, fileName)

def loadModelAndOptimizer(GPT_CONFIG, device, fileName):
	checkPoint = torch.load(fileName, map_location = device)
	model = GPTModel(GPT_CONFIG)
	model.load_state_dict(checkPoint["modelStateDict"])
	model.to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay = 0.1)
	optimizer.load_state_dict(checkPoint["optimizerStateDict"])
	model.train()
	return model, optimizer


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description = 'GPT Model Training Configuration')

	parser.add_argument('--dataDir', type = str, default = 'gutenberg/data/raw',
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

	if args.debug:
		GPT_CONFIG_124M = {
			"vocabSize": 50257,
			"contextLen": 10,
			"embDim": 12,
			"nHeads" : 2,
			"nLayers": 2,
			"dropRateEmbedding": 0.0,
			"dropRateShortcut": 0.0,
			"dropRateAttetion": 0.0,
			"qkvBias": False
		}	

	else:

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

	torch.manual_seed(123)
	device = torch.device("cuda")
	if args.isLoadModel == False:
		model = GPTModel(GPT_CONFIG_124M)
		model.to(device)
		optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay = 0.1)

	else:
		model, optimizer = loadModelAndOptimizer(GPT_CONFIG_124M, device, args.loadedModelName)

	tokenizer = tiktoken.get_encoding("gpt2")

	dataDir = args.dataDir
	allFiles = [os.path.join(path, name) for path, subdirs, files in os.walk(dataDir) for name in files if name.endswith((".txt"))]
	totalFiles = len(allFiles)

	if totalFiles == 0:
		print("No training data was found in your selected input dir")
		quit()

	print("Total Files: ", totalFiles)

	outputDir = Path(args.outputDir)
	outputDir.mkdir(parents=True, exist_ok=True)

	trainLoss, valLoss, tokensSeen = trainModelSimple(model, optimizer, device, 
		numEpochs = args.nEpochs, 
		batchSize = args.batchSize,
		evalFreq = args.evalFreq,
		evalIter = 1,
		printSampleIter = args.printSampleIter,
		outputDir = args.outputDir,
		saveCkptFreq = args.saveCkptFreq,
		startContext = "Every effort moves you",
		tokenizer = tokenizer,
		files = allFiles)

	#epochsTensor = torch.linspace(0, args.nEpochs, len(trainLoss))
	#plotLosses(epochsTensor, tokensSeen, trainLoss, valLoss)

	#torch.save(model.state_dict(), outputDir / "modelPgFinal.pth")
	saveModelAndOptimizer(model, optimizer, args.outputDir / "modelAndOptimizerPgFinal.pth")
	print(f"Maximum GPU memory allocted: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

	tokenIds = genTextSimple(model, textToToken("orca is a wonderful", tokenizer).to(device), 15, GPT_CONFIG_124M["contextLen"], 1.4, 25)
	print("Output text: \n", tokenToText(tokenIds, tokenizer))

	'''with torch.no_grad():
		trainLoss = calcLossLoader(trainLoader, model, device)
		valLoss = calcLossLoader(valLoader, model, device)

	print(trainLoss)

	#print("Output text: \n",tokenToText(tokenIds, tokenizer))'''

	'''inputs = torch.tensor([[16833, 3626, 6100],	# "every effort moves"
				[40, 1107, 568]]).to("cuda")	# "I really like"

	targets = torch.tensor([[3626, 6100, 345],	# "effort moves you"
				[1107, 568, 11311]]).to("cuda")	# "really like chocolate"

	with torch.no_grad():
		logits = model(inputs)

	probas = torch.softmax(logits, dim = -1)

	textIdx = 0
	targetProbs1 = probas[textIdx, [0, 1, 2], targets[textIdx]]
	#print("Text 1:", targetProbs1)

	textIdx = 1
	targetProbs2 = probas[textIdx, [0, 1, 2], targets[textIdx]]
	#print("Text 2:", targetProbs2)

	logProbs = torch.log(torch.cat((targetProbs1, targetProbs2)))

	avgLogProb = torch.mean(logProbs)

	negAvgLogProb = avgLogProb * -1	#this is also refered to as the cross entropy loss


	logitsFlaten = logits.flatten(0, 1)
	targetsFlaten = targets.flatten(0, 1)

	loss = torch.nn.functional.cross_entropy(logitsFlaten, targetsFlaten)
	print(loss)
	print(negAvgLogProb)'''
