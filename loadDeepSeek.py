import torch
import json
import argparse
from functools import partial
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoConfig


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encodedTexts = []
        for entry in data:
            instructionAndInput = formatInput(entry)
            responseText = entry['output']
            fullText = instructionAndInput + responseText
            self.encodedTexts.append(tokenizer.encode(fullText))

    def __getitem__(self, index):
        return self.encodedTexts[index]

    def __len__(self):
        return len(self.data)


def customCollate(batch,
                  padTokenId=50256,  # this needs to be the tokenized version of "<|endoftext|>"
                  ignoreIndex=-100,
                  allowedMaxLen=None,
                  device="cuda"):
    batchMaxLen = max(len(item) + 1 for item in batch)

    inputsList, targetsList = []

    for item in batch:
        newItem = item.copy()
        newItem += [padTokenId]

        padded = (
            newItem + [padTokenId] * (batchMaxLen - len(newItem))
        )
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == padTokenId
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignoreIndex

        if allowedMaxLen is not None:
            targets = targets[:allowedMaxLen]
            inputs = inputs[:allowedMaxLen]

        inputsList.append(inputs)
        targetsList.append(targets)

    inputsTensor = torch.stack(inputsList).to(device)
    targetsTensor = torch.stack(targetsList).to(device)
    return inputsTensor, targetsTensor


def genTextSimple(model, idx, maxNewTokens, contextSize, temp=0.0, topK=None, eosId=None):
    batchSize, initLen = idx.shape
    totalLen = initLen + maxNewTokens

    pastKeyValue = None
    curPos = initLen

    output = torch.empty((batchSize, totalLen), dtype=idx.dtype, device=idx.device)
    output[:, :initLen] = idx

    for _ in range(maxNewTokens):
        # idxCond = idx[:, -contextSize:] old slow version
        idxCond = output[:, curPos - 1:curPos] if pastKeyValue is not None else output[:, curPos - contextSize:curPos]
        with torch.no_grad():
            outputs = model(idxCond, use_cache=True, past_key_values=pastKeyValue)

        logits = outputs[0]
        logits = logits[:, -1, :]  # hugging faces models return more than just the logits when running the forward step on a model
        pastKeyValue = outputs.past_key_values
        if topK is not None:
            topLogits, topPos = torch.topk(logits, topK)
            logits = torch.where(condition=logits < topLogits[:, -1], input=torch.tensor(float('-inf')).to(logits.device), other=logits)

        if temp != 0.0:
            logits = logits / temp
            probas = torch.softmax(logits, dim=-1)
            idxNext = torch.multinomial(probas, num_samples=1)
        else:
            idxNext = torch.argmax(logits, dim=-1, keepdim=True)

        if idxNext.item() == eosId:
            break

        # print(idxNext)
        output[:, curPos] = idxNext.squeeze(1)
        curPos += 1

    return output[:, initLen:curPos]


def generateAndPrintSample(model, tokenizer, device, startContext):
    model.eval()
    context_size = model.posEmb.weight.shape[0]
    encoded = textToToken(startContext, tokenizer).to(device)
    with torch.no_grad():
        tokenIds = genTextSimple(model, encoded, 50, context_size, 1.4, 25)
    decodedText = tokenToText(tokenIds, tokenizer)
    print(decodedText.replace("\n", " "))
    model.train()


def formatInput(entry):
    instructionText = f"Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n### Instruction:\n{entry['instruction']}"

    inputText = (
        f"\n\n### Instruction: \n{entry['input']}" if entry["input"] else ""
    )

    return instructionText + inputText


def evaluateModel(model, trainLoader, valLoader, device, evalIter):
    model.eval()
    with torch.no_grad():
        trainLoss = calcLossLoader(trainLoader, model, device, numBatches=evalIter)
        valLoss = calcLossLoader(valLoader, model, device, numBatches=evalIter)
    model.train()
    return trainLoss, valLoss


def calcLossBatch(inputBatch, targetBatch, model, device):
    inputBatch = inputBatch.to(device)
    targetBatch = targetBatch.to(device)
    logits = model(inputBatch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targetBatch.flatten())
    return loss


def calcLossLoader(dataLoader, model, device, numBatches=None):
    totalLoss = 0
    if len(dataLoader) == 0:
        return float("nan")
    elif numBatches is None:
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


def trainModelSimple(model, trainLoader, valLoader, optimizer, device, numEpochs, evalFreq, evalIter, printSampleIter, startContext, outputDir, saveCkptFreq, tokenizer, batchSize=1024, trainRatio=0.90):
    trainLosses, valLosses, trackTokensSeen = [], [], []
    tokensSeen, globalStep = 0, -1

    try:
        for epoch in range(numEpochs):
            torch.cuda.empty_cache()
            print("Training ...")
            model.train()
            for inputBatch, targetBatch in trainLoader:
                optimizer.zero_grad()
                loss = calcLossBatch(inputBatch, targetBatch, model, device)
                loss.backward()  # calcs loss grad
                optimizer.step()  # updates models weights
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

    except KeyboardInterrupt:
        fileName = outputDir / f"modelPg{globalStep}.pth"
        saveModelAndOptimizer(model, optimizer, fileName)
        print("saved model after Keyboard Int ")

    return trainLosses, valLosses, trackTokensSeen


def textToToken(text, tokenizer):
    encoded = tokenizer.encode(text)
    encodedTensor = torch.tensor(encoded).unsqueeze(0)  # add the batch dim
    return encodedTensor


def tokenToText(tokens, tokenizer):
    flat = tokens.squeeze(0)
    return tokenizer.decode(flat.tolist())


def saveModelAndOptimizer(model, optimizer, fileName):
    torch.save({"modelStateDict": model.state_dict(), "optimizerStateDict": optimizer.state_dict(), }, fileName)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='GPT Model Training Configuration')

    parser.add_argument('--dataFile', type=str, default='training.json',
                        help='Directory containing training data')

    parser.add_argument('--outputDir', type=str, default='modelCheckpoints',
                        help='Directory where model checkpoints will be saved')

    parser.add_argument('--nEpochs', type=int, default=1,
                        help='Number of epochs to train the model')

    parser.add_argument('--printSampleIter', type=int, default=1000,
                        help='Iterations between printing sample outputs')

    parser.add_argument('--evalFreq', type=int, default=1000,
                        help='Frequency of evaulations during training')

    parser.add_argument('--saveCkptFreq', type=int, default=100000,
                        help='Frequence of saving model checkpoints during training')

    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learing rate for the optimizer')

    parser.add_argument('--batchSize', type=int, default=1024,
                        help='Batch size for training')

    parser.add_argument('--debug', type=bool, default=False,
                        help='Uses a very small model for debugging purposes')

    parser.add_argument('--isLoadModel', type=bool, default=False,
                        help='Bool to decided whether we load model or not')

    parser.add_argument('--loadedModelName', type=str, default="model.pth",
                        help='filePath to loaded model')
    args = parser.parse_args()

    # modelName = "huihui-ai/DeepSeek-R1-Distill-Qwen-7B-abliterated-v2"
    # modelName = "gpt2"
    modelName = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    fileName = "model.pth"
    device = "cuda"

    model = AutoModelForCausalLM.from_pretrained(modelName,
                                                 torch_dtype=torch.float16,  # quantize to 16 bit
                                                 device_map="cuda",  # allow to go on to both cpu and gpu
                                                 trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(modelName)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    config = AutoConfig.from_pretrained(modelName)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})

    customizedCollateFn = partial(customCollate, device=device, padTokenId=tokenizer.pad_token_id)

    with open(args.dataFile, "r") as f:
        data = json.load(f)

    trainPortion = int(len(data) * 0.85)
    testPortion = int(len(data) * 0.1)
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
        batch_size=args.batchSize,
        collate_fn=customizedCollateFn,
        shuffle=True,
        drop_last=True,
        num_workers=numWorkers
    )

    testLoader = DataLoader(
        testDataSet,
        batch_size=args.batchSize,
        collate_fn=customizedCollateFn,
        shuffle=False,
        drop_last=False,
        num_workers=numWorkers
    )

    valLoader = DataLoader(
        valDataSet,
        batch_size=args.batchSize,
        collate_fn=customizedCollateFn,
        shuffle=False,
        drop_last=False,
        num_workers=numWorkers
    )

    trainLosses, valLosses, tokensSeen = trainModelSimple(model, trainLoader, valLoader, optimizer, device, numEpochs=args.nEpochs, evalFreq=args.evalFreq, evalIter=5, startContext=formatInput(valData[0]), tokenizer=tokenizer, printSampleIter=args.printSampleIter, outputDir=args.outputDir, saveCkptFreq=args.saveCkptFreq)

    print(tokenizer.eos_token_id)

    # inputTokens = textToToken("### Instruction: \nWrite a simple C program that prints Hello, World! to the console. Keep it basic and do not include file operations or unnecessary complexity.\n### Response:", tokenizer).to(device)
    inputTokens = textToToken("### Instruction: \nWrite a square root function in C <think> </think>?\n### Response:", tokenizer).to(device)

    response = genTextSimple(model,
                             idx=inputTokens,
                             maxNewTokens=1000,
                             contextSize=config.max_position_embeddings,
                             eosId=tokenizer.eos_token_id,
                             temp=.95,
                             topK=50)

    print(tokenToText(response, tokenizer))
