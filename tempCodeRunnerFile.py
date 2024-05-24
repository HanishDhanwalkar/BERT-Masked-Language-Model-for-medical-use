mlm = MLM(data= "data/short/100sentences_eng.txt")
mlm.start()

pred, string = mlm.predict("Diagnosis [MASK] may not know you have atrial fibrillation AFib")

print(pred)
print(string)