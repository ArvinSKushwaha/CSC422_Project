from obj_classify.training import Dataset

dataset = Dataset("./ModelNet10")
inv_mapping, train_data, test_data = dataset.to_data()

train_data.scramble()

for meshes, labels in train_data:
    print(meshes)
    print(labels)
