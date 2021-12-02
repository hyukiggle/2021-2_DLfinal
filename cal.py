from tqdm import tqdm
train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(128),
    ])
dataset = FoodImageFolder(
  './food_data/processed_images/', transform=train_transform, txt_file='./food_data/meta/meta/train.txt')
print("\tLoaded dataset..")
loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=1,shuffle=True,num_workers=4, pin_memory=False)

mean = torch.zeros(3)
std = torch.zeros(3)

print("\tCalculating mean..")
for i, (data,_) in enumerate(tqdm(loader)):
    data = data[0].squeeze(0)
    if (i == 0): size = data.size(1) * data.size(2)
    mean += data.sum((1, 2)) / size

mean /= len(loader)
print(mean)
mean = mean.unsqueeze(1).unsqueeze(2)

print("\tCalculating std..")
for i, (data,_) in enumerate(tqdm(loader)):
    data = data[0].squeeze(0)
    std += ((data - mean) ** 2).sum((1, 2)) / size

std /= len(loader)
std = std.sqrt()
print(std)

train_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.5, 1.5)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation((-10,10)),
        transforms.Resize(128),
        transforms.Normalize(mean=[0.5443, 0.4423, 0.3395],
                                    std=[0.2571, 0.2575, 0.2613])
    ])

train_dataset = FoodImageFolder(
  './food_data/processed_images/', transform=train_transform, txt_file='./food_data/meta/meta/train.txt')

train_loader = DataLoader(train_dataset,
                          batch_size=4, shuffle=True,
                          num_workers=2, pin_memory=True)
# 학습용 이미지를 무작위로 가져오기
dataiter = iter(train_loader)
images, labels = dataiter.next()

# 이미지 보여주기
imshow(torchvision.utils.make_grid(images))
