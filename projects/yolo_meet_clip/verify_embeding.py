import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


num_cls = 80
num_sample = 10000
clip_text_embeding = 512
logit_scale_init_value = 2.6592

bs = 100
# learning_rate = 1e-1
learning_rate = 0.5
epochs = 10000

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, num_sample, num_cls, clip_text_embeding):
        # text_embeds = torch.load('text_embeds.pth').requires_grad_(False)
        text_embeds = torch.load('text_embeds.pth')

        gts = torch.randint(0, num_cls, (num_sample,))
        # to prob
        inputs = torch.zeros((num_sample, num_cls))
        gt_embeds = torch.zeros((num_sample, clip_text_embeding))
        for i, gt in enumerate(gts):
            x = torch.rand((1, num_cls))
            x[:, gt] = 1
            inputs[i] = x
            gt_embeds[i] = text_embeds[gt]

        self.gt_embeds = gt_embeds
        self.inputs = torch.softmax(inputs, dim=1)

    def __len__(self):
        return len(self.gt_embeds)

    def __getitem__(self, idx):
        return self.inputs[idx], self.gt_embeds[idx]


class SimpleModel(nn.Module):
    def __init__(self, num_cls, clip_text_embeding, logit_scale_init_value):
        super().__init__()
        self.visual_projection = nn.Linear(num_cls, clip_text_embeding, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init_value)

        nn.init.normal_(self.visual_projection.weight, std=clip_text_embeding ** -0.5 * 1.0)

    def forward(self, x, gt_embeds=None):
        image_embeds = self.visual_projection(x)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()

        logits_per_image = logit_scale * image_embeds @ gt_embeds.t()
        probs = logits_per_image.softmax(dim=1)

        if self.training:
            logits_per_text = logits_per_image.t()
            loss = clip_loss(logits_per_text)
            return loss, probs
        else:
            return probs


model = SimpleModel(num_cls, clip_text_embeding, logit_scale_init_value).cuda()
model.train()

dataset = SimpleDataset(num_sample, num_cls, clip_text_embeding)
train_dataloader = DataLoader(dataset, batch_size=bs)

test_dataset = SimpleDataset(1000, num_cls, clip_text_embeding)
test_dataloader = DataLoader(test_dataset, batch_size=bs)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    for i, (x, gt_embeds) in enumerate(train_dataloader):
        loss, prob = model(x.cuda(), gt_embeds.cuda())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 20 == 0:
            print(f'epoch: {epoch}, i: {i}, loss: {loss.item()}')
            pred = torch.argmax(prob, dim=1)
            gt = torch.argmax(x, dim=1).cuda()
            correct = (pred == gt).sum().item()
            print(f'epoch: {epoch}, train acc: {correct / len(gt)}')

    model.eval()
    correct = 0
    for i, (x, _) in enumerate(test_dataloader):
        gt_embeds = torch.load('text_embeds.pth')
        probs = model(x.cuda(), gt_embeds.cuda())
        pred = torch.argmax(probs, dim=1)
        gt = torch.argmax(x, dim=1).cuda()
        correct += (pred == gt).sum().item()

    print(f'epoch: {epoch}, val acc: {correct / len(test_dataset)}')
    print('===========================')

    model.train()
    dataset = SimpleDataset(num_sample, num_cls, clip_text_embeding)
    train_dataloader = DataLoader(dataset, batch_size=bs)
