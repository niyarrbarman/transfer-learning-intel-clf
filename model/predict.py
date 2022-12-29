from model import ClassifierModel, Data
from imports import *

def view_classify(img, ps):
    
    classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

    ps = ps.data.cpu().numpy().squeeze()
    img = img.numpy().transpose(1,2,0)
   
    fig, (ax1, ax2) = plt.subplots(figsize=(10,8), ncols=2)
    ax1.imshow(img)
    ax1.axis('off')
    ax2.barh(classes, ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(classes)
    ax2.set_yticklabels(classes)
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()
    plt.savefig("output.jpg")

    return None
  
if __name__ == "__main__":
    
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRAIN_DIR = ENTER-TRAINING-DIRECTORY
    VAL_DIR = ENTER-VALIDATION-DIRECTORY
    
    model = ClassifierModel().to(DEVICE)
    model.load_state_dict(torch.load("/content/best-weights.pt"))
    trainLoader, valLoader, trainset, validset = Data(train_dir=TRAIN_DIR, val_dir=VAL_DIR, batch_size=BATCH_SIZE).load()

    image, label = validset[np.random.randint(0, len(validset))]

    image = image.unsqueeze(0)
    logits = model(image.to(DEVICE))
    probs = nn.Softmax(dim=1)(logits)

    view_classify(image.squeeze(), probs)
